from typing import Optional

import mlflow
import torch
from geoopt.manifolds import PoincareBallExact
from geoopt.optim import RiemannianSGD
from geoopt.tensor import ManifoldParameter
from torch.nn.functional import relu
from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, LRScheduler
from torch.utils.data import DataLoader

from .base import BaseEmbedding
from .poincare_embedding import PoincareEmbedding
from .utils.clone_optimizer import clone_one_group_optimizer
from .utils.eval_tools import evaluate_edge_predictions

# from .utils.clone_optimizer import clone_one_group_optimizer
# from .utils.eval_tools import evaluate_edge_predictions

def get_repulsion_weight(epoch: int, warmup_epochs: int = 50, start_weight: float = 0.001, final_weight: float = 1.0):
    if epoch < warmup_epochs:
        # Linear interpolation from start_weight to final_weight
        alpha = epoch / warmup_epochs
        return (1 - alpha) * start_weight + alpha * final_weight
    else:
        return final_weight


def tangent_space_repulsion_loss(points, ball, k=50, eps=1e-6):
    tangent_points = ball.logmap0(points)  # shape: [n, d]

    norms = tangent_points.pow(2).sum(dim=1, keepdim=True)  # [n, 1]
    dists_sq = norms + norms.t() - 2 * tangent_points @ tangent_points.t()  # [n, n]
    dists_sq = dists_sq.clamp(min=eps)

    dists_sq.fill_diagonal_(float("inf"))

    topk = torch.topk(dists_sq, k=k, largest=False).values  # [n, k]

    # Step 5: Apply repulsion penalty (inverse squared dist)
    loss = (1.0 / topk).mean()
    return loss



def xi(parent: torch.Tensor, child: torch.Tensor, dim: int = -1) -> torch.Tensor:
    parent_norm = parent.norm(dim=dim)
    parent_norm_sq = parent_norm.square()
    child_norm = child.norm(dim=dim)
    child_norm_sq = child_norm.square()

    parent_dot_child = torch.einsum("bij,bij->bi", parent, child)

    numerator = (
        parent_dot_child * (1 + parent_norm_sq)
        - parent_norm_sq * (1 + child_norm_sq)
    )
    denominator = (
        parent_norm * (parent - child).norm(dim=dim)
        * (1 + parent_norm_sq * child_norm_sq - 2 * parent_dot_child).sqrt()
    )

    return (numerator / denominator.clamp_min(1e-15)).clamp(min=-1 + 1e-5, max=1 - 1e-5).arccos()


def psi(x: torch.Tensor, K: float = 0.1, dim: int = -1) -> torch.Tensor:
    x_norm = x.norm(dim=dim)
    arcsin_arg = K * (1 - x_norm.square()) / x_norm.clamp_min(1e-15)
    return arcsin_arg.clamp(min=-1 + 1e-5, max=1 - 1e-5).arcsin()


def energy(
    parent_embeddings: torch.Tensor, child_embeddings, K: float = 0.1
) -> torch.Tensor:
    xi_angles = xi(parent=parent_embeddings, child=child_embeddings, dim=-1)
    psi_parent = psi(x=parent_embeddings, K=K, dim=-1)
    return relu(xi_angles - psi_parent)

def hyperbolic_entailment_cone_loss(
    energies: torch.Tensor, targets: torch.Tensor, margin: float = 0.01
) -> torch.Tensor:
    losses = torch.where(
        condition=targets, input=energies, other=relu(margin - energies)
    ).sum(dim=-1)
    return losses.mean()


class EntailmentConeEmbedding(BaseEmbedding):
    def __init__(
        self, num_embeddings: int, embedding_dim: int, ball: PoincareBallExact, K: float = 0.1
    ) -> None:
        super(EntailmentConeEmbedding, self).__init__(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            ball=ball,
        )
        self.K = K
        self.inner_radius = 2 * K / (1 + (1 + 4 * K ** 2) ** 0.5)

    def forward(self, edges: torch.Tensor) -> torch.Tensor:
        embeddings = super(EntailmentConeEmbedding, self).forward(edges)
        energies = energy(
            parent_embeddings=embeddings[:, :, 0, :],
            child_embeddings=embeddings[:, :, 1, :],
            K=self.K,
        )
        return energies

    def score(self, edges: torch.Tensor, alpha: float = 1) -> torch.Tensor:
        """
        Score function used for predicting directed edges during evaluation.
        Trivial for entailment cones, but not for Poincare embeddings. Note that **_kwargs
        catches unused keywords such as the alpha from Poincare embeddings.
        """
        embeddings = super(EntailmentConeEmbedding, self).forward(edges)
        embedding_norms = embeddings.norm(dim=-1)
        edge_distances = self.ball.dist(x=embeddings[:, :, 0, :], y=embeddings[:, :, 1, :])
        return - (
            1 + alpha * (embedding_norms[:, :, 0] - embedding_norms[:, :, 1])
        ) * edge_distances

    def train_model(
        self,
        dataloader: DataLoader,
        epochs: int,
        optimizer: Optimizer,
        scheduler: Optional[LRScheduler] = None,
        pretrain_epochs: int = 50,
        pretrain_lr: float = 0.1,
        margin: float = 0.01,
        burn_in_epochs: int = 10,
        burn_in_lr_mult: float = 0.1,
        **kwargs
    ):
        mlflow.log_params({
            "pretrain_epochs": pretrain_epochs,
            "pretrain_lr": pretrain_lr,
            "pretrain_scheduler": "consine annealing",
            "pretrain_scheduler_min": 1e-2,
        })

        # Initialize a Poincare embeddings model for pretraining
        poincare_embeddings = PoincareEmbedding(
            num_embeddings=self.num_embeddings,
            embedding_dim=self.embedding_dim,
            ball=self.ball,
        )

        pretraining_optimizer = RiemannianSGD(
            params=poincare_embeddings.parameters(),
            lr=pretrain_lr,
            momentum=0.9,
            dampening=0,
            weight_decay=0.0005,
            nesterov=True,
            stabilize=500
        )

        # Perform pretraining
        poincare_embeddings.train_model(
            dataloader=dataloader,
            epochs=pretrain_epochs,
            optimizer=pretraining_optimizer,
            scheduler=CosineAnnealingLR(pretraining_optimizer, T_max=pretrain_epochs, eta_min=1e-2),
            burn_in_epochs=burn_in_epochs,
            burn_in_lr_mult=burn_in_lr_mult,
        )

        print("Finished pretraining")

        torch.autograd.set_detect_anomaly(True)

        # Copy pretrained embeddings, rescale and clip these and reset optimizer param group
        with torch.no_grad():
            self.weight.copy_(poincare_embeddings.weight)
            self.weight.mul_(0.8)
            self._clip_embeddings()
            optimizer = clone_one_group_optimizer(
                optimizer=optimizer,
                new_params=self.parameters(),
            )

        for epoch in range(epochs):
            avg_total_loss = 0
            avg_repulsion_loss = 0
            avg_entailment_loss = 0

            for batch in dataloader:
                edges = batch["edges"].to(self.weight.device)
                edge_label_targets = batch["edge_label_targets"].to(self.weight.device)

                optimizer.zero_grad()

                energies = self(edges=edges)
                ent_loss = hyperbolic_entailment_cone_loss(
                    energies=energies, targets=edge_label_targets, margin=margin
                )

                repulsion_l = tangent_space_repulsion_loss(self.weight, self.ball) * get_repulsion_weight(epoch) * 0

                loss = ent_loss + repulsion_l

                loss.backward()
                optimizer.step()

                avg_total_loss += loss.item()
                avg_repulsion_loss += repulsion_l.item()
                avg_entailment_loss += ent_loss.item()

                with torch.no_grad():
                    self.weight.data = self.ball.projx(self.weight.data)

            plr = optimizer.param_groups[0]["lr"]

            print(f"Epoch {epoch + 1}:  {avg_total_loss/len(dataloader)}, lr: {plr}")

            norms = self.weight.norm(dim=1)

            mean_norm = norms.mean().item()
            max_norm = norms.max().item()
            min_norm = norms.min().item()

            mlflow.log_metrics({
                "total_entailment_loss": avg_total_loss / len(dataloader),
                "entailment_loss": avg_entailment_loss / len(dataloader),
                "repulsion_loss": avg_repulsion_loss / len(dataloader),
                "entailment_lr": plr,
                "entailment_mean_norm": mean_norm,
                "entailment_max_norm": max_norm,
                "entailment_min_norm": min_norm,
            }, step=epoch)

            if scheduler is not None:
                scheduler.step(epoch=epoch + 1)

    def evaluate_edge_predictions(
        self,
        dataloader: DataLoader,
    ) -> None:
        evaluate_edge_predictions(model=self, dataloader=dataloader)

    def _clip_embeddings(self, epsilon: float = 1e-5) -> None:
        # min_norm = self.inner_radius + epsilon
        norm = self.weight.norm(dim=-1, keepdim=True).clamp_min(epsilon)
        # cond = norm < min_norm
        # projected = self.weight / norm * min_norm
        # new_weight = torch.where(cond, projected, self.weight)
        # self.weight = ManifoldParameter(
        #     data=new_weight, manifold=self.ball
        # )

        max_norm = 1 - epsilon
        cond = norm > max_norm
        projected = self.weight / norm * max_norm
        new_weight = torch.where(cond, projected, self.weight)
        self.weight = ManifoldParameter(
            data=new_weight, manifold=self.ball
        ).cuda()
