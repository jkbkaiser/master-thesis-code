from pathlib import Path
from typing import Optional

import numpy as np

# Define save directory once outside the loop
base = Path(f"./prototypes/clibdb/distortion")
base.mkdir(parents=True, exist_ok=True)


import mlflow
import torch
from geoopt.manifolds import PoincareBallExact
from geoopt.optim import RiemannianSGD
from geoopt.tensor import ManifoldParameter
from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, LRScheduler
from torch.utils.data import DataLoader

from .base import BaseEmbedding
from .poincare_embedding import PoincareEmbedding
from .utils.clone_optimizer import clone_one_group_optimizer
from .utils.eval_tools import evaluate_edge_predictions


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

    loss = (1.0 / topk).mean()
    return loss



def distortion_loss(
    embeddings: torch.Tensor, dist_targets: torch.Tensor, ball: PoincareBallExact, epoch:int, max_epoch:int,
):
    embedding_dists = ball.dist(x=embeddings[:, :, 0, :], y=embeddings[:, :, 1, :])

    # print(embedding_dists)


    dist_loss = (embedding_dists - dist_targets).abs() / dist_targets

    # embedding_dists = ball.dist(x=embeddings[:, :, 0, :], y=embeddings[:, :, 1, :])
    # dist_loss = (embedding_dists - dist_targets).abs() / dist_targets 

    norm_loss = compute_norm_loss(embeddings, dist_targets, ball, epoch, max_epoch)

    return dist_loss.mean(), norm_loss.mean() 


def compute_norm_loss(
    embeddings: torch.Tensor,
    dist_targets: torch.Tensor,
    ball: PoincareBallExact,
    epoch:int,
    max_epoch:int
) -> torch.Tensor:

    embedding_norm = ball.dist0(embeddings,keepdim=True) 
    unique_even_dists = torch.unique(dist_targets[dist_targets % 2 == 0])
    all_even_embedding_norms = [embedding_norm[dist_targets == i] for i in unique_even_dists]
    mean_even_embedding_norms = [norm.mean() for norm in all_even_embedding_norms]
    even_embedding_loss = torch.cat([(even_embedding_norms - mean_even_embedding_norms) for even_embedding_norms, mean_even_embedding_norms in zip(all_even_embedding_norms, mean_even_embedding_norms)])

    return (epoch/max_epoch) * even_embedding_loss.abs()


class DistortionEmbedding(BaseEmbedding):
    def __init__(
        self, num_embeddings: int, embedding_dim: int, ball: PoincareBallExact
    ) -> None:
        super(DistortionEmbedding, self).__init__(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            ball=ball,
        )

    def forward(self, labels: torch.Tensor) -> torch.Tensor:
        embeddings = super(DistortionEmbedding, self).forward(labels)
        return embeddings

    def score(self, edges: torch.Tensor, alpha: float = 1) -> torch.Tensor:
        """
        Score function used for predicting directed edges during evaluation.
        Trivial for entailment cones, but not for Poincare embeddings. Note that **_kwargs
        catches unused keywords such as the alpha from Poincare embeddings.
        """
        embeddings = super(DistortionEmbedding, self).forward(edges)
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
        pretrain_epochs: int = 0,
        pretrain_lr: float = 0.1,
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
            avg_dist_loss = 0
            avg_norm_loss = 0

            for batch in dataloader:
                edges = batch["edges"].to(self.weight.device)
                dist_targets = batch["dist_targets"].to(self.weight.device)

                # print("---")
                # print(edges)
                # print(dist_targets)

                optimizer.zero_grad()
                embeddings = self(edges)

                dist_loss, norm_loss = distortion_loss(
                    embeddings=embeddings,
                    dist_targets=dist_targets,
                    ball=self.ball,
                    epoch=epoch,
                    max_epoch = epochs,
                    # mask = mask
                )

                loss = dist_loss + norm_loss

                loss.backward()
                optimizer.step()

                avg_total_loss += loss.item()
                avg_dist_loss += dist_loss.item()
                avg_norm_loss += norm_loss.item()

                with torch.no_grad():
                    self.weight.data = self.ball.projx(self.weight.data)

            if (epoch + 1) % 100 == 0:
                proto_path = base / f"128_epoch{epoch + 1}.npy"
                np.save(proto_path, self.weight.data.cpu().numpy())
                print(f"Saved prototypes to {proto_path}")

            plr = optimizer.param_groups[0]["lr"]

            print(f"Epoch {epoch + 1}:  {avg_total_loss/len(dataloader)}, lr: {plr}")

            norms = self.weight.norm(dim=1)

            mean_norm = norms.mean().item()
            max_norm = norms.max().item()
            min_norm = norms.min().item()

            mlflow.log_metrics({
                "total_distortion_loss": avg_total_loss / len(dataloader),
                "dist_loss": avg_dist_loss / len(dataloader),
                "norm_loss": avg_norm_loss / len(dataloader),
                "distortion_lr": plr,
                "distortion_mean_norm": mean_norm,
                "distortion_max_norm": max_norm,
                "distortion_min_norm": min_norm,
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
