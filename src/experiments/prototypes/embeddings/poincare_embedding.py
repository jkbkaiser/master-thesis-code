from typing import Optional

import mlflow
import torch
from geoopt.manifolds import PoincareBallExact
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader

from .base import BaseEmbedding


def poincare_embeddings_loss(
    dists: torch.Tensor, targets: torch.Tensor
) -> torch.Tensor:
    logits = dists.neg().exp()
    logits = logits

    numerator = torch.where(condition=targets, input=logits, other=0).sum(dim=-1)
    denominator = logits.sum(dim=-1)
    loss = (numerator / denominator).log().mean().neg()
    return loss


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

class PoincareEmbedding(BaseEmbedding):
    def __init__(
        self, num_embeddings: int, embedding_dim: int, ball: PoincareBallExact
    ) -> None:
        super(PoincareEmbedding, self).__init__(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            ball=ball,
        )

    def forward(self, edges: torch.Tensor) -> torch.Tensor:
        """
        Input:
            - edges: torch.Tensor of shape [batch_size, sample_size, 2],
              where sample_size is generally (1 + #negatives). 
        """
        embeddings = super(PoincareEmbedding, self).forward(edges)
        edge_distances = self.ball.dist(x=embeddings[:, :, 0, :], y=embeddings[:, :, 1, :])
        return edge_distances

    def score(self, edges: torch.Tensor, alpha: float = 1) -> torch.Tensor:
        embeddings = super(PoincareEmbedding, self).forward(edges)
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
        burn_in_epochs: int = 50,
        burn_in_lr_mult: float = 0.1,
    ):
        # Store initial learning rate
        lr = optimizer.param_groups[0]["lr"]

        for epoch in range(epochs):
            # Scale learning rate during burn-in
            if epoch < burn_in_epochs:
                lr_scale = burn_in_lr_mult + (1.0 - burn_in_lr_mult) * (epoch / burn_in_epochs)
                optimizer.param_groups[0]["lr"] = lr * lr_scale
            elif epoch == burn_in_epochs:
                optimizer.param_groups[0]["lr"] = lr

            avg_loss = 0
            avg_repulsion_loss = 0
            avg_poincare_loss = 0
            # avg_norms_loss = 0

            for batch in dataloader:
                edges = batch["edges"].to(self.weight.device)
                edge_label_targets = batch["edge_label_targets"].to(self.weight.device)

                optimizer.zero_grad()

                dists = self(edges=edges)

                poincare_loss = poincare_embeddings_loss(dists=dists, targets=edge_label_targets)

                repulsion_l = tangent_space_repulsion_loss(self.weight, self.ball) * get_repulsion_weight(epoch)

                loss = poincare_loss + repulsion_l

                loss.backward()
                optimizer.step()
                avg_loss += loss.item()

                avg_repulsion_loss += repulsion_l.item()
                avg_poincare_loss += poincare_loss.item()
                # avg_norms_loss += norms_p.item()

                with torch.no_grad():
                    self.weight.data = self.ball.projx(self.weight.data)

            if epoch > burn_in_epochs:
                if scheduler is not None:
                    scheduler.step()

            plr = optimizer.param_groups[0]["lr"]

            print(f"Epoch {epoch + 1}:  {avg_loss/len(dataloader)} lr: {plr}")

            norms = self.weight.norm(dim=1)

            mean_norm = norms.mean().item()
            max_norm = norms.max().item()
            min_norm = norms.min().item()

            mlflow.log_metrics({
                "total_poincare_loss": avg_loss / len(dataloader),
                "repulsion_loss": avg_repulsion_loss / len(dataloader),
                "poincare_loss": avg_poincare_loss / len(dataloader),
                # "norms_loss": avg_norms_loss / len(dataloader),
                "poincare_lr": plr,
                "poincare_mean_norm": mean_norm,
                "poincare_max_norm": max_norm,
                "poincare_min_norm": min_norm,
            }, step=epoch)
