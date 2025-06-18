from typing import Optional

import mlflow
import numpy as np
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
    numerator = torch.where(condition=targets, input=logits, other=0).sum(dim=-1)
    denominator = logits.sum(dim=-1)
    loss = (numerator / denominator).log().mean().neg()
    return loss

# def norms_penalty(prototypes, ball):
#     n = prototypes.size(0)
#     idx = torch.randint(0, n, (64,), device=prototypes.device)
#     sampled = prototypes[idx]
#     dists = ball.dist0(sampled)
#     return dists.mean()

def sampled_poincare_repulsion_loss(points: torch.Tensor, ball: PoincareBallExact, k: int = 64, temperature: float = 0.1) -> torch.Tensor:
    """
    Efficient repulsion loss by sampling k points per prototype.
    """
    n = points.size(0)
    device = points.device

    # Randomly sample k other indices for each point
    idx = torch.randint(0, n, (n, k), device=device)
    rows = torch.arange(n, device=device).unsqueeze(1).expand(-1, k)

    a = points[rows.reshape(-1)]  # shape: [n*k, dim]
    b = points[idx.reshape(-1)]   # shape: [n*k, dim]

    dists = ball.dist(a, b)  # [n*k]
    repulsion = torch.exp(-dists / temperature)
    return repulsion.mean()


# def prototype_loss(prototypes):
#     normed = torch.nn.functional.normalize(prototypes, dim=1)
#     # Dot product of normalized prototypes is cosine similarity.
#     product = torch.matmul(normed, normed.t()) + 1
#     # Remove diagnonal from loss.
#     product -= 2. * torch.diag(torch.diag(product))
#     # Minimize maximum cosine similarity.
#     loss = product.max(dim=1)[0]
#     return loss.mean()

# def repulsion_loss(embeddings: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
#     """
#     Penalizes high cosine similarity between different embeddings.
#     Intended to encourage spread-out representations.
#     """
#     # Normalize embeddings to unit vectors
#     normed = torch.nn.functional.normalize(embeddings, dim=1)
#     sim_matrix = torch.matmul(normed, normed.T)  # shape [N, N]
#
#     # Mask self-similarity
#     sim_matrix.fill_diagonal_(float("-inf"))
#
#     # Softmax-based repulsion loss: encourage uniform spread
#     repulsion = torch.logsumexp(sim_matrix / temperature, dim=1)
#     return repulsion.mean()

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

                # repulsion_l = sampled_poincare_repulsion_loss(self.weight, self.ball)

                # norms_p = norms_penalty(self.weight, self.ball)

                loss = poincare_loss

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
                "total_loss": avg_loss / len(dataloader),
                "repulsion_loss": avg_repulsion_loss / len(dataloader),
                "poincare_loss": avg_poincare_loss / len(dataloader),
                # "norms_loss": avg_norms_loss / len(dataloader),
                "poincare_lr": plr,
                "poincare_mean_norm": mean_norm,
                "poincare_max_norm": max_norm,
                "poincare_min_norm": min_norm,
            }, step=epoch)
