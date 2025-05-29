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

    def train(
        self,
        dataloader: DataLoader,
        epochs: int,
        optimizer: Optimizer,
        scheduler: LRScheduler = None,
        burn_in_epochs: int = 10,
        burn_in_lr_mult: float = 0.1,
        store_losses: bool = False,
        store_intermediate_weights: bool = False,
        **kwargs
    ):
        # Store initial learning rate
        lr = optimizer.param_groups[0]["lr"]

        if store_losses:
            losses = []
        if store_intermediate_weights:
            weights = [self.weight.clone().detach()]

        for epoch in range(epochs):
            # Scale learning rate during burn-in
            if epoch < burn_in_epochs:
                lr_scale = burn_in_lr_mult + (1.0 - burn_in_lr_mult) * (epoch / burn_in_epochs)
                optimizer.param_groups[0]["lr"] = lr * lr_scale
            elif epoch == burn_in_epochs:
                optimizer.param_groups[0]["lr"] = lr

            avg_loss = 0

            for idx, batch in enumerate(dataloader):
                edges = batch["edges"].to(self.weight.device)
                edge_label_targets = batch["edge_label_targets"].to(self.weight.device)

                optimizer.zero_grad()

                dists = self(edges=edges)

                loss = poincare_embeddings_loss(dists=dists, targets=edge_label_targets)
                loss.backward()
                optimizer.step()

                if store_losses:
                    losses.append(loss.item())

                avg_loss += loss.item()

            if epoch > burn_in_epochs:
                if scheduler is not None:
                    scheduler.step()

            plr = optimizer.param_groups[0]["lr"]
            print(f"Epoch {epoch + 1}:  {loss} lr: {plr}")

            if store_intermediate_weights:
                weights.append(self.weight.clone().detach())


        return (
            losses if store_losses else None,
            weights if store_intermediate_weights else None,
        )
