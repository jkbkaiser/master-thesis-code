import torch
from geoopt.manifolds import PoincareBallExact
from geoopt.tensor import ManifoldParameter
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader

from .base import BaseEmbedding
from .poincare_embedding import PoincareEmbedding
from .utils.clone_optimizer import clone_one_group_optimizer
from .utils.eval_tools import evaluate_edge_predictions


def distortion_loss(
    embeddings: torch.Tensor, dist_targets: torch.Tensor, ball: PoincareBallExact, epoch:int, max_epoch:int
) -> torch.Tensor:
    embedding_dists = ball.dist(x=embeddings[:, :, 0, :], y=embeddings[:, :, 1, :])
    dist_loss = (embedding_dists - dist_targets).abs() / dist_targets 

    norm_loss = compute_norm_loss(embeddings, dist_targets, ball, epoch, max_epoch)

    if dist_loss.isnan().any():
        print("break")

    return dist_loss.mean() + 0.01 * norm_loss.mean()


def compute_norm_loss(
        embeddings: torch.Tensor, dist_targets: torch.Tensor, ball: PoincareBallExact, epoch:int, max_epoch:int
        ) -> torch.Tensor:
    # tangent_vecs = ball.logmap0(embeddings)
    embedding_norm = ball.dist0(embeddings,keepdim=True) 
    # embedding_norm = ball.norm(embeddings,tangent_vecs,keepdim=True) 
    unique_even_dists = torch.unique(dist_targets[dist_targets % 2 == 0])
    all_even_embedding_norms = [embedding_norm[dist_targets == i] for i in unique_even_dists]
    mean_even_embedding_norms = [norm.mean() for norm in all_even_embedding_norms]
    # print(mean_even_embedding_norms)
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

    def forward(self, edges: torch.Tensor) -> torch.Tensor:
        embeddings = super(DistortionEmbedding, self).forward(edges)
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

    def train(
        self,
        dataloader: DataLoader,
        epochs: int,
        optimizer: Optimizer,
        scheduler: LRScheduler = None,
        pretrain_epochs: int = 100,
        pretrain_lr: float = 5.0,
        burn_in_epochs: int = 10,
        burn_in_lr_mult: float = 0.1,
        store_losses: bool = False,
        store_intermediate_weights: bool = False,
        **kwargs
    ) -> None:
        # Initialize a Poincare embeddings model for pretraining
        poincare_embeddings = PoincareEmbedding(
            num_embeddings=self.num_embeddings,
            embedding_dim=self.embedding_dim,
            ball=self.ball,
        )

        # Copy the optimizer, but change the parameters to the Poincare embeddings model weights
        pretraining_optimizer = clone_one_group_optimizer(
            optimizer=optimizer,
            new_params=poincare_embeddings.parameters(),
            lr=pretrain_lr,
            momentum=0.9,
            weight_decay=0.005,
        )

        # TODO: properly copy scheduler instead of ignoring scheduler for pretraining

        # Perform pretraining
        losses, weights = poincare_embeddings.train(
            dataloader=dataloader,
            epochs=pretrain_epochs,
            optimizer=pretraining_optimizer,
            scheduler=None,
            burn_in_epochs=burn_in_epochs,
            burn_in_lr_mult=burn_in_lr_mult,
            store_losses=store_losses,
            store_intermediate_weights=store_intermediate_weights,
            **kwargs
        )

        # Copy pretrained embeddings, rescale and clip these and reset optimizer param group
        with torch.no_grad():
            self.weight.copy_(poincare_embeddings.weight)
            self.weight.mul_(0.8)
            self._clip_embeddings()
            optimizer = clone_one_group_optimizer(
                optimizer=optimizer,
                new_params=self.parameters(),
            )
            print(optimizer)

        for epoch in range(epochs):
            for idx, batch in enumerate(dataloader):
                edges = batch["edges"].to(self.weight.device)
                dist_targets = batch["dist_targets"].to(self.weight.device)

                optimizer.zero_grad()

                embeddings = self(edges=edges)

                loss = distortion_loss(
                    embeddings=embeddings,
                    dist_targets=dist_targets,
                    ball=self.ball,
                    epoch=epoch,
                    max_epoch = epochs
                )

                loss.backward()
                optimizer.step()

                if not (epoch + 1) % 20:
                    print(f"Epoch {epoch + 1}, batch {idx + 1}/{len(dataloader)}:  {loss}")
                    if store_intermediate_weights:
                        weights.append(self.weight.clone().detach())

                if store_losses:
                    losses.append(loss.item())

            if store_intermediate_weights:
                weights.append(self.weight.clone().detach())

            if scheduler is not None:
                scheduler.step(epoch=epoch + 1)

        return (
            losses if store_losses else None,
            weights if store_intermediate_weights else None,
        )

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
