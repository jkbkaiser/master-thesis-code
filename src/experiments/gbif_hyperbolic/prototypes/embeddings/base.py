import numpy as np
import torch
import torch.nn as nn
from geoopt.manifolds import PoincareBallExact
from geoopt.tensor import ManifoldParameter, ManifoldTensor

from src.constants import DEVICE


class BaseEmbedding(nn.Module):
    def __init__(
        self, num_embeddings: int, embedding_dim: int, ball: PoincareBallExact
    ) -> None:
        super(BaseEmbedding, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.ball = ball

        prototypes = np.load("./prototypes/gbif_genus_species_100k/genus_species_poincare/128.npy")
        prototypes_t = torch.tensor(prototypes, dtype=torch.float32, device=DEVICE)

        # Project onto manifold
        prototypes_proj = self.ball.projx(prototypes_t)

        manifold_tensor = ManifoldTensor(prototypes_proj, manifold=self.ball).to(DEVICE)

        # Wrap as trainable manifold parameter
        self.weight = ManifoldParameter(manifold_tensor)

        # self.weight = ManifoldParameter(
        #     data=ManifoldTensor(num_embeddings, embedding_dim, manifold=ball).to(DEVICE)
        # )
        # self.reset_embeddings()

    def reset_embeddings(self) -> None:
        nn.init.uniform_(
            tensor=self.weight,
            a=-0.001,
            b=0.001,
        )

    # def reset_embeddings(self) -> None:
    #     with torch.no_grad():
    #         direction = torch.randn(self.num_embeddings, self.embedding_dim, device=self.weight.device)
    #         direction = direction / direction.norm(dim=-1, keepdim=True)
    #
    #         # Sample radius close to 1, but not too close (e.g. avoid numerical instability near 1.0)
    #         r = torch.empty(self.num_embeddings, 1, device=self.weight.device).uniform_(0.7, 0.99)
    #
    #         self.weight.data = self.ball.projx(direction * r)

    def forward(self, labels: torch.Tensor) -> torch.Tensor:
        return self.weight[labels]
