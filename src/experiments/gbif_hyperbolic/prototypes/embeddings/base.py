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

        self.weight = ManifoldParameter(
            data=ManifoldTensor(num_embeddings, embedding_dim, manifold=ball)
        ).to(DEVICE)

        self.reset_embeddings()

    def reset_embeddings(self) -> None:
        nn.init.uniform_(
            tensor=self.weight,
            a=-0.001,
            b=0.001,
        )

    def forward(self, labels: torch.Tensor) -> torch.Tensor:
        return self.weight[labels]

    def score(self, edges: torch.Tensor, **kwargs) -> torch.Tensor:
        raise NotImplementedError

