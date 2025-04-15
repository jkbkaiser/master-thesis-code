import optax
from flax import nnx

from src.shared.jax.backbones.t2t_vit.t2t_vit import T2TViT


class Baseline(nnx.Module):
    def __init__(
        self,
        backbone: T2TViT,
        embed_features: int,
        out_features: int,
        rngs: nnx.Rngs,
    ):
        self.backbone = backbone
        self.head = nnx.Linear(embed_features, out_features, rngs=rngs)

        self.criterion = optax.softmax_cross_entropy_with_integer_labels

    def __call__(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return x

    def predict(self, logits):
        return logits.argmax(axis=1)

    def loss(self, logits, genus_labels, species_lables):
        return self.criterion(logits, species_lables).mean()
