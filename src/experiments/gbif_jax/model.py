from flax import nnx

from src.shared.jax.backbones.t2t_vit.t2t_vit import load_pretrained_t2t_vit

from .models import Baseline

MODEL_DICT = {
    "baseline": Baseline
}

def load_model(model_name: str, rngs: nnx.Rngs):
    backbone = load_pretrained_t2t_vit()

    model_cls = MODEL_DICT[model_name]

    model = model_cls(
        backbone,
        embed_features=384,
        out_features=1985,
        rngs=rngs,
    )

    return model
