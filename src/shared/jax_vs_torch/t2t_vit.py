import time

import jax
import jax.numpy as jnp
import torch
from flax import nnx

from src.shared.jax.backbones.t2t_vit.t2t_vit import (T2TViT,
                                                      load_pretrained_t2t_vit)
from src.shared.torch.backbones.t2t_vit.t2t_vit import T2T_ViT


@nnx.jit
def test_step_jax(model: T2TViT, x: jax.Array):
  def loss_fn(model: T2TViT):
    logits = model(x)
    return logits.mean()

  loss, grads = nnx.value_and_grad(loss_fn)(model)
  return loss


def test_jax():
    model = load_pretrained_t2t_vit()

    input = jnp.ones((16, 3, 224, 224), dtype=jnp.float32)
    test_step_jax(model, input)

    start = time.time()

    for _ in range(100):
        test_step_jax(model, input)

    duration = time.time() - start
    print(f"Jax avg time per step: {duration / 100:.6f} seconds")

test_jax()

# def test_step_torch(model, x):
#     model.zero_grad()
#     out = model(x)
#     loss = out.mean()  # dummy loss
#     loss.backward()
#     return loss
#
# def test_torch():
#     input = torch.ones((16, 3, 224, 224)).to("cuda")
#     model = T2T_ViT(
#         tokens_type="transformer",
#         embed_dim=384,
#         depth=8,
#         num_heads=6,
#         mlp_ratio=3.0
#     ).to("cuda")
#
#     test_step_torch(model, input)
#
#     start = time.time()
#
#     for _ in range(100):
#         test_step_torch(model, input)
#
#     duration = time.time() - start
#     print(f"Torch avg time per step: {duration / 100:.6f} seconds")
#
# test_torch()
