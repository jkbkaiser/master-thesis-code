import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import torch
from flax import nnx

from src.shared.jax.backbones.t2t_vit.token_transformer import TokenTransformer
from src.shared.jax.backbones.t2t_vit.transformer_block import TransformerBlock
from src.shared.jax.modules import Unfold


class TokensToToken(nnx.Module):
    def __init__(self, embed_features, batch_size: int, rngs: nnx.Rngs):
        self.batch_size = batch_size
        self.embed_features = embed_features

        self.img_size = 224
        self.in_chans = 3
        self.token_features = 64

        self.num_patches = (self.img_size // (4 * 2 * 2)) * (
            self.img_size // (4 * 2 * 2)
        )

        self.soft_split1 = Unfold(
            kernel_size=(7, 7), stride=(4, 4), padding=(2, 2)
        )
        self.soft_split2 = Unfold(
            kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)
        )
        self.soft_split3 = Unfold(
            kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)
        )
        self.transformer1 = TokenTransformer(
            self.in_chans * 7 * 7, self.token_features, rngs=rngs
        )
        self.transformer2 = TokenTransformer(
            self.token_features * 3 * 3, self.token_features, rngs=rngs
        )
        self.project = nnx.Linear(
            self.token_features * 3 * 3, self.embed_features, rngs=rngs
        )

    def __call__(self, x):
        x = self.soft_split1(x).transpose(0, 2, 1)

        x = self.transformer1(x)
        x = x.transpose(0, 2, 1).reshape(self.batch_size, 64, 56, 56)
        x = self.soft_split2(x).transpose(0, 2, 1)

        x = self.transformer2(x)
        x = x.transpose(0, 2, 1).reshape(self.batch_size, 64, 28, 28)
        x = self.soft_split3(x).transpose(0, 2, 1)

        x = self.project(x)

        return x


def get_sinusoidal_encoding(num_positions, embed_features):
    w = np.arange(0, embed_features // 2)

    def enc_single_position(pos):
        angles = pos / np.pow(10000, 2 * w / embed_features)
        sin = np.sin(angles)
        cos = np.cos(angles)

        # Interleave sin and cos: [sin1, cos1, sin2, cos2, ...]
        interleaved = np.stack((sin, cos), axis=1).reshape(-1)
        return interleaved

    positions = np.arange(0, num_positions)
    return np.array([enc_single_position(pos) for pos in positions])


class T2TViT(nnx.Module):
    def __init__(
        self,
        rngs: nnx.Rngs,
        mlp_ratio: float = 4.0,
        mlp_dropout: float = 0.0,
        path_dropout: float = 0.0,
        attn_dropout: float = 0.0,
        num_heads: int = 6,
        depth: int = 12,
        batch_size: int = 16,
    ):
        self.embed_features = 384

        self.tokens_to_token = TokensToToken(
            self.embed_features, batch_size=batch_size, rngs=rngs
        )

        self.cls_token = nnx.Param(jnp.zeros((1, 1, self.embed_features)))
        self.pos_embed = get_sinusoidal_encoding(self.tokens_to_token.num_patches + 1, self.embed_features)


        self.dropout = nnx.Dropout(mlp_dropout, rngs=rngs)

        dpr = jnp.linspace(0, path_dropout, depth)

        self.blocks = [
            TransformerBlock(
                self.embed_features, 
                self.embed_features,
                mlp_ratio=mlp_ratio,
                mlp_dropout=mlp_dropout,
                attn_dropout=attn_dropout,
                path_dropout=dpr[i].item(),
                num_heads=num_heads,
                rngs=rngs,
            )
            for i in range(depth)
        ]

        self.norm = nnx.LayerNorm(self.embed_features, rngs=rngs)

    def __call__(self, x):
        B = x.shape[0]
        embedding = self.tokens_to_token(x)
        cls_tokens = jnp.broadcast_to(self.cls_token.value, (B, 1, self.embed_features))
        embedding = jnp.concatenate([cls_tokens, embedding], axis=1)
        embedding = embedding + self.pos_embed
        embedding = self.dropout(embedding)

        for block in self.blocks:
            embedding = block(embedding)

        embedding = self.norm(embedding)

        cls_token = embedding[:, 0]
        return cls_token



def load_pretrained_weights(model: T2TViT, state_dict):
    for i, (key, tensor) in enumerate(list(state_dict.items())):
        np_tensor = tensor.cpu().numpy()

        if key == "cls_token":
            model.cls_token.value = np_tensor
        elif key == "pos_embed":
            model.pos_embed = np_tensor
        elif key.startswith("tokens_to_token."):
            parts = key.split(".")
            param_name = parts[-1]

            if parts[1] == "project":
                module = getattr(model.tokens_to_token, "project")

                if param_name == "weight":
                    module.kernel.value = np_tensor.T
                elif param_name == "bias":
                    module.bias.value = np_tensor

            elif parts[1].startswith("attention"):
                attention_block = parts[1]
                transformer_block = attention_block.replace("attention", "transformer")

                submodule = parts[2]

                block = getattr(model.tokens_to_token, transformer_block)
                module = getattr(block, submodule)

                # Assign weights, handling transpositions if necessary
                if parts[3] == "qkv" or parts[3] == "proj" or parts[3].startswith("fc"):
                    module = getattr(module, parts[3])


                if param_name == "weight":
                    if submodule.startswith("norm"):
                        module.scale.value = np_tensor
                    else:
                        module.kernel.value = np_tensor.T
                elif param_name == "bias":
                    module.bias.value = np_tensor
                else:
                    print("did not cover t2t attn module", key, i)
            else:
                print("did not cover t2t module", key, i)
        elif key.startswith("blocks."):
            parts = key.split(".")
            block_idx = int(parts[1])  # e.g., blocks.0 → 0
            submodule = parts[2]       # e.g., norm1, attn, mlp

            block = model.blocks[block_idx]

            # Handle normalization layers
            if submodule.startswith("norm"):
                ln = getattr(block, submodule)
                if parts[-1] == "weight":
                    ln.scale.value = np_tensor
                elif parts[-1] == "bias":
                    ln.bias.value = np_tensor
                else:
                    print("Unhandled norm param:", key)

            # Handle attention or mlp layers
            elif submodule in ["attn", "mlp"]:
                layer = getattr(block, submodule)

                if parts[3] in ["qkv", "proj", "fc1", "fc2"]:
                    sublayer = getattr(layer, parts[3])
                    if parts[-1] == "weight":
                        sublayer.kernel.value = np_tensor.T  # linear weights need transpose
                    elif parts[-1] == "bias":
                        sublayer.bias.value = np_tensor
                    else:
                        print("Unhandled attn/mlp param:", key)
                else:
                    print("Unknown sublayer:", key)
        elif key.startswith("head."):
            pass
            # parts = key.split(".")
            # param_name = parts[-1]
            #
            # module = model.head
            #
            # if param_name == "weight":
            #     module.kernel.value = np_tensor.T
            # elif param_name == "bias":
            #     module.bias.value = np_tensor
        elif key.startswith("norm."):
            parts = key.split(".")
            param_name = parts[-1]

            module = model.norm

            if param_name == "weight":
                module.scale.value = np_tensor
            elif param_name == "bias":
                module.bias.value = np_tensor

        else:
            print("did not cover", key, i)



def load_pretrained_t2t_vit():
    rngs = nnx.Rngs(params=0, dropout=jax.random.key(1))

    model = T2TViT(
        depth=14,
        mlp_ratio=3.0,
        rngs=rngs,
    )

    PRETRAINED_WEIGHTS_DIR = Path(os.getcwd()) / "pretrained_weights"
    PRETRAINED_T2T_VITT_T14 = PRETRAINED_WEIGHTS_DIR / "81.7_T2T_ViTt_14.pth.tar"

    checkpoint = torch.load(PRETRAINED_T2T_VITT_T14, weights_only=True, map_location="cpu")

    weights = checkpoint["state_dict_ema"]

    load_pretrained_weights(model, weights)

    return model


if __name__ == "__main__":
    @nnx.jit
    def test_step_jax(model: T2TViT, x: jax.Array):
      def loss_fn(model: T2TViT):
        logits = model(x)
        return logits.mean()

      loss, grads = nnx.value_and_grad(loss_fn)(model)
      return loss

    model = load_pretrained_t2t_vit()

    input = jnp.ones((16, 3, 224, 224), dtype=jnp.float32)

    test_step_jax(model, input)

    # input = torch.ones((16, 3, 224, 224)).to("cuda")
    # model = T2T_ViT(
    #     tokens_type="transformer",
    #     embed_dim=384,
    #     depth=8,
    #     num_heads=6,
    #     mlp_ratio=3.0
    # ).to("cuda")

