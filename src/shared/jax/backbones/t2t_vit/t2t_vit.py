from typing import cast

import jax
import jax.numpy as jnp
import torch
from flax import nnx

from src.shared.jax.modules import Unfold
from src.shared.torch.backbones.t2t_vit.t2t_vit import t2t_vit_t_14


class Attention(nnx.Module):
    def __init__(
        self,
        in_features: int,
        qkv_features: int,
        out_features: int,
        rngs: nnx.Rngs,
        num_heads: int = 8,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
    ):
        self.num_heads = num_heads
        self.qkv_features = qkv_features
        self.out_features = out_features

        head_dim = in_features // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nnx.Linear(in_features, qkv_features * 3, rngs=rngs)
        self.attn_dropout = nnx.Dropout(attn_dropout)
        self.proj = nnx.Linear(qkv_features, out_features, rngs=rngs)
        self.proj_dropout = nnx.Dropout(proj_dropout)

    def __call__(self, x):
        B, N, _ = x.shape

        qkv = self.qkv(x)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.qkv_features)
        qkv = qkv.transpose((2, 0, 3, 1, 4))

        q, k, v = qkv[0], qkv[2], qkv[2]

        attn = jax.nn.softmax(
            (q * self.scale) @ k.transpose(0, 1, 3, 2)
        )

        x = (attn @ v).transpose(0, 2, 1, 3).reshape(B, N, self.qkv_features)
        x = self.proj(x)
        x = self.proj_dropout(x)

        x = v.squeeze(1) + x

        return x

class MLP(nnx.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        act_layer,
        drop: float,
        rngs: nnx.Rngs,
    ):
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.rngs = rngs
        self.act_layer = act_layer
        self.drop = drop

    def setup(self):
        self.fc1 = nnx.Linear(self.in_features, self.hidden_features, rngs=rngs)
        self.act = self.act_layer
        self.fc2 = nnx.Linear(self.hidden_features, self.out_features, rngs=rngs)
        self.dropout = nnx.Dropout(self.drop)

    def __call__(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TokenTransformer(nnx.Module):
    def __init__(
        self,
        in_features: int,
        token_features: int,
        rngs: nnx.Rngs,
        mlp_ratio: float = 1.0,
        drop_path: float = 0.0,
        drop: float = 0.0,
    ):
        self.in_features = in_features
        self.token_features = token_features
        self.rngs = rngs
        self.drop_path = drop_path
        self.drop = drop
        self.mlp_ratio = mlp_ratio

    def setup(self):
        self.norm1 = nnx.LayerNorm(self.in_features, rngs=rngs)
        self.attn = Attention(
            self.in_features,
            self.token_features,
            self.token_features,
            num_heads=1,
            rngs=self.rngs,
        )
        self.dropout_path = nnx.Dropout(rate=self.drop_path) if self.drop_path > 0.0 else lambda x: x
        self.norm2 = nnx.LayerNorm(self.token_features, rngs=rngs)
        self.mlp = MLP(
            in_features=self.token_features,
            hidden_features=int(self.token_features * self.mlp_ratio),
            out_features=self.token_features,
            act_layer=nnx.gelu,
            drop=self.drop,
            rngs=rngs,
        )

    def __call__(self, x):
        x = self.norm1(x)
        x = self.attn(x)
        x = self.norm2(x)
        x = x + self.dropout_path(self.mlp(x))
        return x


class TokensToToken(nnx.Module):
    def __init__(self, embed_features, batch_size: int, rngs: nnx.Rngs):
        self.batch_size = batch_size
        self.embed_features = embed_features

        self.img_size = 244
        self.in_chans = 3
        self.token_features = 64

    def setup(self):
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


class T2TViT(nnx.Module):
    def __init__(self):
        self.embed_features = 384

    def setup(self):
        self.tokens_to_token = TokensToToken(
            self.embed_features, batch_size=16, rngs=rngs
        )
        self.cls_token = nnx.Param(jnp.zeros((1, 1, self.embed_features)))

    def __call__(self, x):
        B = x.shape[0]
        tokens = self.tokens_to_token(x)
        cls_token_array = cast(jnp.ndarray, self.cls_token)
        cls_tokens = jnp.broadcast_to(cls_token_array, (B, 1, self.embed_features))
        tokens = jnp.concatenate([cls_tokens, tokens], axis=1)
        tokens = tokens + self.pos_embed
        return t


if __name__ == "__main__":
    t = torch.ones((16, 3, 224, 224))
    model = t2t_vit_t_14()
    out = model(t)

    print("---")

    rngs = nnx.Rngs(params=0, dropout=jax.random.key(1))
    input = jnp.ones((16, 3, 224, 224))
    model = T2TViT()
    output = model(input)
