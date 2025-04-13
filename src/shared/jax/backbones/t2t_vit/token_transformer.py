import jax
from flax import nnx

from src.shared.jax.backbones.t2t_vit.mlp import MLP


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
        qkv_bias: bool = False,
    ):
        self.num_heads = num_heads
        self.in_features = in_features
        self.qkv_features = qkv_features
        self.out_features = out_features

        head_dim = in_features // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nnx.Linear(
            in_features,
            qkv_features * 3,
            use_bias=qkv_bias,
            rngs=rngs
        )
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


class TokenTransformer(nnx.Module):
    def __init__(
        self,
        in_features: int,
        token_features: int,
        rngs: nnx.Rngs,
        mlp_ratio: float = 1.0,
        path_dropout: float = 0.0,
        attn_dropout: float = 0.0,
        mlp_dropout: float = 0.0,
    ):
        self.norm1 = nnx.LayerNorm(in_features, rngs=rngs)
        self.attn = Attention(
            in_features,
            token_features,
            token_features,
            attn_dropout=attn_dropout,
            num_heads=1,
            rngs=rngs,
        )
        self.dropout_path = nnx.Dropout(rate=path_dropout) if path_dropout > 0.0 else lambda x: x
        self.norm2 = nnx.LayerNorm(token_features, rngs=rngs)
        self.mlp = MLP(
            in_features=token_features,
            hidden_features=int(token_features * mlp_ratio),
            out_features=token_features,
            act_layer=nnx.gelu,
            dropout=mlp_dropout,
            rngs=rngs,
        )

    def __call__(self, x):
        x = self.norm1(x)
        x = self.attn(x)
        x = self.norm2(x)
        x = x + self.dropout_path(self.mlp(x))
        return x
