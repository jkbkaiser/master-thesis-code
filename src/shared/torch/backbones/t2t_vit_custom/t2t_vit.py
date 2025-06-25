# Copyright (c) [2012]-[2021] Shanghai Yitu Technology Co., Ltd.
#
# This source code is licensed under the Clear BSD License
# LICENSE file in the root directory of this file
# All rights reserved.
"""
T2T-ViT
"""

import geoopt
import numpy as np
import torch
import torch.nn as nn
from timm.layers.weight_init import trunc_normal_
from timm.models import load_pretrained, register_model

from .token_performer import Token_performer
from .token_transformer import Token_transformer
from .transformer_block import Block, get_sinusoid_encoding


def _cfg(url="", **kwargs):
    return {
        "url": url,
        "num_classes": 1000,
        "input_size": (3, 224, 224),
        "pool_size": None,
        "crop_pct": 0.9,
        "interpolation": "bicubic",
        "mean": (0.485, 0.456, 0.406),
        "std": (0.229, 0.224, 0.225),
        "classifier": "head",
        **kwargs,
    }


default_cfgs = {
    "T2t_vit_t_14": _cfg(),
}


class T2T_module(nn.Module):
    """
    Tokens-to-Token encoding module
    """

    def __init__(
        self,
        img_size=224,
        tokens_type="performer",
        in_chans=3,
        embed_dim=768,
        token_dim=64,
    ):
        super().__init__()

        if tokens_type == "transformer":
            self.soft_split0 = nn.Unfold(
                kernel_size=(7, 7), stride=(4, 4), padding=(2, 2)
            )
            self.soft_split1 = nn.Unfold(
                kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)
            )
            self.soft_split2 = nn.Unfold(
                kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)
            )

            self.attention1 = Token_transformer(
                dim=in_chans * 7 * 7, in_dim=token_dim, num_heads=1, mlp_ratio=1.0
            )
            self.attention2 = Token_transformer(
                dim=token_dim * 3 * 3, in_dim=token_dim, num_heads=1, mlp_ratio=1.0
            )
            self.project = nn.Linear(token_dim * 3 * 3, embed_dim)

        elif tokens_type == "performer":
            self.soft_split0 = nn.Unfold(
                kernel_size=(7, 7), stride=(4, 4), padding=(2, 2)
            )
            self.soft_split1 = nn.Unfold(
                kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)
            )
            self.soft_split2 = nn.Unfold(
                kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)
            )

            # self.attention1 = Token_performer(dim=token_dim, in_dim=in_chans*7*7, kernel_ratio=0.5)
            # self.attention2 = Token_performer(dim=token_dim, in_dim=token_dim*3*3, kernel_ratio=0.5)
            self.attention1 = Token_performer(
                dim=in_chans * 7 * 7, in_dim=token_dim, kernel_ratio=0.5
            )
            self.attention2 = Token_performer(
                dim=token_dim * 3 * 3, in_dim=token_dim, kernel_ratio=0.5
            )
            self.project = nn.Linear(token_dim * 3 * 3, embed_dim)

        elif (
            tokens_type == "convolution"
        ):  # just for comparison with conolution, not our model
            # for this tokens type, you need change forward as three convolution operation
            self.soft_split0 = nn.Conv2d(
                3, token_dim, kernel_size=(7, 7), stride=(4, 4), padding=(2, 2)
            )  # the 1st convolution
            self.soft_split1 = nn.Conv2d(
                token_dim, token_dim, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)
            )  # the 2nd convolution
            self.project = nn.Conv2d(
                token_dim, embed_dim, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)
            )  # the 3rd convolution

        self.num_patches = (img_size // (4 * 2 * 2)) * (
            img_size // (4 * 2 * 2)
        )  # there are 3 sfot split, stride are 4,2,2 seperately

    def forward(self, x):
        # step0: soft split
        x = self.soft_split0(x).transpose(1, 2)

        # iteration1: re-structurization/reconstruction
        x = self.attention1(x)
        B, new_HW, C = x.shape
        x = x.transpose(1, 2).reshape(B, C, int(np.sqrt(new_HW)), int(np.sqrt(new_HW)))
        # iteration1: soft split
        x = self.soft_split1(x).transpose(1, 2)

        # iteration2: re-structurization/reconstruction
        x = self.attention2(x)
        B, new_HW, C = x.shape
        x = x.transpose(1, 2).reshape(B, C, int(np.sqrt(new_HW)), int(np.sqrt(new_HW)))
        # iteration2: soft split
        x = self.soft_split2(x).transpose(1, 2)

        # final tokens
        x = self.project(x)

        return x


class ClassifierHead(nn.Module):
    def __init__(self, in_features, hidden_features, prototype_dim, ball, prototypes, temp):
        super().__init__()
        self.ball = ball
        self.prototypes = prototypes
        self.temp = temp

        self.mlp = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.GELU(),
            nn.Linear(hidden_features, hidden_features),
            nn.GELU(),
            nn.Linear(hidden_features, prototype_dim),
        )

    def forward(self, x):
        x = self.mlp(x)
        x_hyp = self.ball.expmap0(x)
        dists = self.ball.dist(self.prototypes[None, :, :], x_hyp[:, None, :])
        logits = - dists / self.temp

        pred = logits.argmax(dim=1)
        pred_prototypes = self.prototypes[pred]
        feedback = self.ball.logmap0(pred_prototypes.detach())

        return logits, feedback


class T2T_ViT_Custom(nn.Module):
    def __init__(
        self,
        img_size=224,
        tokens_type="performer",
        in_chans=3,
        num_classes=1000,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        token_dim=64,
        taxonomic_levels=1,
        prototypes=None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = (
            embed_dim  # num_features for consistency with other models
        )
        self.taxonomic_levels = taxonomic_levels
        self.prototypes = prototypes

        self.tokens_to_token = T2T_module(
            img_size=img_size,
            tokens_type=tokens_type,
            in_chans=in_chans,
            embed_dim=embed_dim,
            token_dim=token_dim,
        )
        num_patches = self.tokens_to_token.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            data=get_sinusoid_encoding(n_position=num_patches + 1, d_hid=embed_dim),
            requires_grad=False,
        )
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)

        # Classifier head
        self.ball = geoopt.PoincareBallExact(c=1.5)  # or use your curvature setting
        self.temp = 0.07  # or pass it as a parameter

        if self.prototypes is None:
            raise ValueError("Prototypes must be provided")

        self.heads = nn.ModuleList([
            ClassifierHead(
                in_features=embed_dim,
                hidden_features=4096 if l == taxonomic_levels - 1 else 1024,
                prototype_dim=self.prototypes[l].shape[1],
                ball=self.ball,
                prototypes=self.prototypes[l],
                temp=self.temp,
            )
            for l in range(taxonomic_levels)
        ])

        self.feedback_proj = nn.ModuleList([
            nn.Linear(self.prototypes[l].shape[1], embed_dim)
            for l in range(taxonomic_levels)
        ])

        trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

        self.default_cfg = {}

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore(drop=False)
    def no_weight_decay(self):
        return {"cls_token"}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes):
        self.num_classes = num_classes
        self.head = (
            nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        )

    def forward_features(self, x):
        B = x.shape[0]
        x = self.tokens_to_token(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks[:-self.taxonomic_levels]:
            x = blk(x)

        outputs = []
        feedback = None
        for i, blk in enumerate(self.blocks[-self.taxonomic_levels:]):
            if feedback is not None:
                projected_feedback = self.feedback_proj[i](feedback)
                x = torch.cat([x, projected_feedback.unsqueeze(1)], dim=1)

            x = blk(x)

            x_cls = self.norm(x)[:, 0]

            logits, feedback_proto = self.heads[i](x_cls)
            feedback = self.ball.logmap0(feedback_proto)

            outputs.append(logits)

        return outputs

    def forward(self, x):
        x = self.forward_features(x)
        return x


@register_model
def t2t_vit_t_14_custom(pretrained=False, taxonomic_levels=1, prototypes=None, **kwargs):
    if pretrained:
        kwargs.setdefault("qk_scale", 384**-0.5)
    model = T2T_ViT_Custom(
        tokens_type="transformer",
        embed_dim=384,
        depth=14,
        num_heads=6,
        mlp_ratio=3.0,
        taxonomic_levels=taxonomic_levels,
        prototypes=prototypes,
        **kwargs,
    )
    model.default_cfg = default_cfgs["T2t_vit_t_14"]
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get("in_chans", 3)
        )
    return model
