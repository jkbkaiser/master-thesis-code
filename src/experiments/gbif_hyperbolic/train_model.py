import argparse
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm.auto import tqdm

from src.constants import DEVICE
from src.experiments.gbif_baselines.lighting import (PRETRAINED_T2T_VITT_T14,
                                                     PRETRAINED_VITAEv2)
from src.shared.datasets import Dataset, DatasetVersion
from src.shared.torch.backbones.t2t_vit.t2t_vit import t2t_vit_t_14
from src.shared.torch.backbones.t2t_vit.utils import load_for_transfer_learning
from src.shared.torch.backbones.vitaev2.ViTAEv2 import ViTAEv2_B

from .models.uniform import Uniform

torch.set_float32_matmul_precision("medium")

BACKBONE_DICT = {
    "t2t_vit": (t2t_vit_t_14, PRETRAINED_T2T_VITT_T14, 384),
    "vitaev2": (ViTAEv2_B, PRETRAINED_VITAEv2, 1024),
}

def create_model(model_hparams, ds):
    init, path_to_weights, out_features = BACKBONE_DICT[model_hparams["backbone_name"]]
    backbone = init(
        drop_rate=0.1,
        attn_drop_rate=0.1,
        drop_path_rate=0.1,
    )

    load_for_transfer_learning(
        backbone,
        path_to_weights,
        use_ema=True,
        strict=False,
    )

    model = Uniform(backbone, out_features, **model_hparams, ds=ds)

    return model


def run(args):
    curvature = 1
    scale_factor = 0.95

    ds = Dataset(args.dataset)
    ds.load(batch_size=args.batch_size, use_torch=True)

    genus_species_matrix = ds.hierarchy[0]
    _, num_classes = genus_species_matrix.shape

    prototypes = torch.from_numpy(np.load(args.prototypes)).float()
    prototypes = F.normalize(prototypes, p=2, dim=1)
    prototypes = prototypes.cuda() * scale_factor / math.sqrt(curvature)

    model_hparams = {
        "backbone_name": args.backbone,
        "architecture": num_classes,
    }

    model = create_model(model_hparams, args.dataset)

    model.to(DEVICE)

    optimizer = optim.SGD(
        model.parameters(),
        lr=args.learning_rate,
        momentum=args.momentum,
        nesterov=True,
        weight_decay=args.weight_decay,
    )

    for epoch in tqdm(range(args.epochs)):
        avg_loss = 0

        for i, batch in enumerate(ds.train_dataloader):
            imgs, genus_labels, species_labels = batch

            imgs = imgs.to(DEVICE)
            species_labels = species_labels.to(DEVICE)

            logits = model.forward(imgs)
            loss = model.loss_fn(logits, genus_labels, species_labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            avg_loss += loss

        print(avg_loss / len(ds.train_dataloader))

    print("F")


def parse_args():
    parser = argparse.ArgumentParser(
        prog="Hyperbolic embeddings",
        description="Training script for training a hyperbolic model using uniform embeddings",
    )
    parser.add_argument(
        "-d",
        "--dataset",
        required=True,
        type=DatasetVersion,
        choices=[v.value for v in DatasetVersion],
    )
    parser.add_argument("--batch-size", default=16, required=False, type=int)
    parser.add_argument('--learning-rate', default=0.1, type=float)
    parser.add_argument('--epochs', dest="epochs", default=25, type=int,)

    parser.add_argument(
        "--weight-decay", default=1e-2, required=False, type=float
    )

    parser.add_argument(
        "--momentum", default=0.9, required=False, type=float
    )

    parser.add_argument(
        "--backbone",
        default="t2t_vit",
        required=False,
        type=str,
        choices=["t2t_vit", "vitaev2"],
    )

    parser.add_argument('--prototypes', default="./prototypes/prototypes-64-gbif_genus_species_10k.npy", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args)
