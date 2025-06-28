import argparse
import json
import os
import uuid
from pathlib import Path

import geoopt
import mlflow
import networkx as nx
import numpy as np
import requests
import torch
from dotenv import load_dotenv
from geoopt import PoincareBallExact
from geoopt.optim import RiemannianSGD
from torch.utils.data import DataLoader

from src.constants import CACHE_DIR, DEVICE, GOOGLE_BUCKET_URL
from src.shared.datasets import Dataset, DatasetVersion


def compute_recursive_prototypes_hyperbolic(hierarchy_list, base_prototypes, ball, norm_decay=1.0, eps=1e-8):
    prototypes_per_level = [ball.projx(base_prototypes)]

    for level, hierarchy in enumerate(reversed(hierarchy_list)):
        child_prototypes = prototypes_per_level[0]  # Most recent (lower-level)
        num_parents = hierarchy.shape[0]

        weights = hierarchy / (hierarchy.sum(dim=1, keepdim=True) + eps)  # [P, C]

        parent_means = []
        for p in range(num_parents):
            indices = hierarchy[p].nonzero(as_tuple=True)[0]
            x = child_prototypes[indices]
            w = weights[p, indices]
            mean = ball.weighted_midpoint(x, w)

            if norm_decay is not None:
                norm_scale = torch.exp(torch.tensor(-norm_decay * (len(hierarchy_list) - level), device=mean.device))
                mean = mean * norm_scale

            parent_means.append(mean)

        parent_means = torch.stack(parent_means, dim=0)
        prototypes_per_level.insert(0, parent_means)

    return torch.cat(prototypes_per_level, dim=0)


# def compute_genus_prototypes_hyperbolic(species_prototypes, hierarchy, ball, eps=1e-8):
#     species_prototypes = ball.projx(species_prototypes)
#
#     genus_weights = hierarchy / (hierarchy.sum(dim=1, keepdim=True) + eps)  # [G, S]
#
#     genus_means = []
#     for g in range(hierarchy.shape[0]):
#         indices = hierarchy[g].nonzero(as_tuple=True)[0]  # species indices for genus g
#         x = species_prototypes[indices] 
#         w = genus_weights[g, indices]
#         genus_mean = ball.weighted_midpoint(x, w)
#         genus_means.append(genus_mean)
#
#     genus_means = torch.stack(genus_means, dim=0)
#
#     # Stack genus and species prototypes
#     final_prototypes = torch.cat([genus_means, species_prototypes], dim=0)
#     return final_prototypes


def get_hierarchy(dataset_version, reload: bool = False):
    path = CACHE_DIR / f"{dataset_version}/hierarchy.npz"

    if reload or not path.is_file():
        url = f"{GOOGLE_BUCKET_URL}/{dataset_version}/hierarchy.npz?id={uuid.uuid4()}"
        response = requests.get(url)

        if response.status_code != 200:
            raise Exception(f"Could not retrieve hierarchy for {dataset_version}")

        with open(path, "wb") as f:
            f.write(response.content)

    data = np.load(path)

    if "data" in data:
        # Single hierarchy matrix
        return [data["data"].squeeze()]

    # Otherwise, assume multiple levels: level_0, level_1, ...
    hierarchy = [data[key] for key in sorted(data.files, key=lambda k: int(k.split("_")[1]))]
    return hierarchy


def get_metadata(dataset_version, reload: bool = False):
    directory = CACHE_DIR / dataset_version
    path = directory / "metadata.json"

    if reload or not path.is_file():
        url = f"{GOOGLE_BUCKET_URL}/{dataset_version}/metadata.json?id={uuid.uuid4()}"

        response = requests.get(url)
        if response.status_code != 200:
            raise Exception(f"Could not retrieve metadata for {dataset_version}, status: {response.status_code}")
        metadata = response.json()

        if not directory.exists():
            directory.mkdir(parents=True)

        with open(path, "w") as f:
            json.dump(metadata, f)

        return metadata

    with open(path, "r") as f:
        return json.load(f)



def run(args):
    hierarchies = get_hierarchy(args.dataset)
    metadata = get_metadata(args.dataset)

    hierarchies = [torch.tensor(hierarchy) for hierarchy in hierarchies]

    base = Path(f"./prototypes/{args.dataset}/species_hypersphere")
    f = base / f"{args.dims}.npy"
    unit_sphere_prototypes = torch.tensor(np.load(f))
    ball = geoopt.PoincareBallExact(c=args.curvature)
    new_prototypes = compute_recursive_prototypes_hyperbolic(hierarchies, unit_sphere_prototypes, ball=ball)

    base = Path(f"./prototypes/{args.dataset}/avg_hyperbolic")

    if not base.is_dir():
        base.mkdir(parents=True)

    path = f"{base}/{args.dims}.npy"

    np.save(path, new_prototypes.numpy())
    print(path)



def parse_args():
    parser = argparse.ArgumentParser(
        prog="Hyperbolic embeddings",
        description="Training script for embedding genus and species in hyperbolic space",
    )
    parser.add_argument("--batch-size", default=16, required=False, type=int)
    parser.add_argument(
        "-d",
        "--dataset",
        required=True,
        type=DatasetVersion,
        choices=[v.value for v in DatasetVersion],
    )
    parser.add_argument(
        "--dims",
        default=2,
        required=True,
        type=int,
    )
    parser.add_argument(
        "--curvature",
        default=3,
        required=True,
        type=float
    )
    parser.add_argument('--lr', dest="learning_rate", default=0.1, type=float)
    parser.add_argument("--reload", action="store_true", default=False, required=False)
    parser.add_argument('--momentum', dest="momentum", default=0.9, type=float)
    parser.add_argument('--epochs', dest="epochs", default=10000, type=int,)
    parser.add_argument('--resdir', default="./prototypes", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args)
