import argparse
import uuid
from pathlib import Path

import numpy as np
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from src.constants import CACHE_DIR, DEVICE, GOOGLE_BUCKET_URL
from src.shared.datasets import DatasetVersion


def prototype_loss(prototypes):
    # Dot product of normalized prototypes is cosine similarity.
    product = torch.matmul(prototypes, prototypes.t()) + 1
    # Remove diagnonal from loss.
    product -= 2. * torch.diag(torch.diag(product))
    # Minimize maximum cosine similarity.
    loss = product.max(dim=1)[0]
    return loss.mean(), product.max()


def get_hierarchy(dataset_version, reload: bool = False):
    path = CACHE_DIR / f"{dataset_version}/hierarchy.npz"

    if reload or not path.is_file():
        url = f"{GOOGLE_BUCKET_URL}/{dataset_version}/hierarchy.npz?id={uuid.uuid4()}"
        response = requests.get(url)

        if response.status_code != 200:
            raise Exception(f"Could not retrieve metadata for {dataset_version}")

        with open(path, "wb") as f:
            f.write(response.content)

    hierarchy = np.load(path)["data"]
    return hierarchy.squeeze()

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
    hierarchy = get_hierarchy(args.dataset)
    metadata = get_metadata(args.dataset)

    num_genus, num_species = hierarchy.shape
    print(num_genus, num_species)

    prototypes = torch.randn(num_species, args.dims, device=DEVICE)
    prototypes = nn.Parameter(F.normalize(prototypes, p=2, dim=1))

    optimizer = optim.SGD([prototypes], lr=args.learning_rate, momentum=args.momentum)

    for i in range(args.epochs):
        optimizer.zero_grad()

        loss, sep = prototype_loss(prototypes)

        loss.backward()
        optimizer.step()

        with torch.no_grad():
            prototypes.data = F.normalize(prototypes.data, p=2, dim=1)

        print(f"{i} {args.epochs} {sep}")

    np.save(f"{args.resdir}/prototypes-{args.dims}-{args.dataset}.npy", prototypes.data.cpu().numpy())

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
    parser.add_argument('--lr', dest="learning_rate", default=0.1, type=float)
    parser.add_argument("--reload", action="store_true", default=False, required=False)
    parser.add_argument('--momentum', dest="momentum", default=0.9, type=float)
    parser.add_argument('--epochs', dest="epochs", default=10000, type=int,)
    parser.add_argument('--resdir', default="./prototypes", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args)
