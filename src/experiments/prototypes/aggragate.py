import argparse
import os
from pathlib import Path

import mlflow
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dotenv import load_dotenv

from src.constants import DEVICE
from src.shared.datasets import ClibdbDataset, Dataset, DatasetVersion

load_dotenv()

MLFLOW_SERVER = os.environ["MLFLOW_SERVER"]
CHECKPOINT_DIR = os.environ["CHECKPOINT_DIR"]

def prototype_loss(prototypes, hierarchy, alpha=0.5, beta=0.3, gamma=0.2, eps=1e-8):
    prototypes = F.normalize(prototypes, dim=1)  # [S, D]

    # ---------- Push term (species) ----------
    product = torch.matmul(prototypes, prototypes.t()) + 1
    product -= 2. * torch.diag(torch.diag(product))
    push_loss_species = product.max(dim=1)[0].mean()

    # ---------- Pull term (species -> genus center) ----------
    genus_weights = (hierarchy / (torch.sum(hierarchy, dim=1, keepdim=True) + eps)).to(torch.float32)  # [G, S]
    genus_means = genus_weights @ prototypes  # [G, D]
    genus_means = F.normalize(genus_means, dim=1)

    # Map each species to its genus mean (reconstructed prototypes)
    recon_prototypes = hierarchy.t().to(torch.float32) @ genus_means  # [S, D]
    recon_prototypes = F.normalize(recon_prototypes, dim=1)

    # Cosine similarity between original and reconstructed (pull toward genus center)
    pull_sim = (prototypes * recon_prototypes).sum(dim=1)  # [S]
    closeless = pull_sim.mean()
    pull_loss = 1.0 - closeless

    # ---------- Push term (reconstructed prototypes/genus) ----------
    # Compute pairwise cosine similarity for genus prototypes (recon_prototypes)
    product_genus = torch.matmul(recon_prototypes, recon_prototypes.t()) + 1
    product_genus -= 2. * torch.diag(torch.diag(product_genus))  # Remove diagonal
    push_loss_genus = product_genus.max(dim=1)[0].mean()

    # ---------- Combined loss ----------
    loss = alpha * push_loss_species + beta * pull_loss + gamma * push_loss_genus

    return loss, push_loss_species.item(), closeless, push_loss_genus.item()

# def compute_genus_prototypes(species_prototypes, hierarchy, eps=1e-8):
#     genus_weights = (hierarchy / (torch.sum(hierarchy, dim=1, keepdim=True) + eps)).to(torch.float32)  # [G, S]
#     genus_means = genus_weights @ species_prototypes  # [G, D]
#     genus_means = F.normalize(genus_means, dim=1) / 2
#
#     num_genus, num_species = hierarchy.shape
#
#     final_prototypes = torch.zeros(sum(hierarchy.shape), genus_means.shape[1], device=species_prototypes.device)
#
#     final_prototypes[:num_genus] = torch.Tensor(genus_means)
#     final_prototypes[num_genus:] = torch.Tensor(species_prototypes)
#
#     return final_prototypes

def compute_hierarchical_prototypes(leaf_prototypes, hierarchy_levels, scale_base=2.0, eps=1e-8):
    num_levels = len(hierarchy_levels) + 1  # include species
    # Create increasing norm schedule: [0.0, ..., 1.0]
    norms = [1 - 1 / (scale_base ** (num_levels - i - 1)) for i in range(num_levels)]

    all_prototypes = []
    current_prototypes = leaf_prototypes
    current_level = 0

    all_prototypes.append(current_prototypes * norms[current_level])

    for i, matrix in enumerate(reversed(hierarchy_levels)):  # from leaf to root
        current_level += 1

        parent_count, child_count = matrix.shape
        matrix = matrix.to(leaf_prototypes.device).to(torch.float32)
        weights = matrix / (matrix.sum(dim=1, keepdim=True) + eps)  # Normalize
        parent_prototypes = weights @ current_prototypes
        parent_prototypes = F.normalize(parent_prototypes, dim=1)

        parent_prototypes = parent_prototypes * norms[current_level]

        all_prototypes.append(parent_prototypes)
        current_prototypes = parent_prototypes

    return torch.cat(all_prototypes[::-1], dim=0)


# def compute_hierarchical_prototypes(leaf_prototypes, hierarchy_levels, scale_base=2.0, eps=1e-8):
#     all_prototypes = []
#     current_prototypes = leaf_prototypes
#     current_norm = 1.0  # Start at unit norm
#
#     all_prototypes.append(current_prototypes * current_norm)
#
#     for i, matrix in enumerate(reversed(hierarchy_levels)):  # from leaf to root
#         parent_count, child_count = matrix.shape
#         matrix = matrix.to(leaf_prototypes.device).to(torch.float32)
#         weights = matrix / (matrix.sum(dim=1, keepdim=True) + eps)  # Normalize
#         parent_prototypes = weights @ current_prototypes  # [N_parent, D]
#         parent_prototypes = F.normalize(parent_prototypes, dim=1)
#
#         current_norm = current_norm / scale_base  # Decay exponentially
#         parent_prototypes = parent_prototypes * current_norm
#
#         all_prototypes.append(parent_prototypes)
#         current_prototypes = parent_prototypes
#
#     # Reverse again so root is first
#     return torch.cat(all_prototypes[::-1], dim=0)

def run(args):
    # ds = Dataset(args.dataset)
    ds = ClibdbDataset(args.dataset)
    ds.load(batch_size=args.batch_size, use_torch=True, reload=args.reload)

    hierarchy_levels = [torch.tensor(h).to(DEVICE) for h in ds.hierarchy]  # e.g. [G←S, F←G, ...]

    mlflow.set_tracking_uri(MLFLOW_SERVER)
    mlflow.set_experiment("prototypes")

    leaf_classes = hierarchy_levels[-1].shape[1]  # S (num species)
    prototypes = torch.randn(leaf_classes, args.dims, device=DEVICE)
    prototypes = nn.Parameter(F.normalize(prototypes, p=2, dim=1))

    optimizer = optim.SGD([prototypes], lr=args.learning_rate, momentum=args.momentum)

    with mlflow.start_run():
        mlflow.log_params({
            "type": "avg_multi",
            "lr": args.learning_rate,
            "epochs": args.epochs,
            "entailment_optimizer": "sgd",
        })

        for i in range(args.epochs):
            optimizer.zero_grad()

            # Use only bottom matrix for loss: push/pull among leaves & their parents
            loss, sep_species, closeness, sep_genus = prototype_loss(prototypes, hierarchy_levels[-1])
            loss.backward()
            optimizer.step()

            if i % 100 == 0:
                mlflow.log_metrics({
                    "loss": loss.item(),
                }, step=i)

            with torch.no_grad():
                prototypes.data = F.normalize(prototypes.data, p=2, dim=1)

            print(f"{i} / {args.epochs} {sep_species=} {sep_genus=} {closeness=} {loss.item()=}")

        # Final hierarchical prototypes
        with torch.no_grad():
            full_prototypes = compute_hierarchical_prototypes(
                prototypes.data,
                hierarchy_levels=hierarchy_levels,
                scale_base=2.0
            ).cpu()

        out_path = args.resdir / args.dataset.value / f"avg_multi/{args.dims}.npy"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_path, full_prototypes.numpy())
        print(f"Saved to {out_path}")


#
# def run(args):
#     ds = Dataset(args.dataset)
#     ds.load(batch_size=args.batch_size, use_torch=True, reload=args.reload)
#
#     genus_species_matrix = torch.tensor(ds.hierarchy[0]).to(torch.float32).to(DEVICE)
#     num_genus, num_species = genus_species_matrix.shape
#
#     prototypes = torch.randn(num_species, args.dims, device=DEVICE)
#     prototypes = nn.Parameter(F.normalize(prototypes, p=2, dim=1))
#
#     optimizer = optim.SGD([prototypes], lr=args.learning_rate, momentum=args.momentum)
#
#     for i in range(args.epochs):
#         optimizer.zero_grad()
#
#         loss, sep_species, clos, sep_genus = prototype_loss(prototypes, genus_species_matrix)
#
#         loss.backward()
#         optimizer.step()
#
#         with torch.no_grad():
#             prototypes.data = F.normalize(prototypes.data, p=2, dim=1)
#
#         l = loss.item()
#         c = clos.item()
#         print(f"{i} {args.epochs} {sep_species=} {sep_genus=} {c=} {l=}")
#
#
#     prototypes = prototypes.data.cpu()
#
#     base = Path(f"./prototypes/{args.dataset}/avg_genus")
#     f = base / f"{args.dims}.npy"
#     np.save(f, prototypes.numpy())
#     print(f"saved to {f}")


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
    parser.add_argument('--lr', dest="learning_rate", default=1, type=float)
    parser.add_argument("--reload", action="store_true", default=False, required=False)
    parser.add_argument('--momentum', dest="momentum", default=0.9, type=float)
    parser.add_argument('--epochs', dest="epochs", default=10000, type=int,)
    parser.add_argument('--resdir', default="./prototypes", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args)
