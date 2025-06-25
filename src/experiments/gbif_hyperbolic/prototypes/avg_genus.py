import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from geoopt import PoincareBallExact

from src.constants import DEVICE
from src.shared.datasets import Dataset, DatasetVersion


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
    recon_prototypes = hierarchy.t() @ genus_means  # [S, D]
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

def compute_genus_prototypes(species_prototypes, hierarchy, eps=1e-8):
    genus_weights = (hierarchy / (torch.sum(hierarchy, dim=1, keepdim=True) + eps)).to(torch.float32)  # [G, S]
    genus_means = genus_weights @ species_prototypes  # [G, D]
    genus_means = F.normalize(genus_means, dim=1) / 2

    num_genus, num_species = hierarchy.shape

    final_prototypes = torch.zeros(sum(hierarchy.shape), genus_means.shape[1], device=species_prototypes.device)

    final_prototypes[:num_genus] = torch.Tensor(genus_means)
    final_prototypes[num_genus:] = torch.Tensor(species_prototypes)

    return final_prototypes


def run(args):
    ds = Dataset(args.dataset)
    ds.load(batch_size=args.batch_size, use_torch=True, reload=args.reload)

    genus_species_matrix = torch.tensor(ds.hierarchy[0]).to(torch.float32).to(DEVICE)
    num_genus, num_species = genus_species_matrix.shape

    prototypes = torch.randn(num_species, args.dims, device=DEVICE)
    prototypes = nn.Parameter(F.normalize(prototypes, p=2, dim=1))

    optimizer = optim.SGD([prototypes], lr=args.learning_rate, momentum=args.momentum)

    for i in range(args.epochs):
        optimizer.zero_grad()

        loss, sep_species, clos, sep_genus = prototype_loss(prototypes, genus_species_matrix)

        loss.backward()
        optimizer.step()

        with torch.no_grad():
            prototypes.data = F.normalize(prototypes.data, p=2, dim=1)

        l = loss.item()
        c = clos.item()
        print(f"{i} {args.epochs} {sep_species=} {sep_genus=} {c=} {l=}")


    prototypes = prototypes.data.cpu()

    # ball = PoincareBallExact(c=args.curvature)
    # final_prototypes = compute_genus_prototypes_hyperbolic(prototypes, genus_species_matrix.cpu(), ball)

    # final_prototypes = compute_genus_prototypes(prototypes, genus_species_matrix.cpu())

    # print(final_prototypes.shape)

    base = Path(f"./prototypes/{args.dataset}/avg_genus")
    f = base / f"{args.dims}.npy"
    np.save(f, prototypes.numpy())
    print(f"saved to {f}")


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
