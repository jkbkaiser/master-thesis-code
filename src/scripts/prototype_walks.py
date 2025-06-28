import os
import random
import uuid

import matplotlib.pyplot as plt
import numpy as np
import requests
import torch
from sklearn.manifold import TSNE

from src.constants import CACHE_DIR, GOOGLE_BUCKET_URL
from src.shared.datasets import DatasetVersion


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
        return [data["data"].squeeze()]
    return [data[key] for key in sorted(data.files, key=lambda k: int(k.split("_")[1]))]


def compute_offsets(hierarchy_levels):
    level_sizes = [level.shape[0] for level in hierarchy_levels]
    level_sizes.append(hierarchy_levels[-1].shape[1])  # final species level
    offsets = [0]
    for size in level_sizes:
        offsets.append(offsets[-1] + size)
    return offsets, level_sizes


def visualize_tsne_by_level(proto_name, dim, dataset_version, output_dir, sample_size=1000):
    level_names = ["class", "order", "family", "subfamily", "genus", "species"]
    hierarchy_levels = get_hierarchy(dataset_version)
    offsets, level_sizes = compute_offsets(hierarchy_levels)

    proto_path = f"./prototypes/clibdb/{proto_name}/{dim}.npy"
    if not os.path.exists(proto_path):
        raise FileNotFoundError(f"Prototype file not found: {proto_path}")

    prototypes = np.load(proto_path)

    # Assign taxonomic level to each prototype
    level_labels = np.zeros(prototypes.shape[0], dtype=int)
    for i, (start, size) in enumerate(zip(offsets[:-1], level_sizes)):
        level_labels[start:start + size] = i

    # Subsample
    indices = sorted(random.sample(range(len(prototypes)), min(sample_size, len(prototypes))))
    sampled_protos = prototypes[indices]
    sampled_labels = level_labels[indices]

    # t-SNE
    proj = TSNE(n_components=2, random_state=42).fit_transform(sampled_protos)
    proj -= proj.min(axis=0)
    proj /= proj.max(axis=0)
    proj = proj * 2 - 1

    # Plot
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(proj[:, 0], proj[:, 1], c=sampled_labels, cmap="viridis", s=16)
    cbar = plt.colorbar(scatter, ticks=range(len(level_sizes)))
    cbar.set_label("Taxonomic Level")
    cbar.set_ticklabels(level_names)
    plt.title(f"t-SNE Projection Colored by Taxonomic Level ({proto_name})")
    plt.xlabel("t-SNE Dim 1")
    plt.ylabel("t-SNE Dim 2")
    plt.grid(True)
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f"tsne_by_level_{proto_name}.png"))
    plt.show()


if __name__ == "__main__":
    proto_name = "distortion"  # Change to your prototype name
    dimensionality = 128
    dataset_version = DatasetVersion.CLIBDB.value
    output_dir = "./output_results/tsne_by_level"

    visualize_tsne_by_level(proto_name, dimensionality, dataset_version, output_dir, sample_size=1700)
