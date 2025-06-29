import os
import random

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

from src.shared.datasets import DatasetVersion, get_hierarchy


def compute_offsets(hierarchy_levels):
    level_sizes = [level.shape[0] for level in hierarchy_levels]
    level_sizes.append(hierarchy_levels[-1].shape[1])  # final species level
    offsets = [0]
    for size in level_sizes:
        offsets.append(offsets[-1] + size)
    return offsets, level_sizes


def visualize_all_tsne(proto_configs, dim, dataset_version, output_path, sample_size=1000):
    level_names = ["class", "order", "family", "subfamily", "genus", "species"]
    hierarchy_levels = get_hierarchy(dataset_version)
    offsets, level_sizes = compute_offsets(hierarchy_levels)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for idx, (proto_name, title) in enumerate(proto_configs):
        proto_path = f"./prototypes/clibdb/{proto_name}/{dim}.npy"
        if not os.path.exists(proto_path):
            print(f"[Warning] File not found: {proto_path}")
            continue

        prototypes = np.load(proto_path)

        level_labels = np.zeros(prototypes.shape[0], dtype=int)
        for i, (start, size) in enumerate(zip(offsets[:-1], level_sizes)):
            level_labels[start:start + size] = i

        indices = sorted(random.sample(range(len(prototypes)), min(sample_size, len(prototypes))))
        sampled_protos = prototypes[indices]
        sampled_labels = level_labels[indices]

        proj = TSNE(n_components=2, random_state=42).fit_transform(sampled_protos)
        proj -= proj.min(axis=0)
        proj /= proj.max(axis=0)
        proj = proj * 2 - 1

        ax = axes[idx]
        scatter = ax.scatter(proj[:, 0], proj[:, 1], c=sampled_labels, cmap="viridis", s=10)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)

    # Shared colorbar
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])
    norm = plt.Normalize(vmin=0, vmax=len(level_sizes) - 1)
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax, ticks=range(len(level_sizes)))
    cbar.set_label("Taxonomic Level")
    cbar.set_ticklabels(level_names)
    plt.tight_layout(rect=[0, 0, 0.85, 0.95])
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.show()


if __name__ == "__main__":
    proto_configs = [
        ("distortion", "Distortion"),
        ("entailment_cones", "Entailment Cones"),
        ("avg_multi", "Aggregate"),
        ("genus_species_poincare", "Poincaré"),
    ]
    dimensionality = 128
    dataset_version = DatasetVersion.BIOSCAN.value
    output_path = "./output_results/tsne_by_level_combined.png"

    visualize_all_tsne(proto_configs, dimensionality, dataset_version, output_path, sample_size=1700)
