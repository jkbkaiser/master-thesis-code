import json
import os
import random
import uuid

import datasets
import geoopt
import matplotlib.pyplot as plt
import numpy as np
import requests
import seaborn as sns
import torch
from sklearn.manifold import TSNE

from src.constants import CACHE_DIR, GOOGLE_BUCKET_URL
from src.shared.datasets import DatasetVersion

labels = True

# Define directory for saving results and images
# prototype_name = "entailment_cones"  # Example prototype name, adjust as needed
prototype_name = "genus_species_poincare"  # Example prototype name, adjust as needed
dimensionality = 128  # Example dimensionality, adjust as needed


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



import matplotlib.cm as cm

name_map = {
    "genus_species_poincare": "Poincaré",
    "entailment_cones": "Entailment Cones",
    "distortion": "Distortion"
}

ball = geoopt.PoincareBallExact(c=1.5)
hierarchy_levels = get_hierarchy(DatasetVersion.CLIBDB.value)

# Define prototype names and shared colormap
prototype_names = ["genus_species_poincare", "entailment_cones", "distortion"]
colors = cm.tab10.colors  # 10 distinct colors for levels

level_names = ["class", "order", "family", "subfamily", "genus", "species"]
level_sizes = [level.shape[0] for level in hierarchy_levels]
level_sizes.append(hierarchy_levels[-1].shape[1])

# Compute offsets for slicing
offsets = [0]
for size in level_sizes:
    offsets.append(offsets[-1] + size)

print(len(offsets), len(level_names), len(level_sizes))


fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 9), sharex=True)

for idx, prototype_name in enumerate(prototype_names):
    # Load prototypes and hierarchy
    prototypes = np.load(f"./prototypes/clibdb/{prototype_name}/{dimensionality}.npy")
    prototypes_t = torch.tensor(prototypes)
    per_level_norms = {}

    offsets = [0]
    for size in level_sizes:
        offsets.append(offsets[-1] + size)

    for i, name in enumerate(level_names):
        start, end = offsets[i], offsets[i + 1]
        level_protos = prototypes_t[start:end]
        dists = ball.dist0(level_protos).detach().cpu().numpy()
        per_level_norms[name] = dists

    # Plot each level’s norms
    ax = axes[idx]
    for i, (name, norms) in enumerate(per_level_norms.items()):
        ax.hist(norms, bins=30, alpha=0.5, label=name if idx == 0 else "", color=colors[i])

    ax.set_ylabel(name_map.get(prototype_name, prototype_name), fontsize=12)
    ax.grid(True)

# axes[-1].set_xlabel("Hyperbolic Norm (distance to origin)", fontsize=12)

# Add single legend BELOW all subplots
handles, labels_ = axes[0].get_legend_handles_labels()
fig.legend(handles, labels_, loc="lower center", ncol=6, bbox_to_anchor=(0.5, 0), fontsize=12)

# Adjust layout so legend has space
plt.tight_layout(rect=[0, 0.07, 1, 1])  # Make sure bottom (0.15) is large enough

sns.despine()

# Save and include extra space in export
plt.savefig("./output_results/hyperbolic_norms_per_level_all.png", dpi=600, bbox_inches="tight")

plt.show()
