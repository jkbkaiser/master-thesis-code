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
            raise Exception(f"Could not retrieve metadata for {dataset_version}")

        with open(path, "wb") as f:
            f.write(response.content)

    hierarchy = np.load(path)["data"]
    return hierarchy.squeeze()

# Load data
hierarchy = get_hierarchy(DatasetVersion.GBIF_GENUS_SPECIES_100K.value)
prototypes = np.load("./prototypes/gbif_genus_species_100k/distortion/128.npy")
prototypes_t = torch.tensor(prototypes)

print(prototypes.shape)

# Norms distribution
prototype_norms = prototypes_t.norm(dim=1)
print(f"prototype norm mean {prototype_norms.mean().item()}")
print(f"prototype norm max {prototype_norms.max().item()}")
print(f"prototype norm min {prototype_norms.min().item()}")
plt.hist(prototype_norms)
plt.show()

# TSNE projection
proj = TSNE(n_components=2).fit_transform(prototypes)
proj -= proj.min(axis=0)
proj /= proj.max(axis=0)
proj = proj * 2 - 1

num_genus = hierarchy.shape[0]
num_species = hierarchy.shape[1]
t = hierarchy.sum(axis=1)
chosen_genus_indices = (-t).argsort()[:10]

# Setup plot
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim(-1.1, 1.1)
ax.set_ylim(-1.1, 1.1)
ax.set_aspect("equal")
ax.axhline(0, color='gray', linewidth=0.5)
ax.axvline(0, color='gray', linewidth=0.5)
circle = plt.Circle((0, 0), 1, color='black', fill=False, linestyle='dashed')
ax.add_patch(circle)

# Define colors
colors = plt.cm.get_cmap("tab10", 10)

for idx, genus_idx in enumerate(chosen_genus_indices):
    genus_proto = proj[genus_idx]
    ax.scatter(*genus_proto, color=colors(idx), label=f"Genus {genus_idx}", edgecolor='black')

    # Get species belonging to this genus
    species_indices = np.where(hierarchy[genus_idx] == 1)[0]
    for species_rel_idx in species_indices:
        species_proto = proj[2883 + species_rel_idx]
        ax.scatter(*species_proto, color=colors(idx), alpha=0.5, s=30, zorder=2)

plt.legend()
plt.title("t-SNE of Prototypes")
plt.show()
