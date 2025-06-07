from typing import cast

import datasets
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import cm
from matplotlib.colors import LogNorm

from src.constants import NUM_PROC
from src.shared.datasets import DatasetVersion

# Load dataset
VERSION = DatasetVersion.GBIF_GENUS_SPECIES_100K
path = f"jkbkaiser/{VERSION.value}"
dataset_dict = cast(datasets.DatasetDict, datasets.load_dataset(path, num_proc=NUM_PROC))

print("Got dataset")

# Collect all species labels
all_species = []
for split in ["train", "test", "valid"]:
    if split in dataset_dict:
        all_species.extend(dataset_dict[split]["species"])
all_species = np.array(all_species)

# Count occurrences per species
_, counts = np.unique(all_species, return_counts=True)

sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))

# Define log-spaced bins
bins = np.logspace(np.log10(1), np.log10(counts.max()), 50)

# Plot histogram with log bins
n, bins, patches = plt.hist(
    counts,
    bins=bins,
    edgecolor='black',
    color=sns.color_palette("deep")[0],
)

# Apply color gradient
# norm = plt.Normalize(n.min(), n.max())
norm = LogNorm(vmin=max(n.min(), 1), vmax=n.max())
cmap = cm.get_cmap("coolwarm")

for count, patch in zip(n, patches):
    patch.set_facecolor(cmap(norm(count)))

# Set log scale on x and y axes
plt.xscale("log")
plt.yscale("log")

sns.despine(left=False, bottom=False, top=True, right=True)
plt.grid(True, which="both", linestyle="--", linewidth=0.5)

# Labeling
plt.xlabel("Number of Occurrences per Species", fontsize=14)
plt.ylabel("Number of Species", fontsize=14)
# plt.title("Log-Log Histogram of Species Occurrence Frequencies", fontsize=20)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.tight_layout()

# Save and show
plt.savefig("long_tail_distribution.png", dpi=600)
plt.show()
