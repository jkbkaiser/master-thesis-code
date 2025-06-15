from typing import cast

import datasets
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from src.constants import NUM_PROC
from src.shared.datasets import DatasetVersion

VERSION = DatasetVersion.GBIF_GENUS_SPECIES_100K
path = f"jkbkaiser/{VERSION.value}"

dataset_dict = cast(datasets.DatasetDict, datasets.load_dataset(path, num_proc=NUM_PROC))

all_species = []
for split in ["train", "test", "valid"]:
    if split in dataset_dict:
        all_species.extend(dataset_dict[split]["species"])
all_species = np.array(all_species)

t, counts = np.unique(all_species, return_counts=True)

thresholds = [100, 50, 10, 5]
counts_per_threshold = [np.sum(counts > thresh) for thresh in thresholds]

bins = [0, 5, 10, 50, 100, 1000]
labels = []

for i, bin in enumerate(bins[:-1]):
    labels.append(f"{bin}-{bins[i+1]} occurences")

bin_indices = np.digitize(counts, bins, right=True)
counts_per_bin = [(bin_indices == i).sum() for i in range(1, len(bins))]

colors = sns.color_palette("deep")

plt.figure(figsize=(8, 8))
wedges, texts, autotexts = plt.pie(
    counts_per_bin,
    autopct="%1.1f%%",
    startangle=0,
    colors=colors,
    wedgeprops={"edgecolor": "white"},
    pctdistance=1.1,
)

centre_circle = plt.Circle((0, 0), 0.50, fc="white")
plt.gca().add_artist(centre_circle)

for text in texts + autotexts:
    text.set_fontsize(14)

plt.title("Distribution of Species by Occurrence Frequency", fontsize=20)

plt.legend(
    wedges,
    labels, 
    loc="upper right",
    bbox_to_anchor=(1.1, 0.95),
    fontsize=14
)

plt.savefig("species_distribution_100k.png", dpi=300, bbox_inches="tight")
plt.show()
