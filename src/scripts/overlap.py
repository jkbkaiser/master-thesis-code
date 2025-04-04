from typing import Set, cast

import datasets

from src.constants import NUM_PROC
from src.shared.datasets import DatasetVersion

VERSION = DatasetVersion.GBIF_GENUS_SPECIES_100K

path = f"jkbkaiser/{VERSION.value}"

dataset_dict = cast(datasets.DatasetDict, datasets.load_dataset(path, num_proc=NUM_PROC))

# Extract unique species IDs
train_species: Set[int] = set(dataset_dict["train"]["species"])
val_species: Set[int] = set(dataset_dict["valid"]["species"])

# Compute overlap and differences
common_species = train_species & val_species  # Intersection
val_only_species = val_species - common_species  # Species only in validation

# Print results
print(f"Total species in training: {len(train_species)}")
print(f"Total species in validation: {len(val_species)}")
print(f"Species overlap: {len(common_species)} ({len(common_species) / len(val_species) * 100:.2f}%)")
print(f"Species only in validation: {len(val_only_species)} ({len(val_only_species) / len(val_species) * 100:.2f}%)")

