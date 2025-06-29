import json
import uuid
from enum import Enum

import numpy as np
import requests

from src.constants import CACHE_DIR, GOOGLE_BUCKET_URL


class DatasetSplit(str, Enum):
    TRAIN = "train"
    VALID = "valid"
    TEST = "test"


class DatasetVersion(str, Enum):
    GBIF_FLAT_10K = "gbif_flat_10k"
    GBIF_GENUS_SPECIES_10K = "gbif_genus_species_10k"
    GBIF_GENUS_SPECIES_10K_EMBEDDINGS = "gbif_genus_species_10k_embeddings"
    GBIF_GENUS_SPECIES_100K = "gbif_genus_species_100k"
    GBIF_COLEOPTERA_HIERARCHICAL_FULL = "gbif_coleoptera_hierarchical_full"
    BIOSCAN = "bioscan"
    BIOSCAN_UNSEEN = "bioscan_unseen"


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

    hierarchy = [data[key] for key in sorted(data.files, key=lambda k: int(k.split("_")[1]))]
    return hierarchy
