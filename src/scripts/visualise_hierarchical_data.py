import json
import uuid

import numpy as np
import requests

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
        # Single hierarchy matrix
        return [data["data"].squeeze()]

    # Otherwise, assume multiple levels: level_0, level_1, ...
    hierarchy = [data[key] for key in sorted(data.files, key=lambda k: int(k.split("_")[1]))]
    return hierarchy


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




hierarchy = get_hierarchy(DatasetVersion.CLIBDB)
print(len(hierarchy))
print(hierarchy[0].shape)


metadata = get_metadata(DatasetVersion.CLIBDB)
print(metadata["per_level"][0].keys())

label_maps = [lvl["id2label"] for lvl in metadata["per_level"]]


def build_node(level, index):
    label = label_maps[level][str(index)]
    node = {"name": label}

    if level >= len(hierarchy):
        return node  # Leaf node

    row = hierarchy[level][index]  # This is a 1D array
    child_indices = np.where(row == 1)[0]

    if len(child_indices) > 0:
        node["children"] = [build_node(level + 1, i) for i in child_indices]

    return node

# Build the root node — assuming index 0 at level 0 is the root
root = build_node(0, 0)

# Save to file
with open("tree_data.json", "w") as f:
    json.dump(root, f, indent=2)
