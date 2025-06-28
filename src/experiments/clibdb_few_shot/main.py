import json
import uuid

import datasets
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


def run():
    dataset_dict = datasets.load_dataset("jkbkaiser/clibdb_unseen")
    hierarchy = get_hierarchy(DatasetVersion.CLIBDB_UNSEEN)

    print(dataset_dict)
    print(hierarchy)

if __name__ == "__main__":
    run()
