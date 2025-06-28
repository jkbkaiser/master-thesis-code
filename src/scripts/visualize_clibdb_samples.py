import json
import random
import uuid
from typing import cast

import datasets
import matplotlib.pyplot as plt
import numpy as np
import requests
import seaborn as sns
from matplotlib import cm
from matplotlib.colors import LogNorm

from src.constants import CACHE_DIR, GOOGLE_BUCKET_URL, NUM_PROC
from src.shared.datasets import DatasetVersion

VERSION = DatasetVersion.CLIBDB
path = f"jkbkaiser/{VERSION.value}"
dataset_dict = cast(datasets.DatasetDict, datasets.load_dataset(path, num_proc=NUM_PROC))

print(dataset_dict)

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


metadata = get_metadata(DatasetVersion.CLIBDB)
print(metadata["per_level"][0].keys())

label_maps = [lvl["id2label"] for lvl in metadata["per_level"]]

# Use the train split
train_dataset = dataset_dict["train"]

# Group by order
order_to_samples = {}
for example in train_dataset:
    order = example["order"]
    if order not in order_to_samples:
        order_to_samples[order] = []
    order_to_samples[order].append(example)

# Sample one image from six different orders
selected_orders = random.sample(list(order_to_samples.keys()), 10)
selected_images = [random.choice(order_to_samples[order]) for order in selected_orders]

from torchvision import transforms

# Define a center crop transform to a fixed size (e.g., 224x224)
crop_size = 224
center_crop = transforms.CenterCrop(crop_size)

fig, axs = plt.subplots(2, 5, figsize=(12, 8))
for ax, sample, order in zip(axs.flat, selected_images, selected_orders):
    image = sample["image"]  # already a PIL Image
    image = center_crop(image)  # center crop the image
    ax.imshow(image)
    label = label_maps[1][str(order)]
    ax.set_title(label.capitalize())
    ax.axis("off")

plt.tight_layout()
plt.savefig("clibdb_samples.png", dpi=800)
plt.show()

