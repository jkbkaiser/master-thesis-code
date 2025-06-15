import json
import uuid
from typing import cast

import datasets
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import requests

from src.constants import CACHE_DIR, GOOGLE_BUCKET_URL, NUM_PROC
from src.shared.datasets import DatasetVersion

VERSION = DatasetVersion.GBIF_GENUS_SPECIES_100K
path = f"jkbkaiser/{VERSION.value}"
dataset_dict = cast(datasets.DatasetDict, datasets.load_dataset(path, num_proc=NUM_PROC))


def get_metadata(version, reload: bool):
    directory = CACHE_DIR / version.value
    path = directory / "metadata.json"

    if reload or not path.is_file():
        url = f"{GOOGLE_BUCKET_URL}/{version.value}/metadata.json?id={uuid.uuid4()}"

        response = requests.get(url)
        if response.status_code != 200:
            raise Exception(f"Could not retrieve metadata for {version.value}, status: {response.status_code}")
        metadata = response.json()

        if not directory.exists():
            directory.mkdir(parents=True)

        with open(path, "w") as f:
            json.dump(metadata, f)

        return metadata

    with open(path, "r") as f:
        return json.load(f)

ds = dataset_dict["train"].select(range(10000))

indices = [
    [2558, 1305, 2196, 2321],
    [951, 5568, 7988],
    [461, 7038, 9043],
]
metadata = get_metadata(VERSION, False)
species_labels = metadata["per_level"][1]["id2label"]

flat_indices = [i for row in indices for i in row]
images = [(ds[i]["image"], ds[i]["species"]) for i in flat_indices]

fig = plt.figure(figsize=(18, 14))
gs = gridspec.GridSpec(3, 12, figure=fig, hspace=0.2)

for i in range(4):
    ax = fig.add_subplot(gs[0, i * 3:i * 3 + 3])
    img, spec = images[i]
    ax.imshow(img)
    title = species_labels[str(spec)].capitalize()
    ax.set_title(title, fontsize=16)
    ax.axis("off")

for i in range(3):
    ax = fig.add_subplot(gs[1, i * 4:i * 4 + 4])
    img, spec = images[i + 4]
    ax.imshow(img)
    title = species_labels[str(spec)].capitalize()
    ax.set_title(title, fontsize=16)
    ax.axis("off")

for i in range(3):
    ax = fig.add_subplot(gs[2, i * 4:i * 4 + 4])
    img, spec = images[i + 7]
    ax.imshow(img)
    title = species_labels[str(spec)].capitalize()
    ax.set_title(title, fontsize=16)
    ax.axis("off")


plt.subplots_adjust(
    left=0.03,
    right=0.97,
    top=0.95,
    bottom=0.05,
)
plt.show()
