import random
from typing import cast

import datasets
import matplotlib.pyplot as plt
from torchvision import transforms

from src.constants import NUM_PROC
from src.shared.datasets import DatasetVersion, get_metadata

VERSION = DatasetVersion.BIOSCAN
path = f"jkbkaiser/{VERSION.value}"
dataset_dict = cast(datasets.DatasetDict, datasets.load_dataset(path, num_proc=NUM_PROC))
metadata = get_metadata(DatasetVersion.BIOSCAN)
label_maps = [lvl["id2label"] for lvl in metadata["per_level"]]
train_dataset = dataset_dict["train"]


order_to_samples = {}
for example in train_dataset:
    order = example["order"]
    if order not in order_to_samples:
        order_to_samples[order] = []
    order_to_samples[order].append(example)

selected_orders = random.sample(list(order_to_samples.keys()), 10)
selected_images = [random.choice(order_to_samples[order]) for order in selected_orders]


crop_size = 224
center_crop = transforms.CenterCrop(crop_size)

fig, axs = plt.subplots(2, 5, figsize=(12, 8))
for ax, sample, order in zip(axs.flat, selected_images, selected_orders):
    image = sample["image"]
    image = center_crop(image)
    ax.imshow(image)
    label = label_maps[1][str(order)]
    ax.set_title(label.capitalize())
    ax.axis("off")

plt.tight_layout()
plt.savefig("clibdb_samples.png", dpi=800)
plt.show()
