import json
import math
from collections import defaultdict
from typing import cast

import datasets
import matplotlib.pyplot as plt

from src.constants import NUM_PROC
from src.shared.datasets import DatasetVersion

VERSION = DatasetVersion.GBIF_GENUS_SPECIES_100K
path = f"jkbkaiser/{VERSION.value}"
dataset_dict = cast(datasets.DatasetDict, datasets.load_dataset(path, num_proc=NUM_PROC))

ds = dataset_dict["train"].select(range(10000))

genus_map = defaultdict(lambda: defaultdict(list))

for i, row in enumerate(ds):
    genus_map[row["genus"]][row["species"]].append(i)

print(json.dumps(genus_map, indent=2))

candidate_genera = {
    genus: species_map
    for genus, species_map in genus_map.items()
}

print(f"Found {len(candidate_genera)} candidate genera")

row_id = 0
images_per_page = 10
cols = 5
quit_flag = False

for genus, species_map in candidate_genera.items():
    for species, indices in species_map.items():
        if quit_flag:
            break

        images = [(i, ds[i]["image"]) for i in indices]
        num_images = len(images)

        for page_start in range(0, num_images, images_per_page):
            page_end = page_start + images_per_page
            page_images = images[page_start:page_end]

            rows = math.ceil(len(page_images) / cols)
            fig, axs = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
            axs = axs.flatten() if rows > 1 else [axs]

            try:
                for i, (sample_id, img) in enumerate(page_images):
                    print(type(axs[i]))
                    axs[i].imshow(img)
                    axs[i].axis("off")
                    axs[i].set_title(f"Sample_id {sample_id}", fontsize=10)
            except:
                pass

            for i in range(len(page_images), len(axs)):
                axs[i].axis("off")

            page_num = page_start // images_per_page + 1
            total_pages = math.ceil(num_images / images_per_page)

            fig.suptitle(
                f"[{row_id}] species: {species} genus: {genus} — Page {page_num}/{total_pages} — {num_images} total images",
                fontsize=16
            )
            plt.tight_layout()
            plt.show()

            action = input("Press [Enter] for next page, [n] for next species, [q] to quit: ").strip().lower()

            if action == "n":
                break

        row_id += 1
