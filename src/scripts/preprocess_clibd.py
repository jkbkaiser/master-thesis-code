import argparse
import json
import random
from collections import Counter, defaultdict
from itertools import chain
from pathlib import Path
from typing import cast

import numpy as np
from datasets import DatasetDict, Features, Image, Value, load_dataset
from google.cloud import storage

from src.constants import GOOGLE_BUCKET, GOOGLE_PROJECT, NUM_PROC
from src.shared.datasets import DatasetVersion

RANKS = ["class", "order", "family", "subfamily", "genus", "species"]

HUGGING_FACE_SOURCE_DATASET = "jkbkaiser/clibd-raw"

VERSION = DatasetVersion.CLIBDB
HUGGING_FACE_PROCESSED_GENUS_SPECIES_DATASET = f"jkbkaiser/{VERSION.value}"

DATA_DIR = Path("./data")
GBIF_DATA_DIR = DATA_DIR / "gbif"
EXTRACTION_DIR = GBIF_DATA_DIR / "extraction"

if not EXTRACTION_DIR.exists():
    EXTRACTION_DIR.mkdir(parents=True)


def extract_names(entry):
    name_parts = entry["species"].split(" ")

    if len(name_parts) < 2:
        print("Could not extract", name_parts)

    return {
        "image": entry["image"],
        "genus": name_parts[0].lower(),
        "species": " ".join([name.lower() for name in name_parts]),
    }


def upload_metadata(mappings, destination_blob_name):
    d = (EXTRACTION_DIR / f"{VERSION.value}")
    if not d.exists():
        d.mkdir(parents=True)
    source_file_name = EXTRACTION_DIR / destination_blob_name

    with open(source_file_name, "w") as f:
        f.write(json.dumps(mappings))

    storage_client = storage.Client(GOOGLE_PROJECT)
    bucket = storage_client.bucket(GOOGLE_BUCKET)
    blob = bucket.blob(destination_blob_name)

    if blob.exists():
        print("deleted existing blob")
        blob.delete()

    blob.cache_control = "no-store,no-cache,max-age=0"
    blob.upload_from_filename(source_file_name, if_generation_match=None)
    print(f"File {source_file_name} uploaded to {destination_blob_name}.")


def upload_hierarchies(hierarchies, blob_name):
    d = (EXTRACTION_DIR / f"{VERSION.value}")
    if not d.exists():
        d.mkdir(parents=True)
    source_file_name = EXTRACTION_DIR / blob_name

    np.savez(source_file_name, **{f"level_{i}": m for i, m in enumerate(hierarchies)})

    client = storage.Client(GOOGLE_PROJECT)
    bucket = client.bucket(GOOGLE_BUCKET)
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(source_file_name)

    print(f"File uploaded to gs://{GOOGLE_BUCKET}/{blob_name}")


def get_label_mappings(dataset_dict, level):
    all_values = list(chain.from_iterable(dataset_dict[split][level] for split in dataset_dict))
    unique_values = sorted(set(v.lower() for v in all_values))
    label2id = {v: i for i, v in enumerate(unique_values)}
    id2label = {i: v for v, i in label2id.items()}
    return id2label, label2id


def map_taxonomy_to_ids(row, label2id_dict):
    for level in RANKS:
        row[level] = label2id_dict[level][row[level].lower()]
    return row


def extract_hierarchy_tensor(ds_dict, label2id_dict):
    parent_masks = []

    # Combine all rows from all splits into one iterable
    all_rows = []
    for split in ds_dict:
        all_rows.extend(ds_dict[split])

    for i in range(len(RANKS) - 1):
        parent_level = RANKS[i]
        child_level = RANKS[i + 1]

        num_parents = len(label2id_dict[parent_level])
        num_children = len(label2id_dict[child_level])

        mask = np.zeros((num_parents, num_children), dtype=np.float32)

        for row in all_rows:
            parent_id = row[parent_level]
            child_id = row[child_level]
            mask[parent_id, child_id] = 1

        parent_masks.append(mask)

    return parent_masks


def get_fequencies(dataset, key):
    counts = Counter()
    for split in dataset.keys():
        for example in dataset[split]:
            val = example[key]
            counts[val] += 1
    return dict(counts)


def stratified_custom_split(dataset, label_col="species", seed=42):
    random.seed(seed)
    label2indices = defaultdict(list)

    for idx, example in enumerate(dataset):
        print(idx, example, label_col)
        label2indices[example[label_col]].append(idx)

    train_indices, valid_indices, test_indices = [], [], []

    for label, indices in label2indices.items():
        if len(indices) < 3:
            continue

        random.shuffle(indices)

        n = len(indices)
        n_train = int(n * 5 / 7)
        n_valid = int(n * 1 / 7)

        train_indices.extend(indices[:n_train])
        valid_indices.extend(indices[n_train:n_train + n_valid])
        test_indices.extend(indices[n_train + n_valid:])

    ds_dict = DatasetDict({
        "train": dataset.select(train_indices),
        "valid": dataset.select(valid_indices),
        "test": dataset.select(test_indices)
    })

    return ds_dict


def run(_):
    ds_dict: DatasetDict = cast(DatasetDict, load_dataset(HUGGING_FACE_SOURCE_DATASET, num_proc=NUM_PROC))
    print(ds_dict)

    id2label_dict = {}
    label2id_dict = {}

    for level in RANKS:
        id2label, label2id = get_label_mappings(ds_dict, level)
        id2label_dict[level] = id2label
        label2id_dict[level] = label2id

    for split in ds_dict:
        ds_dict[split] = ds_dict[split].map(
            lambda row: map_taxonomy_to_ids(row, label2id_dict),
            num_proc=NUM_PROC,
            desc=f"Mapping taxonomy for {split}"
        )

    features = Features({
        "image": Image(),
        **{level: Value(dtype="int32") for level in RANKS}
    })

    ds_dict = ds_dict.cast(features)

    hierarchies = extract_hierarchy_tensor(ds_dict, label2id_dict)

    frequencies = {
        level: get_fequencies(ds_dict, level) for level in RANKS
    }

    ds_dict.push_to_hub(HUGGING_FACE_PROCESSED_GENUS_SPECIES_DATASET, private=True)

    metadata = {
        "per_level": [
            {
                "id2label": id2label_dict[level],
                "count": len(id2label_dict[level]),
                "frequencies": frequencies[level],
            } for level in RANKS
        ]
    }

    upload_metadata(metadata, f"{VERSION.value}/metadata.json")
    upload_hierarchies(hierarchies=hierarchies, blob_name=f"{VERSION.value}/hierarchy.npz")


def parse_args():
    parser = argparse.ArgumentParser(
        prog="GBIF hier dataset",
        description="Scripts process gbif dataset into a hierarchical dataset",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args)
