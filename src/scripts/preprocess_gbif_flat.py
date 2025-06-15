import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import cast

import numpy as np
from datasets import DatasetDict, Features, Image, Value, load_dataset
from google.cloud import storage

from src.constants import GOOGLE_BUCKET, GOOGLE_PROJECT
from src.shared.datasets import DatasetVersion

HUGGING_FACE_SOURCE_DATASET = "jkbkaiser/gbif_raw_10k"

VERSION = DatasetVersion.GBIF_FLAT_10K
HUGGING_FACE_PROCESSED_FLAT_DATASET = f"jkbkaiser/{VERSION.value}"

DATA_DIR = Path("./data")
GBIF_DATA_DIR = DATA_DIR / "gbif"
EXTRACTION_DIR = GBIF_DATA_DIR / "extraction"


def valid_species(entry):
    return len(entry["species"].split(" ")) >= 2


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


def extract_names(entry):
    name_parts = entry["species"].split(" ")

    if len(name_parts) != 2:
        print("Could not extract", name_parts)

    return {
        "image": entry["image"],
        "genus": name_parts[0].lower(),
        "species": " ".join([name.lower() for name in name_parts]),
    }


def get_flat_mapping(ds):
    ds = ds.with_format("numpy")

    uniq_genus = ds.unique("genus")["data"]
    uniq_species = ds.unique("species")["data"]

    combined = uniq_genus + uniq_species

    split = len(uniq_genus)

    id2labels = {i: elem for i, elem in enumerate(combined)}
    labels2id = {elem: i for i, elem in enumerate(combined)}

    return id2labels, labels2id, split


def get_fequencies(dataset):
    counts = Counter()
    for split in dataset.keys():
        for example in dataset[split]:
            species = example['species']
            genus = example['genus']
            counts[species] += 1
            counts[genus] += 1
    return dict(counts)


def map_to_label(row, labels2id):
    row["genus"] = labels2id[row["genus"].lower()]
    row["species"] = labels2id[row["species"].lower()]
    return row


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


def run(args):
    ds: DatasetDict = cast(DatasetDict, load_dataset(HUGGING_FACE_SOURCE_DATASET))
    ds = ds.filter(valid_species)

    ds = ds.remove_columns(
        [
            "kingdom_key",
            "phylum_key",
            "order_key",
            "family_key",
            "genus_key",
            "scientific_name",
            "id",
            "sex",
            "life_stage",
            "continent",
        ]
    )
    ds = ds.map(extract_names)

    genus = ds["data"]["genus"]
    uniq, counts = np.unique(genus, return_counts=True)
    num_geni = len(uniq)
    label = uniq[np.argmax(counts)]
    print(f"most occuring genus before mapping: {label}")

    species = ds["data"]["species"]
    uniq, counts = np.unique(species, return_counts=True)
    num_species = len(uniq)
    label = uniq[np.argmax(counts)]
    print(f"most occuring species before mapping: {label}")

    id2label, label2id, split = get_flat_mapping(ds)

    print(f"Split: {split}")
    print(f"number of geni: {num_geni}")
    print(f"number of species: {num_species}")
    print(f"number of labels: {len(id2label)}")

    ds = ds.map(lambda x: map_to_label(x, label2id))
    features = Features(
        {
            "image": Image(mode=None, decode=True, id=None),
            "species": Value(dtype="int32", id=None),
            "genus": Value(dtype="int32", id=None),
        }
    )
    ds = ds.cast(features)

    genus = ds["data"]["genus"]
    uniq, counts = np.unique(genus, return_counts=True)
    label = uniq[np.argmax(counts)]
    most_occuring_genus = id2label[label]
    print(f"most occuring genus after mapping: {most_occuring_genus}")

    species = ds["data"]["species"]
    uniq, counts = np.unique(species, return_counts=True)
    label = uniq[np.argmax(counts)]
    most_occuring_species = id2label[label]
    print(f"most occuring species after mapping: {most_occuring_species}")


    ds_dict = stratified_custom_split(ds["data"])
    print(ds_dict)

    ds_dict.push_to_hub(HUGGING_FACE_PROCESSED_FLAT_DATASET, private=True)

    freq = get_fequencies(ds_dict)

    metadata = {
        "per_level": [
            {
                "id2label": id2label,
                "count": len(id2label),
                "split": split,
                "frequencies": freq,
            },
        ],
    }

    upload_metadata(metadata, f"{VERSION.value}/metadata.json")


def parse_args():
    parser = argparse.ArgumentParser(
        prog="GBIF flat dataset",
        description="Scripts process gbif dataset into a flat hierarchical dataset",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args)
