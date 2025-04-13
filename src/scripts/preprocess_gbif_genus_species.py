import argparse
import json
import random
from collections import Counter, defaultdict
from itertools import chain
from pathlib import Path
from typing import cast

import numpy as np
from datasets import Dataset, DatasetDict, Features, Image, Value, load_dataset
from google.cloud import storage

from src.constants import GOOGLE_BUCKET, GOOGLE_PROJECT, NUM_PROC
from src.shared.datasets import DatasetVersion

HUGGING_FACE_SOURCE_DATASET = "jkbkaiser/gbif_raw_100k"

VERSION = DatasetVersion.GBIF_GENUS_SPECIES_100K
HUGGING_FACE_PROCESSED_GENUS_SPECIES_DATASET = f"jkbkaiser/{VERSION.value}"

DATA_DIR = Path("./data")
GBIF_DATA_DIR = DATA_DIR / "gbif"
EXTRACTION_DIR = GBIF_DATA_DIR / "extraction"

if not EXTRACTION_DIR.exists():
    EXTRACTION_DIR.mkdir(parents=True)


def valid_species(entry):
    return len(entry["species"].split(" ")) >= 2


def extract_names_batched(batch):
    names = batch["species"]
    genus = []
    species = []

    for name in names:
        name_parts = name.split(" ")
        if len(name_parts) < 2:
            genus.append("")
            species.append("")
        else:
            genus.append(name_parts[0].lower())
            species.append(" ".join([part.lower() for part in name_parts]))

    batch["genus"] = genus
    batch["species"] = species
    return batch


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
    np.savez(source_file_name, data=hierarchies)

    client = storage.Client(GOOGLE_PROJECT)
    bucket = client.bucket(GOOGLE_BUCKET)
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(source_file_name)

    print(f"File uploaded to gs://{GOOGLE_BUCKET}/{blob_name}")


def get_genus_mapping(ds):
    geni = list(set(chain.from_iterable(ds.unique("genus").values())))
    geni2id = {elem: i for i, elem in enumerate(geni)}
    id2geni = {i: elem for i, elem in enumerate(geni)}
    return id2geni, geni2id


def get_species_mapping(ds):
    species = list(set(chain.from_iterable(ds.unique("species").values())))
    species2id = {elem: i for i, elem in enumerate(species)}
    id2species = {i: elem for i, elem in enumerate(species)}
    return id2species, species2id


def map_to_id(row, geni2id, species2id):
    row["genus"] = geni2id[row["genus"].lower()]
    row["species"] = species2id[row["species"].lower()]
    return row


def extract_hierarchy(ds):
    species_to_genus = {}
    genus_to_species = {}

    for row in ds["data"]:
        species = row["species"]
        genus = row["genus"]

        if species not in species_to_genus:
            species_to_genus[species] = genus
        elif species_to_genus[species] != genus:
            print("mismatch", species_to_genus[species], genus)

        if genus not in genus_to_species:
            genus_to_species[genus] = [species]
        else:
            if species not in genus_to_species[genus]:
                genus_to_species[genus] += [species]

    num_species = len(species_to_genus)
    num_genus = len(genus_to_species)

    genus_to_species_mask = np.zeros((num_genus, num_species))

    for key, values in genus_to_species.items():
        for value in values:
            genus_to_species_mask[key, value] = 1

    return [genus_to_species_mask]


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
    ds: DatasetDict = cast(DatasetDict, load_dataset(HUGGING_FACE_SOURCE_DATASET, num_proc=NUM_PROC))
    ds = ds.filter(valid_species, num_proc=NUM_PROC)
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
    ds = ds.map(extract_names_batched, batched=True, num_proc=NUM_PROC)

    genus = ds["data"]["genus"]
    uniq, counts = np.unique(genus, return_counts=True)
    label = uniq[np.argmax(counts)]
    print(f"most occuring genus before mapping: {label}")

    species = ds["data"]["species"]
    uniq, counts = np.unique(species, return_counts=True)
    label = uniq[np.argmax(counts)]
    print(f"most occuring species before mapping: {label}")

    id2genus, genus2id = get_genus_mapping(ds)
    id2species, species2id = get_species_mapping(ds)

    print(f"number of geni: {len(id2genus)}")
    print(f"number of species: {len(id2species)}")

    ds = ds.map(lambda x: map_to_id(x, genus2id, species2id))
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
    most_occuring_genus = id2genus[label]
    print(f"most occuring genus after mapping: {most_occuring_genus}")

    species = ds["data"]["species"]
    uniq, counts = np.unique(species, return_counts=True)
    label = uniq[np.argmax(counts)]
    most_occuring_species = id2species[label]
    print(f"most occuring species after mapping: {most_occuring_species}")

    hierarchy = extract_hierarchy(ds)

    print("extracted hierarchy, creating dict")

    ds_dict = stratified_custom_split(ds["data"])
    print(ds_dict)

    species_freq = get_fequencies(ds_dict, "species")
    genus_freq = get_fequencies(ds_dict, "genus")

    print("created dict, uploading ...")

    ds_dict.push_to_hub(HUGGING_FACE_PROCESSED_GENUS_SPECIES_DATASET, private=True)

    print("created metadata and hierarchies, uploading ...")

    metadata = {
        "per_level": [
            {
                "id2label": id2genus,
                "count": len(id2genus),
                "frequencies": genus_freq,
            },
            {
                "id2label": id2species,
                "count": len(id2species),
                "frequencies": species_freq,
            },
        ],
    }

    upload_metadata(metadata, f"{VERSION.value}/metadata.json")
    upload_hierarchies(hierarchies=hierarchy, blob_name=f"{VERSION.value}/hierarchy.npz")


def parse_args():
    parser = argparse.ArgumentParser(
        prog="GBIF hier dataset",
        description="Scripts process gbif dataset into a hierarchical dataset",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args)
