import argparse
import json
import os
from itertools import chain
from pathlib import Path
from typing import cast

import numpy as np
from datasets import DatasetDict, Features, Image, Value, load_dataset
from dotenv import load_dotenv
from google.cloud import storage

load_dotenv()

HUGGING_FACE_SOURCE_DATASET = "jkbkaiser/thesis-gbif-raw-large"
HUGGING_FACE_PROCESSED_HIER_DATASET = "jkbkaiser/thesis-gbif-hier-large"
GOOGLE_PROJECT = os.getenv("GOOGLE_PROJECT")
GOOGLE_BUCKET_URL = os.getenv("GOOGLE_BUCKET_URL")
BUCKET_NAME = "thesis-gbif-mappings-large"

DATA_DIR = Path("./data")
GBIF_DATA_DIR = DATA_DIR / "gbif"
EXTRACTION_DIR = GBIF_DATA_DIR / "extraction"

if not EXTRACTION_DIR.exists():
    EXTRACTION_DIR.mkdir(parents=True)


def valid_species(entry):
    return len(entry["species"].split(" ")) >= 2


def extract_names(entry):
    name_parts = entry["species"].split(" ")

    if len(name_parts) < 2:
        print("Could not extract", name_parts)

    return {
        "image": entry["image"],
        "genus": name_parts[0].lower(),
        "species": " ".join([name.lower() for name in name_parts]),
    }


def upload_mappings(mappings, destination_blob_name):
    source_file_name = EXTRACTION_DIR / destination_blob_name
    with open(source_file_name, "w") as f:
        f.write(json.dumps(mappings))

    storage_client = storage.Client(GOOGLE_PROJECT)
    bucket = storage_client.bucket(BUCKET_NAME)
    blob = bucket.blob(destination_blob_name)

    if blob.exists():
        print("deleted existing blob")
        blob.delete()

    blob.cache_control = "no-store,no-cache,max-age=0"
    blob.upload_from_filename(source_file_name, if_generation_match=None)
    print(f"File {source_file_name} uploaded to {destination_blob_name}.")


def upload_numpy_array(mask, destination_blob_name):
    source_file_name = EXTRACTION_DIR / destination_blob_name
    with open(source_file_name, "wb") as f:
        np.save(f, mask)

    storage_client = storage.Client(GOOGLE_PROJECT)
    bucket = storage_client.bucket(BUCKET_NAME)
    blob = bucket.blob(destination_blob_name)

    if blob.exists():
        print("deleted existing blob")
        blob.delete()

    blob.cache_control = "no-store,no-cache,max-age=0"
    blob.upload_from_filename(source_file_name, if_generation_match=None)
    print(f"File {source_file_name} uploaded to {destination_blob_name}.")


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
    genus_to_species_masks = np.zeros((num_genus, num_species))

    for key, values in genus_to_species.items():
        for value in values:
            genus_to_species_masks[key, value] = 1

    species_to_genus_mask = np.zeros(num_species, dtype=np.int32)
    for key, value in species_to_genus.items():
        species_to_genus_mask[key] = value

    return genus_to_species_masks, species_to_genus_mask


def extract_frequencies(species):
    num_samples = [5, 10, 50, 100, 500, 1000]
    _, counts = np.unique(species, return_counts=True)

    # Check that uniq is in order!

    a = []
    for n in num_samples:
        mask = counts < n
        a.append(mask)

    return np.stack(a)


def run(_):
    ds: DatasetDict = cast(DatasetDict, load_dataset(HUGGING_FACE_SOURCE_DATASET))
    ds = ds.filter(valid_species)
    ds = ds.map(extract_names)
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

    freq = extract_frequencies(species)

    genus_to_species_mask, species_to_genus_mask = extract_hierarchy(ds)

    ds_train_validtest = ds["data"].train_test_split(test_size=0.2, seed=42)
    ds_validtest = ds_train_validtest["test"].train_test_split(test_size=0.5, seed=42)
    ds_dict = DatasetDict(
        {
            "train": ds_train_validtest["train"],
            "valid": ds_validtest["train"],
            "test": ds_validtest["test"],
        }
    )

    ds_dict.push_to_hub(HUGGING_FACE_PROCESSED_HIER_DATASET, private=True)

    mappings = {
        "id2species": id2species,
        "species2id": species2id,
        "id2genus": id2genus,
        "genus2id": genus2id,
    }

    upload_mappings(mappings, "mappings_per_level.json")

    upload_numpy_array(freq, "freq.npy")
    upload_numpy_array(species_to_genus_mask, "species_to_genus.npy")
    upload_numpy_array(genus_to_species_mask, "genus_to_species.npy")


def parse_args():
    parser = argparse.ArgumentParser(
        prog="GBIF hier dataset",
        description="Scripts process gbif dataset into a hierarchical dataset",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args)
