import argparse
import json
import os
from pathlib import Path
from typing import cast

import numpy as np
from datasets import DatasetDict, Features, Image, Value, load_dataset
from dotenv import load_dotenv
from google.cloud import storage

load_dotenv()
HUGGING_FACE_SOURCE_DATASET = "jkbkaiser/thesis-gbif-raw-large"
HUGGING_FACE_PROCESSED_FLAT_DATASET = "jkbkaiser/thesis-gbif-flat-large"
GOOGLE_PROJECT = os.getenv("GOOGLE_PROJECT")
GOOGLE_BUCKET_URL = os.getenv("GOOGLE_BUCKET_URL")
BUCKET_NAME = "thesis-gbif-mappings-large"

DATA_DIR = Path("./data")
GBIF_DATA_DIR = DATA_DIR / "gbif"
EXTRACTION_DIR = GBIF_DATA_DIR / "extraction"


def valid_species(entry):
    return len(entry["species"].split(" ")) >= 2


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


def map_to_label(row, labels2id):
    row["genus"] = labels2id[row["genus"].lower()]
    row["species"] = labels2id[row["species"].lower()]
    return row


def run(_):
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

    ds_train_validtest = ds["data"].train_test_split(test_size=0.2, seed=42)
    ds_validtest = ds_train_validtest["test"].train_test_split(test_size=0.5, seed=42)
    ds_dict = DatasetDict(
        {
            "train": ds_train_validtest["train"],
            "valid": ds_validtest["train"],
            "test": ds_validtest["test"],
        }
    )

    ds_dict.push_to_hub(HUGGING_FACE_PROCESSED_FLAT_DATASET, private=True)

    flat_mappings = {
        "id2labels": id2label,
        "labels2id": label2id,
        "split": split,
    }
    upload_mappings(flat_mappings, "flat_mapping.json")


def parse_args():
    parser = argparse.ArgumentParser(
        prog="GBIF flat dataset",
        description="Scripts process gbif dataset into a flat hierarchical dataset",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args)
