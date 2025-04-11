import argparse
import csv
import shutil
import time
import uuid
from multiprocessing.pool import ThreadPool
from pathlib import Path

import datasets
import pandas as pd
import requests
from datasets import Features, Image, Value, load_dataset
from PIL import Image as PILImage
from PIL import UnidentifiedImageError
from tqdm import tqdm

DATA_DIR = Path("./data")
GBIF_DATA_DIR = DATA_DIR / "gbif"
HF_DATA_DIR = DATA_DIR / "hf"
HF_IMG_DIR = HF_DATA_DIR / "images"

HUGGING_FACE_DATASET = "jkbkaiser/gbif_raw_10k"

DF_FIELDS = [
    # "kingdomKey",
    # "phylumKey",
    # "orderKey",
    # "familyKey",
    # "genusKey",
    # "scientificName",
    "species",
    # "sex",
    # "lifeStage",
    # "continent",
]


FIELDS = [
    "id",
    "kingdom_key",
    "phylum_key",
    "order_key",
    "family_key",
    "genus_key",
    "scientific_name",
    "species",
    "sex",
    "life_stage",
    "continent",
]


def retry(func, attempts):
    sleep = 5
    result = None

    for _ in range(attempts):
        try:
            result = func()
            break
        except Exception:
            time.sleep(sleep)
        sleep *= 2

    if result is None:
        raise Exception("Failed after three attempts")

    return result


def download_img(ident, img_url, directory_path: Path):
    img_filename = f"{ident}.jpeg"
    response = retry(lambda: requests.get(img_url), 3)

    if response.status_code != 200:
        raise Exception(f"Failed request with status {response.status_code}")

    fmt = response.headers["Content-Type"]
    if fmt not in ["image/jpeg", "image/jpg", "jpeg"]:
        raise Exception(f"Unsupported img format {fmt}")

    with open(directory_path / img_filename, "wb") as f:
        f.write(response.content)

    return img_filename


def download_gbif_entry(entry):
    identifier = uuid.uuid4()
    img_filename = download_img(identifier, entry["identifier"], HF_IMG_DIR)

    entry = {
        "file_name": f"./images/{img_filename}",
        "id": identifier,
        "kingdom_key": entry["kingdomKey"],
        "phylum_key": entry["phylumKey"],
        "order_key": entry["orderKey"],
        "family_key": entry["familyKey"],
        "genus_key": entry["genusKey"],
        "scientific_name": entry["scientificName"],
        "species": entry["species"],
        "sex": entry["sex"],
        "life_stage": entry["lifeStage"],
        "continent": entry["continent"],
    }

    return entry


def download_gbif_entries_parallel(dataframe, num):
    gbif_entry_generator = (entry for _, entry in dataframe.head(num).iterrows())

    def safe_download_gbif_entry(entry):
        try:
            return download_gbif_entry(entry)
        except Exception as e:
            print(f"Error processing {entry}: {e}")
            return None

    with ThreadPool() as pool:
        with tqdm(total=num) as pbar:
            results = []

            for result in pool.imap_unordered(
                safe_download_gbif_entry, gbif_entry_generator
            ):
                if result is not None:
                    results.append(result)
                pbar.update(1)
    return results


def store_csv(data):
    header = [
        "file_name",
        "id",
        "kingdom_key",
        "phylum_key",
        "order_key",
        "family_key",
        "genus_key",
        "scientific_name",
        "species",
        "sex",
        "life_stage",
        "continent",
    ]
    csv_filename = HF_DATA_DIR / "metadata.csv"
    with open(csv_filename, mode="w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=header)
        writer.writeheader()

        for entry in data:
            if entry is not None and entry:
                writer.writerow(entry)


def filter_existing_fields(ds):
    def validate_row(row):
        if row["image"] is None:
            return False
        try:
            with PILImage.open(row["image"]["path"]) as img:
                img.verify()
            return True
        except (UnidentifiedImageError, IOError):
            return False

    ds = ds.cast_column("image", Image(decode=False))
    ds = ds.filter(validate_row)
    return ds


def preprocess_entry(entry):
    img = entry["image"]
    img = img.resize((256, 256))

    def crop_center(img):
        new_height, new_width = 224, 224
        width, height = img.size

        left = (width - new_width) / 2
        top = (height - new_height) / 2
        right = (width + new_width) / 2
        bottom = (height + new_height) / 2

        img = img.crop((left, top, right, bottom))
        return img

    img = crop_center(img)
    entry["image"] = img
    return entry


def preprocess_dataset(ds):
    features = Features(
        {
            "image": Image(mode=None, decode=True, id=None),
            "id": Value(dtype="string", id=None),
            "kingdom_key": Value(dtype="uint32", id=None),
            "phylum_key": Value(dtype="uint32", id=None),
            "order_key": Value(dtype="uint32", id=None),
            "family_key": Value(dtype="uint32", id=None),
            "genus_key": Value(dtype="uint32", id=None),
            "scientific_name": Value(dtype="string", id=None),
            "species": Value(dtype="string", id=None),
            "sex": Value(dtype="string", id=None),
            "life_stage": Value(dtype="string", id=None),
            "continent": Value(dtype="string", id=None),
        }
    )

    ds = ds.cast(features)
    # ds = ds.map(preprocess_entry)

    def valid_image_shape(row):
        return row["image"].mode == "RGB"

    ds = ds.filter(valid_image_shape)
    ds_dict = datasets.DatasetDict({"data": ds})
    return ds_dict


def run(args):
    if HF_IMG_DIR.exists():
        while True:
            user_input = input(
                f"Directory '{HF_IMG_DIR}' already exists. Remove existing data? [Y/n]: "
            )
            if user_input.lower() == "y":
                shutil.rmtree(HF_IMG_DIR)
                print(f"Directory '{HF_IMG_DIR}' and its contents removed.")
                break
            elif user_input.lower() == "n":
                print("Data removal cancelled. Reusing data")
                break
            else:
                print("Invalid input. Please enter 'Y' or 'n'.")
                break
    HF_IMG_DIR.mkdir(parents=True, exist_ok=True)

    multimedia = pd.read_csv(GBIF_DATA_DIR / "multimedia.txt", sep="\t")
    occurrence = pd.read_csv(
        GBIF_DATA_DIR / "occurrence.txt", sep="\t", low_memory=False
    )

    multimedia = multimedia[
        (multimedia["format"] == "image/jpeg")
        | (multimedia["format"] == "jpeg")
        | (multimedia["format"] == "image/png")
    ]

    df = multimedia.merge(occurrence, on="gbifID", how="inner")

    df = df.dropna(subset=DF_FIELDS)

    print("Downloading images...")

    store_csv(download_gbif_entries_parallel(df, args.num_entries))

    print("Finished downloads, loading dataset")

    ds = load_dataset(str(HF_DATA_DIR))["train"]

    print("Filtering images")

    ds = filter_existing_fields(ds)

    print("Preprocessing data")

    ds_dict = preprocess_dataset(ds)

    print("Final data dictionary")

    print(ds_dict)

    print("Pushing to hub")

    # ds_dict.push_to_hub(HUGGING_FACE_DATASET, private=True)


def parse_args():
    parser = argparse.ArgumentParser(
        prog="GBIF extractor",
        description="Scripts for downloading gbif mediadata and creating hugging face datasets",
    )
    parser.add_argument("-ne", "--num-entries", default=1000, required=False, type=int)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args)
