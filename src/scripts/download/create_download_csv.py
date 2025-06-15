import argparse
from pathlib import Path

import pandas as pd
from PIL import Image as PILImage

from src.constants import NUM_PROC

DATA_DIR = Path("./data")
GBIF_DATA_DIR = DATA_DIR / "gbif"
DF_CSV = DATA_DIR / "df.csv"

REQUIRED_DF_FIELDS = [
    "species",
    "genus",
]

def run(args):
    if DF_CSV.exists():
        while True:
            user_input = input(
                f"{DF_CSV} already exists. Remove existing data? [Y/n]: "
            )
            if user_input.lower() == "y":
                DF_CSV.unlink()
                print(f"FILE '{DF_CSV}' and its contents removed.")
                break
            elif user_input.lower() == "n":
                print("Data removal cancelled. Reusing data")
                return
            else:
                print("Invalid input. Please enter 'Y' or 'n'.")
                break

    print(f"creating csv with {args.num_entries} rows")

    dtype_spec = {
        "title": str,
        "description": str,
        "source": str,
        "audience": str,
        "created": str,
        "creator": str,
        "contributor": str,
        "publisher": str,
        "license": str,
        "rightsHolder": str,
    }

    multimedia = pd.read_csv(GBIF_DATA_DIR / "multimedia.txt", sep="\t", dtype=dtype_spec)
    occurrence = pd.read_csv(
        GBIF_DATA_DIR / "occurrence.txt", sep="\t", low_memory=False
    )

    multimedia = multimedia[
        (multimedia["format"] == "image/jpeg")
        | (multimedia["format"] == "jpeg")
        # | (multimedia["format"] == "image/png")
    ]

    multimedia_unique = multimedia.drop_duplicates(subset="gbifID", keep="first")
    df = multimedia_unique.merge(occurrence, on="gbifID", how="inner")
    df = df.dropna(subset=REQUIRED_DF_FIELDS)

    print("Total rows after merging ", len(df))

    df = df.sample(frac=1, random_state=42).head(args.num_entries)

    print("Storing processed data to CSV...")
    df.to_csv(DF_CSV, index=False)


def parse_args():
    parser = argparse.ArgumentParser(
        prog="GBIF extractor",
    )
    parser.add_argument("-ne", "--num-entries", default=100, required=False, type=int)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args)
