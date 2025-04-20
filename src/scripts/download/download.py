import argparse
import asyncio
import csv
import io
import json
import logging
import random
import shutil
import uuid
from pathlib import Path

import aiofiles
import aiohttp
import datasets
import pandas as pd
from datasets import Features, Image, Value, load_dataset
from PIL import Image as PILImage
from PIL import UnidentifiedImageError
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio

from src.constants import NUM_PROC

logging.basicConfig(level=logging.INFO)


DATA_DIR = Path("./data")
GBIF_DATA_DIR = DATA_DIR / "gbif"
HF_DATA_DIR = DATA_DIR / "hf"
HF_IMG_DIR = HF_DATA_DIR / "images"
DF_CSV = DATA_DIR / "df.csv"

HUGGING_FACE_DATASET = "jkbkaiser/gbif_coleoptera_eu_large"

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


async def retry_async(func, attempts):
    sleep_time = 10
    last_exception = None
    result = None
    c = 0

    for i in range(attempts):
        try:
            result = await func()
        except Exception as e:
            last_exception = e
            if i < attempts - 1:
                await asyncio.sleep(sleep_time + random.uniform(0, 3))
                sleep_time *= 1.5

        c += 1

    if result is None:
        logging.warning(f"[Retry Failed] After {attempts} attempts: {str(last_exception)}")

    return result, c


async def download_img(
    ident,
    img_url,
    directory_path: Path,
):
    img_filename = f"{ident}.jpeg"
    headers = {"User-Agent": "Mozilla/5.0"}

    async with aiohttp.ClientSession(headers=headers) as session:
        async def fetch():
            async with session.get(img_url) as response:
                if response.status != 200:
                    raise Exception(f"Failed request with status {response.status}")
                content_type = response.headers.get("Content-Type", "")
                if content_type not in ["image/jpeg", "image/jpg", "jpeg"]:
                    raise Exception(f"Unsupported img format {content_type}")

                content = await response.read()

                if content is None:
                    raise Exception(f"No content returned for {img_url}")

                return content

        content, attempts = await retry_async(fetch, 4)

        if content is None:
            return None

    try:
        image = PILImage.open(io.BytesIO(content))
        image = image.convert("RGB")
        image = image.resize((256, 256), PILImage.Resampling.LANCZOS)
        output = io.BytesIO()
        image.save(output, format="JPEG", quality=95)
        resized_bytes = output.getvalue()
    except Exception as e:
        print(f"Image processing failed for {ident}: {e}")
        return None

    # Save the processed image to disk
    async with aiofiles.open(directory_path / img_filename, "wb") as f:
        await f.write(resized_bytes)

    print(f"Downloaded: {img_filename} after {attempts} attempts from {img_url}")

    return img_filename


async def download_row(entry, sem):
    async with sem:
        identifier = uuid.uuid4()
        img_filename = await download_img(
            identifier,
            entry.identifier,
            HF_IMG_DIR,
        )

        if img_filename is None:
            return None

        return {
            "file_name": f"./images/{img_filename}",
            "id": identifier,
            "kingdom_key": entry.kingdomKey,
            "phylum_key": entry.phylumKey,
            "order_key": entry.orderKey,
            "family_key": entry.familyKey,
            "genus_key": entry.genusKey,
            "scientific_name": entry.scientificName,
            "species": entry.species,
            "sex": entry.sex,
            "life_stage": entry.lifeStage,
            "continent": entry.continent,
        }


def store_csv(data, mode="w"):
    csv_filename = HF_DATA_DIR / "metadata.csv"
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

    with open(csv_filename, mode=mode, newline="") as file:
        writer = csv.DictWriter(file, fieldnames=header)
        if mode == "w":
            writer.writeheader()
        for entry in data:
            if entry:
                writer.writerow(entry)


async def start_downloader(rows, bar_desc):
    sem = asyncio.Semaphore(50)

    tasks = [
        download_row(entry, sem)
        for entry in rows.itertuples(index=False)
    ]

    results = []
    for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc=bar_desc, leave=False):
        # try:
        result = await coro
        if isinstance(result, Exception):
            print(f"RE, {result}")
        if result is not None:
            results.append(result)
        # except Exception as e:
        #     # optionally log or track errors
        #     pass

    return [r for r in results if not isinstance(r, Exception) and r is not None]


async def download(df, batch_size, checkpoint_file=(DATA_DIR/"checkpoint.json")):
    total_batches = (len(df) + batch_size - 1) // batch_size

    try:
        with open(checkpoint_file, "r") as f:
            checkpoint = json.load(f)
            last_batch_idx = checkpoint.get("batch_idx", 0)
            print(f"Resuming from batch {last_batch_idx + 1}/{total_batches}")
    except (FileNotFoundError, json.JSONDecodeError):
        last_batch_idx = 0
        print(f"Starting from the beginning.")

    for batch_idx in range(last_batch_idx, total_batches):
        start = batch_idx * batch_size
        end = start + batch_size
        batch = df.iloc[start:end]

        remaining = total_batches - batch_idx - 1
        bar_desc = f"Batch {batch_idx + 1}/{total_batches} ({remaining} left)"

        batch_results = await start_downloader(batch, bar_desc)

        if batch_results:
            store_csv(batch_results, mode="a")
            print(f"✔ Batch {batch_idx + 1}/{total_batches} done — {len(batch_results)} records saved.")

        with open(checkpoint_file, "w") as f:
            json.dump({"batch_idx": batch_idx + 1}, f)

    print("✅ All batches processed.")


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

    metadata_csv = HF_DATA_DIR / "metadata.csv"

    if metadata_csv.exists():
        while True:
            user_input = input(
                f"Metadata.csv already exists. Remove existing data? [Y/n]: "
            )
            if user_input.lower() == "y":
                metadata_csv.unlink()
                store_csv([], mode="w")
                print(f"Directory '{HF_IMG_DIR}' and its contents removed.")
                break
            elif user_input.lower() == "n":
                print("Data removal cancelled. Reusing data")
                break
            else:
                print("Invalid input. Please enter 'Y' or 'n'.")
                break


    print(f"Reading data to process from {DF_CSV}...")

    df = pd.read_csv(DF_CSV)

    print(len(df))

    print("Downloading images...")

    # asyncio.run(download_gbif_entries_parallel_async(df, batch_size=5000))

    asyncio.run(download(df, batch_size=5000))

    print("Finished downloads, loading dataset")


def parse_args():
    parser = argparse.ArgumentParser(
        prog="GBIF extractor",
        description="Scripts for downloading gbif mediadata and creating hugging face datasets",
    )
    parser.add_argument("-ne", "--num-entries", default=100, required=False, type=int)
    parser.add_argument("--create-csv", action="store_true", default=False)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args)
