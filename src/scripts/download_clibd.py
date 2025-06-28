import os

import pandas as pd
from datasets import Dataset, DatasetDict, Features, Image, Value

RANKS = ["class", "order", "family", "subfamily", "genus", "species"]

DATASET_KWARGS = dict(
    root="~/Datasets/bioscan/",
    modality="image",
    partitioning_version="clibd",
    target_type=RANKS,
    target_format="index",
    download=True,
)

def bioscan_clibd_dataset(metadata_path, image_root, split_file):
    metadata = pd.read_csv(metadata_path, sep="\t")

    with open(split_file, "r") as f:
        sample_keys = list(set(line.strip() for line in f))

    df = metadata[metadata["sampleid"].isin(sample_keys)].copy()
    # df["image"] = df["image_file"].apply(lambda f: os.path.join(image_root, f))
    df["image"] = df.apply(
        lambda row: os.path.join(image_root, f"part{row['chunk_number']}", row["image_file"]),
        axis=1
    )

    columns_to_keep = ["image"] + RANKS
    df = df[columns_to_keep]
    df = df.reset_index(drop=True)

    features = Features({
        "image": Image(),
        "class": Value("string"),
        "order": Value("string"),
        "family": Value("string"),
        "subfamily": Value("string"),
        "genus": Value("string"),
        "species": Value("string"),
    })
    return Dataset.from_pandas(df).cast(features)


# train_ds = bioscan_clibd_dataset(
#     metadata_path="/home/jex/Datasets/bioscan/bioscan1m/BIOSCAN_Insect_Dataset_metadata.tsv",
#     image_root="/home/jex/Datasets/bioscan/bioscan1m/images/cropped_256",
#     split_file="/home/jex/Datasets/bioscan/bioscan1m/CLIBD_partitioning/test_unseen.txt"
# )

val_ds = bioscan_clibd_dataset(
    metadata_path="/home/jex/Datasets/bioscan/bioscan1m/BIOSCAN_Insect_Dataset_metadata.tsv",
    image_root="/home/jex/Datasets/bioscan/bioscan1m/images/cropped_256",
    split_file="/home/jex/Datasets/bioscan/bioscan1m/CLIBD_partitioning/val_unseen.txt"
)

test_ds = bioscan_clibd_dataset(
    metadata_path="/home/jex/Datasets/bioscan/bioscan1m/BIOSCAN_Insect_Dataset_metadata.tsv",
    image_root="/home/jex/Datasets/bioscan/bioscan1m/images/cropped_256",
    split_file="/home/jex/Datasets/bioscan/bioscan1m/CLIBD_partitioning/test_unseen.txt"
)

bioscan_dataset = DatasetDict({
    # "train": train_ds,
    "validation": val_ds,
    "test": test_ds
})

bioscan_dataset.push_to_hub(
    "jkbkaiser/clibd-raw-unseen",
    private=True
)
