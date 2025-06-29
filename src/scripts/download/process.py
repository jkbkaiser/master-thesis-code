from pathlib import Path

import datasets
from datasets import Features, Image, Value, load_dataset

DATA_DIR = Path("./data")
GBIF_DATA_DIR = DATA_DIR / "gbif"
HF_DATA_DIR = DATA_DIR / "hf"
HF_IMG_DIR = HF_DATA_DIR / "images"

HUGGING_FACE_DATASET = "jkbkaiser/gbif_coleoptera_eu_full"


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

    def valid_image_shape(row):
        return row["image"].mode == "RGB"

    ds = ds.filter(valid_image_shape)
    ds_dict = datasets.DatasetDict({"data": ds})
    return ds_dict


if __name__ == "__main__":
    ds = load_dataset(str(HF_DATA_DIR), num_proc=16)["train"]

    print("Loaded ds")
    print(ds)
    print("\n\n\n\n\n")

    ds_dict = preprocess_dataset(ds)
    ds_dict.push_to_hub(HUGGING_FACE_DATASET, private=False)
