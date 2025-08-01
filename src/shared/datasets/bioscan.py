import json
import uuid
from typing import Union, cast

import datasets
import numpy as np
import requests
import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from src.constants import CACHE_DIR, GOOGLE_BUCKET_URL, NUM_PROC
from src.shared.datasets import DatasetSplit, DatasetVersion


class BioscanCustomDataset(data.Dataset):
    def __init__(self, data, transform, use_torch=False):
        self.data = data
        self.transform = transform
        self.use_torch = use_torch

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        elem = self.data[index]

        img = elem["image"]

        if self.transform:
            img = self.transform(img, self.use_torch)

        class_label = (
            elem["class"] if self.use_torch else np.array(elem["class"], dtype=np.int32)
        )

        order_label = (
            elem["order"] if self.use_torch else np.array(elem["order"], dtype=np.int32)
        )

        family_label = (
            elem["family"] if self.use_torch else np.array(elem["family"], dtype=np.int32)
        )

        subfamily_label = (
            elem["subfamily"] if self.use_torch else np.array(elem["subfamily"], dtype=np.int32)
        )

        genus_label = (
            elem["genus"] if self.use_torch else np.array(elem["genus"], dtype=np.int32)
        )
        species_label = (
            elem["species"]
            if self.use_torch
            else np.array(elem["species"], dtype=np.int32)
        )

        return (img, class_label, order_label, family_label, subfamily_label, genus_label, species_label)


class BioscanDataset():
    version: DatasetVersion

    split: int
    hierarchy: list[Union[np.ndarray, torch.Tensor]]
    frequencies: list[Union[np.ndarray, torch.Tensor]]

    metadata: dict
    labelcount_per_level: list[int]
    id2label_per_level: list[dict[int, str]]
    batch_size: int
    use_torch: bool

    train_dataloader: data.DataLoader
    valid_dataloader: data.DataLoader

    def __init__(self, version: DatasetVersion):
        self.version = version

    def load(self, batch_size: int, use_torch: bool = False, reload: bool = False):
        self.batch_size = batch_size
        self.use_torch = use_torch

        print("Setting up dataloaders")

        self.train_dataloader = self._get_dataloader(DatasetSplit.TRAIN)
        self.valid_dataloader = self._get_dataloader(DatasetSplit.VALID)
        self.test_dataloader = self._get_dataloader(DatasetSplit.TEST)

        print("Loading metadata")

        self.metadata = self._get_metadata(reload)
        self.labelcount_per_level = [int(v["count"]) for v in self.metadata["per_level"]]
        self.id2label_per_level = [
            {int(k): v for k, v in level["id2label"].items()}
            for level in self.metadata["per_level"]
        ]

        self.frequencies = self._get_frequencies()
        self.hierarchy = self._get_hierarchy(reload)

        self.train_transform = transforms.Compose([
            transforms.RandomRotation(25),
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.Lambda(lambda x: x / 255.0),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.3, scale=(0.02, 0.2), ratio=(0.3, 3.3), value=0),
        ])

        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.Lambda(lambda x: x / 255.0),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def _get_dataloader(self, split: DatasetSplit):
        path = f"jkbkaiser/{self.version.value}"

        dataset = cast(
            datasets.Dataset,
            datasets.load_dataset(path, split=split.value, num_proc=NUM_PROC).with_format(
                "torch"
            ),
        )

        transform = self._image_to_tensor if split != DatasetSplit.TRAIN else self._image_to_tensor_train
        dataset = BioscanCustomDataset(dataset, transform=transform, use_torch=self.use_torch)

        dataloader = data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=NUM_PROC,
            shuffle=split == DatasetSplit.TRAIN.value,
            drop_last=True,
            collate_fn=lambda batch: self._collate(batch, self.use_torch),
        )

        print(f"\tloaded {split.value} split from {path} dataset")
        return dataloader

    def _image_to_tensor_train(self, img, use_torch=False):
        if self.version in [DatasetVersion.GBIF_GENUS_SPECIES_10K_EMBEDDINGS]:
            return img
        transformed_img = self.train_transform(img.float())
        return transformed_img if use_torch else transformed_img.numpy()

    def _image_to_tensor(self, img, use_torch=False):
        if self.version in [DatasetVersion.GBIF_GENUS_SPECIES_10K_EMBEDDINGS]:
            return img
        transformed_img = self.transform(img.float())
        return transformed_img if use_torch else transformed_img.numpy()

    def _collate(self, batch, use_torch=False):
        if isinstance(batch[0], np.ndarray) or isinstance(batch[0], torch.Tensor):
            if use_torch:
                return torch.stack(batch)
            return np.stack(batch)
        elif isinstance(batch[0], (tuple, list)):
            transposed = zip(*batch)
            return [self._collate(samples, use_torch) for samples in transposed]
        else:
            if use_torch:
                return torch.tensor(batch)
            return np.array(batch)

    def _get_metadata(self, reload: bool):
        directory = CACHE_DIR / self.version.value
        path = directory / "metadata.json"

        if reload or not path.is_file():
            url = f"{GOOGLE_BUCKET_URL}/{self.version.value}/metadata.json?id={uuid.uuid4()}"

            response = requests.get(url)
            if response.status_code != 200:
                raise Exception(f"Could not retrieve metadata for {self.version.value}, status: {response.status_code}")
            metadata = response.json()

            if not directory.exists():
                directory.mkdir(parents=True)

            with open(path, "w") as f:
                json.dump(metadata, f)

            return metadata

        with open(path, "r") as f:
            return json.load(f)

    def _get_frequencies(self) -> list[Union[np.ndarray, torch.Tensor]]:
        freqs = []

        for i, level in enumerate(self.metadata["per_level"]):
            freq = np.zeros(self.metadata["per_level"][i]["count"])

            for k, v in level["frequencies"].items():
                freq[int(k)] = v

            if self.use_torch:
                freq = torch.tensor(freq)

            freqs.append(freq)

        return freqs

    def _get_hierarchy(self, reload: bool = False):
        path = CACHE_DIR / f"{self.version.value}/hierarchy.npz"

        if reload or not path.is_file():
            url = f"{GOOGLE_BUCKET_URL}/{self.version.value}/hierarchy.npz?id={uuid.uuid4()}"
            response = requests.get(url)

            if response.status_code != 200:
                raise Exception(f"Could not retrieve metadata for {self.version.value}")

            with open(path, "wb") as f:
                f.write(response.content)

        hierarchy_npz = np.load(path)
        hierarchy = [hierarchy_npz[f"level_{i}"] for i in range(5)]
        return hierarchy 
