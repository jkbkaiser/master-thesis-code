from typing import cast

import datasets
import numpy as np
import torch
import torch.utils.data as data

NUM_PROC = 10

DATASET_DICT = {
    "hier": "jkbkaiser/thesis-gbif-hier-large",
    "flat": "jkbkaiser/thesis-gbif-flat-large",
}


def image_to_tensor(img, use_torch=False):
    img = np.array(img, dtype=np.float32) / 255.0

    if use_torch:
        return torch.tensor(img, dtype=torch.float32)

    return img


class GBIFDataset(data.Dataset):
    def __init__(self, data, transform, use_torch=False):
        self.data = data
        self.transform = transform
        self.use_torch = use_torch

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        elem = self.data[index]

        img = self.transform(elem["image"], self.use_torch)
        genus_label = (
            elem["genus"] if self.use_torch else np.array(elem["genus"], dtype=np.int32)
        )
        species_label = (
            elem["species"]
            if self.use_torch
            else np.array(elem["species"], dtype=np.int32)
        )

        return (img, genus_label, species_label)


def collate(batch, use_torch=False):
    if isinstance(batch[0], np.ndarray) or isinstance(batch[0], torch.Tensor):
        if use_torch:
            return torch.stack(batch)
        return np.stack(batch)
    elif isinstance(batch[0], (tuple, list)):
        transposed = zip(*batch)
        return [collate(samples, use_torch) for samples in transposed]
    else:
        if use_torch:
            return torch.tensor(batch)
        return np.array(batch)


def load_gbif_dataloader(split: str, batch_size: int, version="flat", use_torch=False):
    path = DATASET_DICT[version]

    dataset = cast(
        datasets.Dataset,
        datasets.load_dataset(path, split=split, num_proc=NUM_PROC).with_format(
            "torch"
        ),
    )

    dataset = GBIFDataset(dataset, transform=image_to_tensor, use_torch=use_torch)

    dataloader = data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=NUM_PROC,
        shuffle=split == "train",
        drop_last=True,
        collate_fn=lambda batch: collate(batch, use_torch),
    )

    print(f"loaded {split} split from {path} dataset")

    return dataloader
