import os
from pathlib import Path

import torch
import tqdm
from datasets import Dataset, DatasetDict

from src.shared.datasets import Dataset as MyDataset
from src.shared.datasets import DatasetVersion
from src.shared.torch.backbones import load_for_transfer_learning, t2t_vit_t_14

torch.set_float32_matmul_precision("medium")

ds = MyDataset(DatasetVersion.GBIF_FLAT_10K)
ds.load(batch_size=16, use_torch=True)

PRETRAINED_WEIGHTS_DIR = Path(os.getcwd()) / "pretrained_weights"
PRETRAINED_T2T_VITT_T14 = PRETRAINED_WEIGHTS_DIR / "81.7_T2T_ViTt_14.pth.tar"
PRETRAINED_VITAEv2 = PRETRAINED_WEIGHTS_DIR / "ViTAEv2-B.pth.tar"

backbone = t2t_vit_t_14()
load_for_transfer_learning(
    backbone,
    PRETRAINED_T2T_VITT_T14,
    use_ema=True,
    strict=False,
)

# Ensure the model is in eval mode
backbone.eval()
backbone.head = torch.nn.Identity()

# Your train_loader should yield (inputs, labels)
def extract_embeddings(loader, model):
    all_embeddings = []
    all_genus_labels = []
    all_species_labels = []

    with torch.no_grad():
        for imgs, genus_labels, species_labels in tqdm.tqdm(loader):
            embeddings = model(imgs)

            all_embeddings.extend(embeddings.cpu().numpy())
            all_genus_labels.extend(genus_labels.cpu().numpy())
            all_species_labels.extend(species_labels.cpu().numpy())

    dataset = Dataset.from_dict({
        "image": all_embeddings,
        "genus": all_genus_labels,
        "species": all_species_labels,
    })
    return dataset

train = extract_embeddings(ds.train_dataloader, backbone)
valid = extract_embeddings(ds.valid_dataloader, backbone)
test = extract_embeddings(ds.test_dataloader, backbone)

dataset_dict = DatasetDict({"train": train, "valid": valid, "test": test})

dataset_dict.push_to_hub(f"jkbkaiser/{DatasetVersion.GBIF_FLAT_10K_EMBEDDINGS.value}", private=True)
