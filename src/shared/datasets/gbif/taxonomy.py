import io
import uuid

import numpy as np
import requests
import torch

from src.constants import DEVICE, GOOGLE_BUCKET_URL_LARGE


def get_genus_to_species_mask():
    response = requests.get(
        f"{GOOGLE_BUCKET_URL_LARGE}/genus_to_species.npy?id={uuid.uuid4()}"
    )
    return torch.tensor(np.load(io.BytesIO(response.content))).to(DEVICE)


def get_species_to_genus_map():
    response = requests.get(
        f"{GOOGLE_BUCKET_URL_LARGE}/species_to_genus.npy?id={uuid.uuid4()}"
    )
    return torch.tensor(np.load(io.BytesIO(response.content))).to(DEVICE)
