import io
import uuid

import numpy as np
import requests
import torch

from src.constants import DEVICE, GOOGLE_BUCKET_URL_LARGE


def get_freq_mask():
    response = requests.get(f"{GOOGLE_BUCKET_URL_LARGE}/freq.npy?id={uuid.uuid4()}")
    return torch.tensor(np.load(io.BytesIO(response.content))).to(DEVICE)
