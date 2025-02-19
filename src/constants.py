import os

import torch
from dotenv import load_dotenv

load_dotenv()

GOOGLE_BUCKET_URL = os.getenv("GOOGLE_BUCKET_URL")
GOOGLE_BUCKET_URL_LARGE = os.getenv("GOOGLE_BUCKET_URL_LARGE")
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
