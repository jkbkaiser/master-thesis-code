import os
from pathlib import Path

import torch
from dotenv import load_dotenv

load_dotenv()

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
NUM_PROC = 9

GOOGLE_PROJECT = os.getenv("GOOGLE_PROJECT")
GOOGLE_BUCKET = os.getenv("GOOGLE_BUCKET")
GOOGLE_BUCKET_URL = os.getenv("GOOGLE_BUCKET_URL")

CACHE_DIR = Path(".cache")
