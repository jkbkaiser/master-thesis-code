from enum import Enum
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.constants import DEVICE


class PrototypeVersion(str, Enum):
    GENUS_SPECIES_POINCARE  = "genus_species_poincare"
    SPECIES_HYPERSPHERE_UNIFORM = "species_hypersphere_uniform"


def get_prototypes(prototype_version, dataset_version, dimensionality):
    prototype_path = Path("./prototypes") / dataset_version / prototype_version / f"{dimensionality}.npy"
    print("Ret", str(prototype_path))
    prototypes = torch.from_numpy(np.load(prototype_path)).float().to(DEVICE)
    # prototypes = F.normalize(prototypes, p=2, dim=1).to(DEVICE)
    return prototypes
