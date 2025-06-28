import math
from enum import Enum
from pathlib import Path

import numpy as np
import torch

from src.constants import DEVICE


class PrototypeVersion(str, Enum):
    GENUS_SPECIES_POINCARE  = "genus_species_poincare"
    AVG_GENUS  = "avg_genus"
    AVG_MULTI = "avg_multi"
    DISTORTION  = "distortion"
    ENTAILMENT_CONES = "entailment_cones"
    HYPERSPHERE_UNIFORM = "hypersphere_uniform"
    SPECIES_HYPERSPHERE = "species_hypersphere"


def get_prototypes(prototype_version, dataset_version, dimensionality):
    prototype_path = Path("./prototypes") / dataset_version / prototype_version / f"{dimensionality}.npy"
    prototypes = torch.from_numpy(np.load(prototype_path)).float().to(DEVICE)

    if prototype_version in [PrototypeVersion.AVG_GENUS, PrototypeVersion.AVG_MULTI]:
        c = 1.5
        prototypes = (prototypes * 0.95) / math.sqrt(c)

    return prototypes
