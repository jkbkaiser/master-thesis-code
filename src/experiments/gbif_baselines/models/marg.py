import torch
import torch.nn as nn
import torch.nn.functional as F

from src.constants import DEVICE


class MARG(nn.Module):
    def __init__(self, backbone, out_features, architecture, ds, **_):
        super().__init__()

        if backbone is not None:
            self.model = backbone
            self.model.head = nn.Linear(out_features, architecture)
        else:
            self.model = nn.Linear(out_features, architecture)

        self.criterion = nn.CrossEntropyLoss()
        self.species2genus = torch.tensor(ds.hierarchy[0].T, device=DEVICE).argmax(dim=1)

    def forward(self, x):
        return self.model(x)

    def pred_fn(self, logits):
        species_probs = F.softmax(logits, dim=1)
        species_preds = species_probs.argmax(dim=1)
        genus_preds = self.species2genus[species_preds]
        return genus_preds, species_preds

    def loss_fn(self, logits, _, species_labels):
        return self.criterion(logits, species_labels)
