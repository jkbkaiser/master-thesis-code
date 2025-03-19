import torch
import torch.nn as nn
import torch.nn.functional as F

from src.constants import DEVICE


class HAC(nn.Module):
    def __init__(self, backbone, out_features, architecture, split, **_):
        super().__init__()
        out_shape = architecture[0]
        self.split = split

        if backbone is not None:
            self.model = backbone
            self.model.head = nn.Linear(out_features, out_shape)
        else:
            self.model = nn.Linear(out_features, out_shape)

        genus_mask = torch.zeros((16, out_shape), dtype=torch.bool, device=DEVICE)
        genus_mask[:, : self.split] = 1
        self.genus_mask = genus_mask.float()
        self.species_mask = (~genus_mask).float()

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def mask_logits(self, logits):
        masked_genus_logits = logits * self.genus_mask
        masked_species_logits = logits * self.species_mask
        return masked_genus_logits, masked_species_logits

    def pred_fn(self, logits):
        genus_logits, species_logits = self.mask_logits(logits)
        return genus_logits.argmax(dim=1), species_logits.argmax(dim=1)

    def loss_fn(self, logits, genus_labels, species_labels):
        genus_logits, species_logits = self.mask_logits(logits)
        genus_loss = self.criterion(genus_logits, genus_labels)
        species_loss = self.criterion(species_logits, species_labels)
        return 0.5 * genus_loss + 0.5 * species_loss

