import torch
import torch.nn as nn
import torch.nn.functional as F

from src.constants import DEVICE


class HAC(nn.Module):
    def __init__(self, backbone, out_features, architecture, split, **_):
        super().__init__()
        self.model = backbone
        out_shape = architecture[0]
        self.model.head = nn.Linear(out_features, out_shape)
        # Weight these to make sure that the 2 positive labels have the same
        # weight as all negative cases combined
        self.pos_weight = torch.tensor([(out_shape - 2) / 2], device=DEVICE)
        self.split = split

        genus_mask = torch.zeros((16, out_shape), dtype=torch.bool, device=DEVICE)
        genus_mask[:, : self.split] = 1
        self.genus_mask = genus_mask.float()
        self.species_mask = (~genus_mask).float()

    def forward(self, x):
        return self.model(x)

    def mask_logits(self, logits):
        masked_genus_logits = logits * self.genus_mask
        masked_species_logits = logits * self.species_mask
        return masked_genus_logits, masked_species_logits

    def pred_fn(self, logits):
        genus_logits, species_logits = self.mask_logits(logits)
        genus_preds = F.softmax(genus_logits, dim=1).argmax(dim=1)
        species_preds = F.softmax(species_logits, dim=1).argmax(dim=1)
        return genus_preds, species_preds

    def loss_fn(self, logits, genus_labels, species_labels):
        multi_label = torch.zeros(logits.shape, device=DEVICE)
        multi_label.scatter_(1, genus_labels.unsqueeze(1), 1)
        multi_label.scatter_(1, species_labels.unsqueeze(1), 1)

        loss = F.binary_cross_entropy_with_logits(
            logits,
            multi_label,
            pos_weight=self.pos_weight,
        )
        return loss
