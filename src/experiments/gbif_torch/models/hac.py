import torch
import torch.nn as nn
import torch.nn.functional as F

from src.constants import DEVICE


class HAC(nn.Module):
    def __init__(self, backbone, out_features, num_classes, **_):
        super().__init__()
        self.model = backbone
        self.model.head = nn.Linear(out_features, num_classes)

        # Make this automatic
        self.split = 180
        # Weight these to make sure that the 2 positive labels have the same
        # weight as all negative cases combined
        self.pos_weight = torch.tensor([(num_classes - 2) / 2], device=DEVICE)

    def forward(self, x):
        return self.model(x)

    def mask_logits(self, logits):
        genus_mask = torch.zeros_like(logits, dtype=torch.bool)
        genus_mask[:, : self.split] = 1

        masked_genus_logits = logits * genus_mask.float()
        masked_species_logits = logits * (~genus_mask).float()

        return masked_genus_logits, masked_species_logits

    def pred_fn(self, logits):
        genus_logits, species_logits = self.mask_logits(logits)
        genus_preds = F.softmax(genus_logits, dim=1).argmax(dim=1)
        species_preds = F.softmax(species_logits, dim=1).argmax(dim=1)
        return genus_preds, species_preds

    def loss_fn(self, logits, genus_labels, species_labels):
        multi_label = torch.zeros(logits.shape, device=genus_labels.device)
        multi_label.scatter_(1, genus_labels.unsqueeze(1), 1)
        multi_label.scatter_(1, species_labels.unsqueeze(1), 1)
        loss = F.binary_cross_entropy_with_logits(
            logits,
            multi_label,
            pos_weight=self.pos_weight,
        )
        return loss
