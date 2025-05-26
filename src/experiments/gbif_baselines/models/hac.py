import torch
import torch.nn as nn

from src.constants import DEVICE


class HAC(nn.Module):
    def __init__(self, backbone, out_features, architecture, **_):
        super().__init__()
        [
            self.num_genus,
            self.num_species,
        ] = architecture
        out_shape = self.num_genus + self.num_species

        if backbone is not None:
            self.model = backbone
            self.model.head = nn.Linear(out_features, out_shape)
        else:
            self.model = nn.Linear(out_features, out_shape)

        genus_mask = torch.zeros((1, out_shape), dtype=torch.bool, device=DEVICE)
        genus_mask[:, :self.num_genus] = 1
        self.genus_mask = genus_mask.float()
        self.species_mask = (~genus_mask).float()

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def mask_logits(self, logits):
        masked_genus_logits = logits * self.genus_mask
        masked_species_logits = logits * self.species_mask
        return masked_genus_logits, masked_species_logits

    def pred_fn(self, logits, *args, **kwargs):
        genus_logits, species_logits = self.mask_logits(logits)
        return genus_logits.argmax(dim=1), species_logits.argmax(dim=1) - self.num_genus

    def loss_fn(self, logits, genus_labels, species_labels, *args, **kwargs):
        genus_logits, species_logits = self.mask_logits(logits)
        genus_loss = self.criterion(genus_logits, genus_labels)
        species_loss = self.criterion(species_logits, species_labels + self.num_genus)
        return 0.5 * genus_loss + 0.5 * species_loss
