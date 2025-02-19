import torch.nn as nn
import torch.nn.functional as F

from src.shared.datasets.gbif.taxonomy import get_species_to_genus_map


class MARG(nn.Module):
    def __init__(self, backbone, out_features, num_classes, **_):
        super().__init__()
        self.model = backbone
        self.model.head = nn.Linear(out_features, num_classes)
        self.criterion = nn.CrossEntropyLoss()
        self.species_to_genus_map = get_species_to_genus_map()

    def forward(self, x):
        return self.model(x)

    def pred_fn(self, logits):
        species_probs = F.softmax(logits, dim=1)
        species_preds = species_probs.argmax(dim=1)
        genus_preds = self.species_to_genus_map[species_preds]
        return genus_preds, species_preds

    def loss_fn(self, logits, genus_labels, species_labels):
        return self.criterion(logits, species_labels)
