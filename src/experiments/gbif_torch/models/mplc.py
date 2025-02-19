import torch.nn as nn
import torch.nn.functional as F

from src.shared.datasets.gbif.taxonomy import get_genus_to_species_mask


class ClassifierModule(nn.Module):
    def __init__(self, out_features, num_classes):
        super().__init__()
        self.classifiers = nn.ModuleList(
            [nn.Linear(out_features, nc) for nc in num_classes]
        )

    def forward(self, features):
        return [classifier(features) for classifier in self.classifiers]


class MPLC(nn.Module):
    def __init__(self, backbone, out_features, num_classes, **_):
        super().__init__()
        self.model = backbone
        self.model.head = ClassifierModule(out_features, num_classes)
        self.criterion = nn.CrossEntropyLoss()
        self.masks = get_genus_to_species_mask()

    def forward(self, x):
        return self.model(x)

    def pred_fn(self, logits):
        [genus_logits, species_logits] = logits
        genus_preds = F.softmax(genus_logits, dim=1).argmax(dim=1)
        species_preds = F.softmax(
            species_logits * self.masks[genus_preds], dim=1
        ).argmax(dim=1)
        return genus_preds, species_preds

    def loss_fn(self, logits, genus_labels, species_labels):
        [genus_logits, species_logits] = logits
        genus_preds = F.softmax(genus_logits, dim=1).argmax(dim=1)
        genus_loss = self.criterion(genus_logits, genus_labels)
        species_loss = self.criterion(
            species_logits * self.masks[genus_preds], species_labels
        )
        return genus_loss * 0.5 + species_loss * 0.5
