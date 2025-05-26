import torch
import torch.nn as nn
import torch.nn.functional as F

from src.constants import DEVICE


class ClassifierModule(nn.Module):
    def __init__(self, out_features, architecture):
        super().__init__()
        self.classifiers = nn.ModuleList(
            [nn.Linear(out_features, nc) for nc in architecture]
        )

    def forward(self, features):
        return [classifier(features) for classifier in self.classifiers]


class MPLC(nn.Module):
    def __init__(self, backbone, out_features, architecture, ds, warmup, **_):
        super().__init__()

        self.warmup = warmup

        if backbone is not None:
            self.model = backbone
            self.model.head = ClassifierModule(out_features, architecture)
        else:
            self.model = ClassifierModule(out_features, architecture)

        self.criterion = nn.CrossEntropyLoss()
        self.masks = torch.tensor(ds.hierarchy[0], device=DEVICE)

    def forward(self, x):
        return self.model(x)

    def pred_fn(self, logits, epoch):
        [genus_logits, species_logits] = logits

        if epoch < self.warmup:
            return genus_logits.argmax(dim=1), species_logits.argmax(dim=1)

        genus_preds = F.softmax(genus_logits, dim=1).argmax(dim=1)
        species_preds = F.softmax(
            species_logits * self.masks[genus_preds], dim=1
        ).argmax(dim=1)
        return genus_preds, species_preds

    def loss_fn(self, logits, genus_labels, species_labels, epoch):
        [genus_logits, species_logits] = logits
        genus_preds = F.softmax(genus_logits, dim=1).argmax(dim=1)
        genus_loss = self.criterion(genus_logits, genus_labels)

        if epoch < self.warmup:
            species_loss = self.criterion(species_logits, species_labels)
        else:
            species_loss = self.criterion(
                species_logits * self.masks[genus_preds], species_labels
            )

        return genus_loss * 0.5 + species_loss * 0.5
