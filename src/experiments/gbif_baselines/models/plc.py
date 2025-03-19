import torch.nn as nn
import torch.nn.functional as F


class ClassifierModule(nn.Module):
    def __init__(self, out_features, architecture):
        super().__init__()
        self.classifiers = nn.ModuleList(
            [nn.Linear(out_features, nc) for nc in architecture]
        )

    def forward(self, features):
        return [classifier(features) for classifier in self.classifiers]


class PLC(nn.Module):
    def __init__(self, backbone, out_features, architecture, **_):
        super().__init__()

        if backbone is not None:
            self.model = backbone
            self.model.head = ClassifierModule(out_features, architecture)
        else:
            self.model = ClassifierModule(out_features, architecture)

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def pred_fn(self, logits):
        [genus_logits, species_logits] = logits
        return genus_logits.argmax(dim=1), species_logits.argmax(dim=1)

    def loss_fn(self, logits, genus_labels, species_labels):
        [genus_logits, species_logits] = logits
        genus_loss = self.criterion(genus_logits, genus_labels)
        species_loss = self.criterion(species_logits, species_labels)
        return 0.5 * genus_loss + 0.5 * species_loss
