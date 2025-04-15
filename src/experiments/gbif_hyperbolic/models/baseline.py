import torch.nn as nn


class Baseline(nn.Module):
    def __init__(self, backbone, out_features, architecture, prototypes, **_):
        super().__init__()

        if backbone is not None:
            self.model = backbone
            self.model.head = nn.Linear(out_features, architecture)
        else:
            self.model = nn.Linear(out_features, architecture)

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def pred_fn(self, logits):
        return logits.argmax(dim=1)

    def loss_fn(self, logits, genus_labels, species_labels):
        return self.criterion(logits, species_labels)
