import torch
import torch.nn as nn
import torch.nn.functional as F


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class Hyperspherical(nn.Module):
    def __init__(self, backbone, out_features, architecture, prototypes, **_):
        super().__init__()

        if backbone is not None:
            self.model = backbone
            self.model.head = Mlp(out_features, 512, prototypes.shape[1])
        else:
            self.model = Mlp(out_features, 512, prototypes.shape[1])

        self.prototypes = prototypes
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        out_feature_euc = self.model(x)
        out_feature_sphere = F.normalize(out_feature_euc, p=2, dim=1)
        return -torch.cdist(out_feature_sphere, self.prototypes)

    def pred_fn(self, logits):
        return logits.argmax(dim=1)

    def loss_fn(self, logits, genus_labels, species_labels):
        return self.criterion(logits, species_labels)
