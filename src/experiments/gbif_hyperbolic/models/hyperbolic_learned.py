import geoopt
import torch.nn as nn


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act1 = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, hidden_features)
        self.act2 = nn.GELU()
        self.fc3 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)
        x = self.act2(x)
        x = self.fc3(x)
        return x


class HyperbolicLearned(nn.Module):
    def __init__(self, backbone, out_features, architecture, prototypes, **_):
        super().__init__()

        if backbone is not None:
            self.model = backbone
            self.model.head = Mlp(out_features, 512, prototypes.shape[1])
        else:
            self.model = Mlp(out_features, 256, prototypes.shape[1])

        c = 3

        self.ball = geoopt.PoincareBallExact(c=c)
        self.prototypes = nn.Parameter((prototypes * 0.95) / c)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        out_feature_euc = self.model(x)
        out_feature_hyp = self.ball.expmap0(out_feature_euc)
        return -self.ball.dist(self.prototypes, out_feature_hyp[:, None, :])

    def pred_fn(self, logits):
        return logits.argmax(dim=1)

    def loss_fn(self, logits, genus_labels, species_labels):
        return self.criterion(logits, species_labels)
