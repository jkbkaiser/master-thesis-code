import geoopt
import torch
import torch.nn as nn


class Uniform(nn.Module):
    def __init__(self, backbone, out_features, architecture, prototypes, **_):
        super().__init__()
        self.model = backbone
        self.model.head = nn.Linear(out_features, 64)
        self.ball = geoopt.PoincareBallExact(c=1)

        self.prototypes = prototypes
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        out_feature_euc = self.model(x)
        # out_feature_hyp = self.ball.expmap0(out_feature_euc)
        # print("F")
        # print(self.prototypes.shape)
        # print(out_feature_hyp.shape)
        # return -self.ball.dist(self.prototypes, out_feature_hyp[:, None, :])

        return -torch.cdist(out_feature_euc, self.prototypes)

        # return out_feature_euc


    def pred_fn(self, logits):
        # print("p", logits.shape)
        # print(logits)
        return logits.argmax(dim=1)

    def loss_fn(self, logits, genus_labels, species_labels):
        return self.criterion(logits, species_labels)
