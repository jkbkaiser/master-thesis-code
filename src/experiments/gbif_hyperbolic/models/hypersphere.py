import geoopt
import torch.nn as nn
import torch.nn.functional as F


class HyperSphere(nn.Module):
    def __init__(self, backbone, out_features, architecture, prototypes, **_):
        super().__init__()
        self.model = backbone
        self.model.head = nn.Linear(out_features, prototypes.shape[1])
        self.ball = geoopt.PoincareBallExact(c=1)

        self.prototypes = prototypes
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        out_feature_euc = self.model(x)
        out_feature_sphere = F.normalize(out_feature_euc, p=2, dim=1)

        # out_feature_hyp = self.ball.expmap0(out_feature_euc)
        # return -self.ball.dist(self.prototypes, out_feature_hyp[:, None, :])
        # return out_feature_euc

        return out_feature_sphere @ self.prototypes.T



    def pred_fn(self, logits):
        # print("p", logits.shape)
        # print(logits)
        return logits.argmax(dim=1)

    def loss_fn(self, logits, genus_labels, species_labels):
        return self.criterion(logits, species_labels)
