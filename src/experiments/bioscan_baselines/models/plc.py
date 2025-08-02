import torch
import torch.nn as nn


class ClassifierModule(nn.Module):
    def __init__(self, out_features, architecture):
        super().__init__()
        self.classifiers = nn.ModuleList(
            [nn.Sequential(
                # nn.Dropout(0.3),
                nn.Linear(out_features, nc)
            ) for nc in architecture]
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

        self.criterion = nn.CrossEntropyLoss(
            # label_smoothing=0.1
        )

    def forward(self, x):
        feat = self.model.forward_features(x)
        return self.model.head(feat), feat

    def pred_fn(self, logits, *args, **kwargs):
        return [logit.argmax(dim=1) for logit in logits]

    def loss_fn(self, logits, *targets):
        losses = [self.criterion(logit, target) for logit, target in zip(logits, targets)]
        weights = torch.tensor([2**i for i in range(len(losses))], dtype=torch.float32)
        weights /= weights.sum()
        return sum(w * loss for w, loss in zip(weights, losses))
