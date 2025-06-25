import torch
import torch.nn as nn


class Custom(nn.Module):
    def __init__(self, backbone, **_):
        super().__init__()
        self.model = backbone
        self.criterion = nn.CrossEntropyLoss()

        torch.autograd.set_detect_anomaly(True)

    def forward(self, x):
        return self.model(x)

    def pred_fn(self, logits):
        return [logit.argmax(dim=1) for logit in logits]

    def loss_fn(self, logits, *labels):
        losses = [
            self.criterion(logit, label)
            for logit, label in zip(logits, labels)
        ]

        weights = torch.linspace(0.3, 1.0, steps=len(losses))**2
        weights = (weights / weights.sum()).to(losses[0].device)
        return torch.stack(losses).dot(weights)
