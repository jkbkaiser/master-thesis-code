import geoopt
import torch
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


class ClassifierModule(nn.Module):
    def __init__(self, in_features, level_dims, embedding_dim):
        super().__init__()
        self.classifiers = nn.ModuleList([
            Mlp(in_features, 4096, embedding_dim) for _ in level_dims
        ])

    def forward(self, features):
        return [classifier(features) for classifier in self.classifiers]


class HierarchicalPoincareRest(nn.Module):
    def __init__(self, backbone, in_features, architecture, prototypes, num_active_levels=None, **_):
        super().__init__()

        print("Instantiating HierarchicalPoincareRest")

        self.level_sizes = architecture
        self.num_levels = len(self.level_sizes)
        self.num_active_levels = num_active_levels or self.num_levels
        self.criterion = nn.CrossEntropyLoss()

        self.prototypes = prototypes
        self.split_prototypes = self._split_prototypes(prototypes, self.level_sizes)
        self.level_prototypes = self.split_prototypes[-self.num_active_levels:]  # bottom-up selection

        prototype_dim = self.prototypes.shape[1]

        if backbone is not None:
            self.model = backbone
            self.model.head = ClassifierModule(in_features, self.level_sizes, prototype_dim)
        else:
            self.model = Mlp(in_features, 256, prototype_dim)

        self.ball = geoopt.PoincareBallExact(c=1.5)

    def _split_prototypes(self, prototypes, level_sizes):
        splits = torch.split(prototypes, level_sizes, dim=0)
        return list(splits)

    def forward(self, x):
        features = self.model(x)[-self.num_active_levels:]  # bottom-up
        hyp_embeddings = [self.ball.expmap0(f) for f in features]
        return [
            -self.ball.dist(p[None, :, :], f[:, None, :]) / 0.07
            for p, f in zip(self.level_prototypes, hyp_embeddings)
        ]

    def embed(self, x):
        features = self.model(x)[-self.num_active_levels:]
        return [self.ball.expmap0(f) for f in features]

    def pred_fn(self, logits):
        logits = logits[-self.num_active_levels:]
        return [logit.argmax(dim=1) for logit in logits]

    def loss_fn(self, logits, *targets):
        logits = logits[-self.num_active_levels:]
        targets = targets[-self.num_active_levels:]
        losses = [self.criterion(logit, target) for logit, target in zip(logits, targets)]
        weights = torch.tensor([2**i for i in range(len(losses))], dtype=torch.float32, device=logits[0].device)
        weights /= weights.sum()
        return sum(w * loss for w, loss in zip(weights, losses))

