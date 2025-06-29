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


class HierarchicalPoincare(nn.Module):
    def __init__(self, backbone, in_features, architecture, prototypes, **_):
        super().__init__()

        print("Intantiating HierarchicalPoincare")

        self.level_sizes = architecture
        self.num_levels = len(self.level_sizes)
        self.criterion = nn.CrossEntropyLoss()

        self.prototypes = prototypes
        self.level_prototypes = self._split_prototypes(prototypes, self.level_sizes)

        prototype_dim = self.prototypes.shape[1]

        if backbone is not None:
            self.model = backbone
            self.model.head = ClassifierModule(in_features, self.level_sizes, prototype_dim)
        else:
            self.model = Mlp(in_features, 256, prototype_dim)

        self.ball = geoopt.PoincareBallExact(c=1.5)

    def _split_prototypes(self, prototypes, level_sizes):
        splits = torch.split(prototypes, level_sizes, dim=0)
        return list(splits)  # List of tensors per level

    def forward(self, x):
        features = self.model(x)
        # hyp_embeddings = [self.ball.expmap0(f) for f in features]

        return [
            - torch.cdist(f, p) / 0.07
            # -self.ball.dist(p[None, :, :], f[:, None, :]) / 0.07
            # -self.ball.dist(p[None, :, :], f[:, None, :]) / 0.07
            for p, f in zip(self.level_prototypes, features)
        ]

    def embed(self, x):
        features = self.model(x)
        # hyp_embeddings = [self.ball.expmap0(f) for f in features]
        # return hyp_embeddings
        return features

    def pred_fn(self, logits):
        return [logit.argmax(dim=1) for logit in logits]

    def loss_fn(self, logits, *targets):
        losses = [self.criterion(logit, target) for logit, target in zip(logits, targets)]
        weights = torch.tensor([2**i for i in range(len(losses))], dtype=torch.float32)
        weights /= weights.sum()
        return sum(w * loss for w, loss in zip(weights, losses))
