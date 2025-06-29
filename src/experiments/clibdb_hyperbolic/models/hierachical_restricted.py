import geoopt
import torch
import torch.nn as nn
import torch.nn.functional as F


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
    def __init__(self, backbone, in_features, architecture, prototypes, ds, num_active_levels=None, **_):
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

        # Store hierarchy mappings from dataset
        self.hierarchy_maps = [torch.tensor(H.T, device=prototypes.device) for H in ds.hierarchy]

    def _split_prototypes(self, prototypes, level_sizes):
        splits = torch.split(prototypes, level_sizes, dim=0)
        return list(splits)

    def forward(self, x):
        features = self.model(x)
        hyp_embeddings = [self.ball.expmap0(f) for f in features]

        # Compute logits only for active levels
        active_features = hyp_embeddings[-self.num_active_levels:]
        active_prototypes = self.split_prototypes[-self.num_active_levels:]

        logits_active = [
            -self.ball.dist(p[None, :, :], f[:, None, :]) / 0.07
            for p, f in zip(active_prototypes, active_features)
        ]

        # Initialize full logits with None
        logits_full = [None for _ in range(self.num_levels)]

        # Fill in the active levels (from bottom up)
        for i, l in enumerate(range(self.num_levels - self.num_active_levels, self.num_levels)):
            logits_full[l] = logits_active[i]

        # Marginalize missing levels using hierarchy
        for i in reversed(range(self.num_levels - self.num_active_levels)):
            # Parent = i, Child = i+1
            child_logits = logits_full[i + 1]  # shape: [B, num_children]
            mapping = self.hierarchy_maps[i].float().to(child_logits.device)  # shape: [num_parents, num_children]

            # Use matrix multiplication: [B, num_children] @ [num_children, num_parents] -> [B, num_parents]
            logits_full[i] = child_logits @ mapping.T

        return logits_full


    def embed(self, x):
        features = self.model(x)[-self.num_active_levels:]
        return [self.ball.expmap0(f) for f in features]

    def pred_fn(self, logits):
        return [logit.argmax(dim=1) for logit in logits]

    def loss_fn(self, logits, *targets):
        losses = [self.criterion(logit, target) for logit, target in zip(logits, targets)]
        weights = torch.tensor([2**i for i in range(len(losses))], dtype=torch.float32, device=logits[0].device)
        weights /= weights.sum()
        return sum(w * loss for w, loss in zip(weights, losses))

