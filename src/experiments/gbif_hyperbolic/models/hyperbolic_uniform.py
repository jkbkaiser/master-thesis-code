import geoopt
import torch.nn as nn

import torch
import torch.nn.functional as F

def hyperbolic_contrastive_loss(embeddings, labels, ball, temperature=0.07):
    """
    embeddings: [B, D] Euclidean embeddings
    labels: [B] int class labels
    ball: geoopt.PoincareBallExact
    """
    # Map to hyperbolic space
    hyp_emb = ball.expmap0(embeddings)  # [B, D]

    # Compute all pairwise hyperbolic distances
    dist_matrix = ball.dist(hyp_emb[:, None, :], hyp_emb[None, :, :])  # [B, B]

    # Convert distances to similarity logits (negative distance)
    logits = -dist_matrix / temperature

    # Mask to remove self-similarity
    B = labels.size(0)
    mask = torch.eye(B, dtype=torch.bool, device=labels.device)
    logits = logits.masked_fill(mask, float('-inf'))

    # Create label similarity mask
    sim_mask = labels[:, None] == labels[None, :]  # [B, B]

    # InfoNCE loss: log-softmax over similarities, mean over positives
    log_prob = F.log_softmax(logits, dim=1)
    loss = -(log_prob * sim_mask).sum(1) / sim_mask.sum(1).clamp(min=1)
    return loss.mean()


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


class HyperbolicUniform(nn.Module):
    def __init__(self, backbone, out_features, architecture, prototypes, **_):
        super().__init__()

        if backbone is not None:
            self.model = backbone
            self.model.head = Mlp(out_features, 4096, prototypes.shape[1])
        else:
            self.model = Mlp(out_features, 256, prototypes.shape[1])

        c = 3

        self.ball = geoopt.PoincareBallExact(c=c)
        self.prototypes = (prototypes * 0.95) / c
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        out_feature_euc = self.model(x)
        out_feature_hyp = self.ball.expmap0(out_feature_euc)
        return -self.ball.dist(self.prototypes, out_feature_hyp[:, None, :]) / 0.07, out_feature_hyp

    def pred_fn(self, logits):
        return logits.argmax(dim=1)

    def loss_fn(self, logits, genus_labels, species_labels, hyp_emb):
        ce_loss = self.criterion(logits, species_labels)
        contrastive = hyperbolic_contrastive_loss(hyp_emb, species_labels, self.ball, temperature=0.07)
        return ce_loss + 0.1 * contrastive
