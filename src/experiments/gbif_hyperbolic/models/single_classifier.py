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

class SingleClassifier(nn.Module):
    def __init__(self, backbone, out_features, architecture, prototypes, prototype_dim, temp, **_):
        super().__init__()

        if backbone is not None:
            self.model = backbone
            self.model.head = Mlp(out_features, 4096, prototype_dim)  # Only one classifier head for species
        else:
            self.model = Mlp(out_features, 256, prototype_dim)

        c = 3

        self.ball = geoopt.PoincareBallExact(c=c)

        self.prototypes = prototypes

        [self.num_genus, self.num_species] = architecture
        self.genus_prototypes = self.prototypes[:self.num_genus]
        self.species_protypes = self.prototypes[self.num_genus:]

        self.temp = temp
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        # Only compute species logits
        feature_euc_species = self.model(x)
        feature_hyp_species = self.ball.expmap0(feature_euc_species)

        # Calculate distances to genus prototypes
        genus_distances = self.ball.dist(self.genus_prototypes[None, :, :], feature_hyp_species[:, None, :])

        # Return the distances to genus prototypes and species logits
        return -genus_distances / self.temp, -self.ball.dist(self.species_protypes[None, :, :], feature_hyp_species[:, None, :]) / self.temp

    def pred_fn(self, logits):
        [genus_logits, species_logits] = logits
        return genus_logits.argmax(dim=1), species_logits.argmax(dim=1)

    def loss_fn(self, logits, genus_labels, species_labels):
        genus_logits, species_logits = logits

        # Use species labels for CrossEntropyLoss
        ce_loss_species = self.criterion(species_logits, species_labels)

        # Compute genus loss based on distance to closest genus prototype
        # genus_loss = genus_logits.min(dim=1)[0].mean()

        return ce_loss_species

