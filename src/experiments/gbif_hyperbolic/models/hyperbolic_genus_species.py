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


class ClassifierModule(nn.Module):
    def __init__(self, out_features, architecture, embedding_dim):
        super().__init__()
        self.classifiers = nn.ModuleList(
            [
                Mlp(out_features, 4096, embedding_dim)
                for _ in architecture
            ]
        )

    def forward(self, features):
        return [classifier(features) for classifier in self.classifiers]


class GenusSpeciesPoincare(nn.Module):
    def __init__(self, backbone, out_features, architecture, prototypes, prototype_dim, temp, **_):
        super().__init__()

        if backbone is not None:
            self.model = backbone
            self.model.head = ClassifierModule(out_features, architecture, prototype_dim)
        else:
            self.model = Mlp(out_features, 256, prototypes.shape[1])

        c = 1.5

        self.ball = geoopt.PoincareBallExact(c=c)
        self.prototypes = prototypes

        [self.num_genus, self.num_species] = architecture
        self.genus_prototypes = self.prototypes[:self.num_genus]
        self.species_protypes = self.prototypes[self.num_genus:]

        self.temp = temp
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        [feature_euc_genus, feature_euc_species] = self.model(x)
        feature_hyp_genus = self.ball.expmap0(feature_euc_genus)
        feature_hyp_species = self.ball.expmap0(feature_euc_species)

        return [
            -self.ball.dist(self.genus_prototypes[None, :, :], feature_hyp_genus[:, None, :]) / self.temp,
            -self.ball.dist(self.species_protypes[None, :, :], feature_hyp_species[:, None, :]) / self.temp
        ]

    def pred_fn(self, logits):
        [genus_logits, species_logits] = logits
        return genus_logits.argmax(dim=1), species_logits.argmax(dim=1)

    def loss_fn(self, logits, genus_labels, species_labels):
        [genus_logits, species_logits] = logits

        ce_loss_genus = self.criterion(genus_logits, genus_labels)
        ce_loss_species = self.criterion(species_logits, species_labels)

        return 0.3 * ce_loss_genus + 0.7 * ce_loss_species
