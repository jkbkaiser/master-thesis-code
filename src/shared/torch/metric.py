import torch

from src.constants import DEVICE


class Metric():
    def __init__(self, ds, num_genus: int, num_species: int):
        self.ds = ds
        self.num_genus = num_genus
        self.num_species = num_species
        freq = self.ds.frequencies[1]
        self.freq = freq.to(DEVICE)
        self.reset()

    def reset(self):
        self.valid_conf_m = torch.zeros((self.num_species, self.num_species), dtype=torch.int64).to(DEVICE)

    def process_train_batch(self, genus_preds, genus_labels, species_logits, species_preds, species_labels):
        correct_genus = genus_preds == genus_labels
        correct_species = species_preds == species_labels
        correct_both = correct_species & correct_genus

        acc_genus = correct_genus.float().mean().item()
        acc_species = correct_species.float().mean().item()
        acc_all = correct_both.float().mean().item()

        acc_top_k = self.topk_accuracy(species_logits, species_labels)

        batch_metrics = {
            "accuracy_genus": acc_genus,
            "accuracy_species": acc_species,
            "accuracy_avg": (acc_genus + acc_species) / 2,
            "accuracy_all": acc_all,
        }

        batch_metrics.update(acc_top_k)

        return batch_metrics

    def compute_valid_conf_m(self, species_preds, species_labels):
        self.valid_conf_m.index_add_(
            0,
            species_labels.view(-1),
            torch.eye(self.num_species, device=DEVICE)[
                species_preds.view(-1)
            ].to(torch.int64),
        )

    def topk_accuracy(self, species_logits, species_labels, topk=(5,)):
        with torch.no_grad():
            max_k = max(topk)
            _, pred = species_logits.topk(max_k, dim=1, largest=True, sorted=True)  # (B, max_k)
            pred = pred.t()
            correct = pred.eq(species_labels.view(1, -1).expand_as(pred))  # (max_k, B)

            res = {}
            for k in topk:
                correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
                res[f"top{k}_accuracy"] = (correct_k / species_labels.size(0)).item()

            return res

    def process_valid_batch(self, genus_preds, genus_labels, species_logits, species_preds, species_labels):
        correct_genus = genus_preds == genus_labels
        correct_species = species_preds == species_labels
        correct_both = correct_species & correct_genus

        acc_genus = correct_genus.float().mean().item()
        acc_species = correct_species.float().mean().item()
        acc_all = correct_both.float().mean().item()

        acc_top_k = self.topk_accuracy(species_logits, species_labels)

        batch_metrics = {
            "accuracy_genus": acc_genus,
            "accuracy_species": acc_species,
            "accuracy_avg": (acc_genus + acc_species) / 2,
            "accuracy_all": acc_all,
        }

        batch_metrics.update(acc_top_k)

        self.compute_valid_conf_m(species_preds, species_labels)

        return batch_metrics

    def compute_recall(self):
        matrix = self.valid_conf_m

        tp_per_class = torch.diag(matrix)
        fn_per_class = matrix.sum(dim=1) - tp_per_class

        recalls = {}

        for k in [5, 10, 50, 100]:
            mask = self.freq <= k  # Select species with occurrence ≤ k

            tp = (tp_per_class * mask).sum()
            fn = (fn_per_class * mask).sum()

            recall = tp / (tp + fn).clamp(min=1)
            recalls[str(k)] = recall.item()

        tp = (tp_per_class).sum()
        fn = (fn_per_class).sum()

        recall = tp / (tp + fn).clamp(min=1)
        recalls["all"] = recall.item()

        return recalls
