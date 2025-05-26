import torch

from src.constants import DEVICE


class Metric():
    def __init__(self, ds, num_genus: int, num_species: int):
        self.ds = ds
        self.num_genus = num_genus
        self.num_species = num_species

        self.species_freq = self.ds.frequencies[1].to(DEVICE)
        self.genus_freq = self.ds.frequencies[0].to(DEVICE)
        self.reset()

    def reset(self):
        self.valid_conf_m_species = torch.zeros((self.num_species, self.num_species), dtype=torch.int64).to(DEVICE)
        self.valid_conf_m_genus = torch.zeros((self.num_genus, self.num_genus), dtype=torch.int64).to(DEVICE)

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

    def compute_valid_conf_m_species(self, species_preds, species_labels):
        self.valid_conf_m_species.index_add_(
            0,
            species_labels.view(-1),
            torch.eye(self.num_species, device=DEVICE)[
                species_preds.view(-1)
            ].to(torch.int64),
        )

    def compute_valid_conf_m_genus(self, genus_preds, genus_labels):
        self.valid_conf_m_genus.index_add_(
            0,
            genus_labels.view(-1),
            torch.eye(self.num_genus, device=DEVICE)[
                genus_preds.view(-1)
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

        self.compute_valid_conf_m_species(species_preds, species_labels)
        self.compute_valid_conf_m_genus(genus_preds, genus_labels)

        return batch_metrics

    def compute_recall(self, matrix, freq):
        tp_per_class = torch.diag(matrix)
        fn_per_class = matrix.sum(dim=1) - tp_per_class

        support = matrix.sum(dim=1)
        nonzero_mask = support > 0
        per_class_recall = tp_per_class / (tp_per_class + fn_per_class).clamp(min=1)

        recalls = {}

        for k in [5, 10, 50, 100]:
            mask = (freq <= k)

            # Raw counts for micro recall
            tp = (tp_per_class * mask).sum()
            fn = (fn_per_class * mask).sum()
            recalls[f"micro_recall_{k}"] = (tp / (tp + fn).clamp(min=1)).item()

            # Macro recall (equal class weight)
            class_recalls = per_class_recall[mask & nonzero_mask]
            recalls[f"macro_recall_{k}"] = class_recalls.mean().item()

            # Weighted recall (by support)
            class_support = support[mask]
            weight = class_support / class_support.sum().clamp(min=1)
            weighted_recall = (per_class_recall[mask] * weight).sum()
            recalls[f"support_weighted_recall_{k}"] = weighted_recall.item()

        # Global micro recall
        tp = tp_per_class.sum()
        fn = fn_per_class.sum()
        recalls["micro_recall_all"] = (tp / (tp + fn).clamp(min=1)).item()

        # Global macro recall
        recalls["macro_recall_all"] = per_class_recall.mean().item()

        # Global support-weighted recall
        weight = support / support.sum().clamp(min=1)
        weighted_recall = (per_class_recall * weight).sum()
        recalls["support_weighted_recall_all"] = weighted_recall.item()
        return recalls
