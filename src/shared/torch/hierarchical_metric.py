import torch

from src.constants import DEVICE


class HierarchicalMetric:
    def __init__(self, ds, num_classes_per_level):
        self.ds = ds
        self.num_classes = num_classes_per_level  # list like [num_genus, num_species, ...]
        self.frequencies = [freq.to(DEVICE) for freq in ds.frequencies]
        self.reset()

    def reset(self):
        self.conf_matrices = [
            torch.zeros((n, n), dtype=torch.int64).to(DEVICE) for n in self.num_classes
        ]
        self.lca_depths = []

    def process_batch(self, preds, labels, logits=None, split="train"):
        if logits is not None and not isinstance(logits, (list, tuple)):
            logits = self._split_logits(logits)

        correct_all = torch.ones_like(labels[0], dtype=torch.bool)
        metrics = {}

        for i, (pred, label) in enumerate(zip(preds, labels)):
            logit = logits[i] if logits is not None else None
            level_metrics, correct = self.process_level(i, pred, label, logit, split)
            metrics.update(level_metrics)
            correct_all &= correct

        metrics["accuracy_all"] = correct_all.float().mean().item()
        metrics["accuracy_avg"] = sum(
            v for k, v in metrics.items() if k.startswith("accuracy_level_")
        ) / len(self.num_classes)

        if split == "valid":
            lca_depths = self.compute_lca_depths(preds, labels)
            self.lca_depths.append(lca_depths)

        return metrics

    def compute_lca_depths(self, preds, labels):
        num_levels = len(preds)
        B = preds[0].size(0)
        lca = torch.zeros(B, dtype=torch.int64, device=preds[0].device)

        for i in range(num_levels):
            match = preds[i] == labels[i]
            if i == 0:
                still_matching = match
            else:
                still_matching &= match
            lca += still_matching.long()
        return lca


    def process_level(self, level_idx, preds, labels, logits=None, split="train"):
        metrics = {}
        correct = preds == labels
        metrics[f"accuracy_level_{level_idx}"] = correct.float().mean().item()

        if split == "valid":
            for t, p in zip(labels.view(-1), preds.view(-1)):
                if 0 <= t < self.num_classes[level_idx] and 0 <= p < self.num_classes[level_idx]:
                    self.conf_matrices[level_idx][t, p] += 1

        return metrics, correct

    def _split_logits(self, flat_logits):
        """Splits concatenated logits into a list per level"""
        split_logits = []
        offset = 0
        for n in self.num_classes:
            split_logits.append(flat_logits[:, offset:offset + n])
            offset += n
        return split_logits

    def topk_accuracy(self, logits, labels, topk=(5,)):
        with torch.no_grad():
            _, pred = logits.topk(max(topk), dim=1, largest=True, sorted=True)
            correct = pred.eq(labels.view(1, -1).expand_as(pred))
            return {
                f"top{k}_accuracy": (correct[:k].reshape(-1).float().sum() / labels.size(0)).item()
                for k in topk
            }

    def compute_lca_stats(self):
        if not self.lca_depths:
            return {}

        all_depths = torch.cat(self.lca_depths)
        stats = {
            "lca_avg": all_depths.float().mean().item(),
            "lca_max": all_depths.max().item(),
        }

        # Optionally: per-depth frequency
        for d in range(len(self.num_classes)):
            stats[f"lca_depth_{d}"] = (all_depths == d).float().mean().item()

        return stats

    def compute_recall(self):
        recall_metrics = {}
        for i, (conf_matrix, freq) in enumerate(zip(self.conf_matrices, self.frequencies)):
            tp = torch.diag(conf_matrix)
            fn = conf_matrix.sum(dim=1) - tp
            support = conf_matrix.sum(dim=1)
            per_class_recall = tp / (tp + fn).clamp(min=1)
            nonzero_mask = support > 0

            for k in [5, 10, 25, 50, 100]:
                mask = freq <= k
                tp_k = (tp * mask).sum()
                fn_k = (fn * mask).sum()
                support_k = support[mask].sum()

                macro = per_class_recall[mask & nonzero_mask].mean().item()
                micro = (tp_k / (tp_k + fn_k).clamp(min=1)).item()
                weighted = (
                    (per_class_recall[mask] * (support[mask] / support_k.clamp(min=1))).sum().item()
                    if support_k > 0 else 0.0
                )

                recall_metrics[f"level_{i}_macro_recall_{k}"] = macro
                recall_metrics[f"level_{i}_micro_recall_{k}"] = micro
                recall_metrics[f"level_{i}_support_weighted_recall_{k}"] = weighted

            recall_metrics[f"level_{i}_macro_recall_all"] = per_class_recall.mean().item()
            recall_metrics[f"level_{i}_micro_recall_all"] = (
                tp.sum() / (tp + fn).sum().clamp(min=1)
            ).item()
            recall_metrics[f"level_{i}_support_weighted_recall_all"] = (
                (per_class_recall * (support / support.sum().clamp(min=1))).sum().item()
            )

        return recall_metrics
