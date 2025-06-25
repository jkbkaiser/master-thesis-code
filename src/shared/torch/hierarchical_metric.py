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

    def process_batch(self, preds, labels, logits=None, split="train"):
        correct_all = torch.ones_like(labels[0], dtype=torch.bool)
        metrics = {}

        for i, (p, y) in enumerate(zip(preds, labels)):
            correct = p == y
            correct_all &= correct
            metrics[f"accuracy_level_{i}"] = correct.float().mean().item()

            if split == "valid":
                self.conf_matrices[i].index_add_(
                    0,
                    y.view(-1),
                    torch.eye(self.num_classes[i], device=DEVICE)[p.view(-1)].to(torch.int64)
                )

        metrics["accuracy_all"] = correct_all.float().mean().item()
        metrics["accuracy_avg"] = sum(
            v for k, v in metrics.items() if "accuracy_level_" in k
        ) / len(self.num_classes)

        if logits:
            metrics.update(self.topk_accuracy(logits[-1], labels[-1]))

        return metrics

    def topk_accuracy(self, logits, labels, topk=(5,)):
        with torch.no_grad():
            max_k = max(topk)
            # logits: [B, num_classes]
            # labels: [B]
            _, pred = logits.topk(max_k, dim=1, largest=True, sorted=True)  # [B, max_k]
            correct = pred.eq(labels.view(-1, 1).expand_as(pred))  # [B, max_k]

            return {
                f"top{k}_accuracy": (correct[:, :k].float().sum() / labels.size(0)).item()
                for k in topk
            }

    # def topk_accuracy(self, logits, labels, topk=(5,)):
    #     with torch.no_grad():
    #         _, pred = logits.topk(max(topk), dim=1, largest=True, sorted=True)
    #         correct = pred.eq(labels.view(1, -1).expand_as(pred))
    #         return {
    #             f"top{k}_accuracy": (correct[:k].reshape(-1).float().sum() / labels.size(0)).item()
    #             for k in topk
    #         }

    def compute_recall(self):
        recall_metrics = {}
        for i, (conf_matrix, freq) in enumerate(zip(self.conf_matrices, self.frequencies)):
            tp = torch.diag(conf_matrix)
            fn = conf_matrix.sum(dim=1) - tp
            support = conf_matrix.sum(dim=1)
            per_class_recall = tp / (tp + fn).clamp(min=1)
            nonzero_mask = support > 0

            for k in [5, 10, 50, 100]:
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
