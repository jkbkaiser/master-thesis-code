import torch
import torch.nn as nn

from src.constants import DEVICE


class HAC(nn.Module):
    def __init__(self, backbone, out_features, architecture, **_):
        """
        architecture: List[int], number of classes per level (from top to bottom)
        """
        super().__init__()
        self.level_sizes = architecture
        self.num_levels = len(architecture)
        self.level_offsets = self._compute_offsets(architecture)
        self.total_classes = sum(architecture)

        if backbone is not None:
            self.model = backbone
            self.model.head = nn.Linear(out_features, self.total_classes)
        else:
            self.model = nn.Linear(out_features, self.total_classes)

        self.masks = self._create_level_masks()
        self.criterion = nn.CrossEntropyLoss()

    def _compute_offsets(self, architecture):
        """Cumulative start index for each level"""
        offsets = [0]
        for n in architecture[:-1]:
            offsets.append(offsets[-1] + n)
        return offsets

    def _create_level_masks(self):
        masks = []
        for i in range(self.num_levels):
            mask = torch.zeros((1, self.total_classes), dtype=torch.bool, device=DEVICE)
            start = self.level_offsets[i]
            end = start + self.level_sizes[i]
            mask[:, start:end] = 1
            masks.append(mask.float())
        return masks

    def forward(self, x):
        return self.model(x)

    def mask_logits(self, logits):
        return [
            logits * mask for mask in self.masks
        ]

    def pred_fn(self, logits, *args, **kwargs):
        masked_logits = self.mask_logits(logits)
        preds = []
        for i, masked in enumerate(masked_logits):
            start = self.level_offsets[i]
            pred = masked.argmax(dim=1) - start
            preds.append(pred)
        return preds  # list of predictions per level

    def loss_fn(self, logits, *targets, **kwargs):
        """
        targets: Tuple of true labels per level
        """
        masked_logits = self.mask_logits(logits)
        losses = []
        for i, (logit, target) in enumerate(zip(masked_logits, targets)):
            offset_target = target + self.level_offsets[i]
            losses.append(self.criterion(logit, offset_target))
        return sum(losses) / len(losses)  # average across all levels

