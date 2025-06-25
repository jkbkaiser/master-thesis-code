import math
import os
from pathlib import Path

import lightning as L
import torch

from src.experiments.gbif_hyperbolic_custom.models.custom import Custom
from src.shared.datasets import Dataset, DatasetVersion
from src.shared.prototypes import get_prototypes
from src.shared.torch.backbones import (load_for_transfer_learning,
                                        t2t_vit_t_14_custom)
from src.shared.torch.hierarchical_metric import HierarchicalMetric

PRETRAINED_WEIGHTS_DIR = Path(os.getcwd()) / "pretrained_weights"
PRETRAINED_T2T_VITT_T14 = PRETRAINED_WEIGHTS_DIR / "81.7_T2T_ViTt_14.pth.tar"

torch.set_float32_matmul_precision("high")

MODEL_DICT = {
    "custom": Custom,
}

BACKBONE_DICT = {
    "t2t_vit": (t2t_vit_t_14_custom, PRETRAINED_T2T_VITT_T14, 384),
}


def create_model(model_name, model_hparams, ds):
    prototypes = get_prototypes(model_hparams["prototypes"], ds.version.value, model_hparams["prototype_dim"])

    architecture = model_hparams["architecture"]
    taxonomic_levels = len(architecture)

    level_prototypes = []
    start = 0
    for num_classes in architecture:
        end = start + num_classes
        level_prototypes.append(prototypes[start:end])  # Slice for each level
        start = end

    model_hparams["prototypes"] = level_prototypes

    if ds.version in [DatasetVersion.GBIF_GENUS_SPECIES_10K_EMBEDDINGS]:
        out_features = 384
        backbone = None
    else:

        init, path_to_weights, out_features = BACKBONE_DICT[model_hparams["backbone_name"]]
        backbone = init(
            prototypes=level_prototypes,
            taxonomic_levels=taxonomic_levels,
            # drop_rate=0.0,
            # attn_drop_rate=0.0,
            # drop_path_rate=0.0,
        )

        load_for_transfer_learning(
            backbone,
            path_to_weights,
            use_ema=True,
            strict=False,
        )

    cls = MODEL_DICT[model_name]
    model = cls(backbone)

    if ds.version in [DatasetVersion.GBIF_GENUS_SPECIES_10K_EMBEDDINGS]:
        pass
    else:
        if model_hparams["freeze_backbone"]:
            print("Freezing backbone")
            for param in model.model.parameters():
                param.requires_grad = False

            for param in model.model.head.parameters():
                param.requires_grad = True

    return model


def linear_warmup_cosine_decay(warmup_steps, total_steps):
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return lr_lambda


class LightningGBIF(L.LightningModule):
    def __init__(
        self,
        model_name,
        model_hparams,
        optimizer_name,
        optimizer_hparams,
        ds: Dataset,
    ):
        super().__init__()
        self.ds = ds
        self.optimizer_name = optimizer_name
        self.optimizer_hparams = optimizer_hparams
        self.model = create_model(model_name, model_hparams, ds)

        self.freeze_backbone = model_hparams["freeze_backbone"]
        # self.freeze_epochs = model_hparams["freeze_epochs"]

        self.pred_fn = self.model.pred_fn
        self.loss_fn = self.model.loss_fn

        # [self.num_classes_genus, self.num_classes_species] = ds.labelcount_per_level
        # self.metric: Metric = Metric(ds, self.num_classes_genus, self.num_classes_species)

        self.metric = HierarchicalMetric(ds, ds.labelcount_per_level)

    def log_epoch(self, value, name=None):
        if isinstance(value, dict):
            if name is not None:
                value = {name + key: value for key, value in value.items()}
            self.log_dict(value, on_step=False, on_epoch=True)
        elif name is not None:
            self.log(name, value, on_step=False, on_epoch=True)

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        lr = self.optimizer_hparams["learning_rate"]
        weight_decay = self.optimizer_hparams["weight_decay"]

        params = filter(lambda p: p.requires_grad, self.model.parameters())
        optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

        return {"optimizer": optimizer}

    def training_step(self, batch):
        self.log("step", self.current_epoch)

        imgs = batch[0]
        labels = batch[1:]  # list of label tensors per taxonomic level

        logits = self(imgs)
        loss = self.loss_fn(logits, *labels)
        preds = self.pred_fn(logits)

        metrics = self.metric.process_batch(preds, labels, logits, split="train")

        self.log_epoch(loss, "train_loss")
        self.log_epoch(metrics, "train_")

        lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log_epoch(lr, "lr")

        return loss

    def validation_step(self, batch):
        imgs = batch[0]
        labels = batch[1:]

        logits = self(imgs)
        preds = self.pred_fn(logits)

        metrics = self.metric.process_batch(preds, labels, logits, split="valid")
        self.log_epoch(metrics, "valid_")

    def on_validation_epoch_end(self):
        recall_metrics = self.metric.compute_recall()
        self.log_epoch(recall_metrics, "valid_recall_")
