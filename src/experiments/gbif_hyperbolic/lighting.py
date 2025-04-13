import math
import os
from pathlib import Path

import lightning as L
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from src.constants import DEVICE
from src.experiments.gbif_hyperbolic.models.hypersphere import HyperSphere
from src.shared.datasets import Dataset, DatasetType
from src.shared.torch.backbones import (ViTAEv2_B, load_for_transfer_learning,
                                        t2t_vit_t_14)
from src.shared.torch.metric import Metric

PRETRAINED_WEIGHTS_DIR = Path(os.getcwd()) / "pretrained_weights"
PRETRAINED_T2T_VITT_T14 = PRETRAINED_WEIGHTS_DIR / "81.7_T2T_ViTt_14.pth.tar"
PRETRAINED_VITAEv2 = PRETRAINED_WEIGHTS_DIR / "ViTAEv2-B.pth.tar"

torch.set_float32_matmul_precision("high")

MODEL_DICT = {
    "hypersphere": HyperSphere,
}

BACKBONE_DICT = {
    "t2t_vit": (t2t_vit_t_14, PRETRAINED_T2T_VITT_T14, 384),
    "vitaev2": (ViTAEv2_B, PRETRAINED_VITAEv2, 1024),
}

def get_prototypes(prototype, ds):
    # curvature = 1
    # scale_factor = 0.95

    prototype_path = Path("./prototypes") / ds.version.value / f"prototypes-{prototype}-{ds.version.value}.npy"
    prototypes = torch.from_numpy(np.load(prototype_path)).float()
    prototypes = F.normalize(prototypes, p=2, dim=1).to(DEVICE)

    # prototypes = prototypes.cuda() * scale_factor / math.sqrt(curvature)

    return prototypes

def create_model(model_name, model_hparams, ds):
    prototypes = get_prototypes(model_hparams["prototypes"], ds)
    model_hparams["prototypes"] = prototypes

    init, path_to_weights, out_features = BACKBONE_DICT[model_hparams["backbone_name"]]
    backbone = init(
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
    )

    load_for_transfer_learning(
        backbone,
        path_to_weights,
        use_ema=True,
        strict=False,
    )

    cls = MODEL_DICT[model_name]
    model = cls(backbone, out_features, **model_hparams, ds=ds)

    # if model_hparams["freeze_backbone"]:
    print("Freezing backbone")
    for param in model.model.parameters():
        param.requires_grad = False

    for param in model.model.head.parameters():
        param.requires_grad = True

    return model

def get_num_classes(ds: Dataset):
    if ds.type == DatasetType.GENUS_SPECIES:
        return ds.labelcount_per_level
    if ds.type == DatasetType.FLAT:
        total = ds.labelcount_per_level[0]
        split = ds.metadata["per_level"][0]["split"]
        return split, total - split
    raise Exception("could not retrieve num classes")


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
        if not self.freeze_backbone:
            self.freeze_epochs = model_hparams["freeze_epochs"]

        self.pred_fn = self.model.pred_fn
        self.loss_fn = self.model.loss_fn

        [self.num_classes_genus, self.num_classes_species] = get_num_classes(ds)
        self.metric: Metric = Metric(ds, self.num_classes_genus, self.num_classes_species)

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
        params = filter(lambda p: p.requires_grad, self.model.parameters())

        optim_dict = {
            "adam": optim.AdamW,
            "sgd": optim.SGD,
        }

        optimizer = optim_dict[self.optimizer_name](params, **self.optimizer_hparams)

        total_steps = self.trainer.estimated_stepping_batches

        assert total_steps is not None
        assert self.trainer.max_epochs is not None

        epoch_steps = total_steps // int(self.trainer.max_epochs)

        fixed_lr_epochs = self.freeze_epochs
        step_offset = fixed_lr_epochs * epoch_steps

        warmup_steps = 500
        warmup_lr_init = 1e-6
        base_lr = self.optimizer_hparams["lr"]
        min_lr = 1e-4

        def lr_lambda(current_step):
            current_epoch = current_step // epoch_steps

            if current_epoch < fixed_lr_epochs:
                return 1e-3 / base_lr

            adjusted_step = current_step - step_offset

            if adjusted_step < warmup_steps:
                return (warmup_lr_init / base_lr) + (
                    (1.0 - warmup_lr_init / base_lr) * (adjusted_step / warmup_steps)
                )

            decay_steps = total_steps - warmup_steps
            decay_step = adjusted_step - warmup_steps
            cosine_decay = 0.5 * (1 + math.cos(math.pi * decay_step / decay_steps))
            return (min_lr / base_lr) + (1 - min_lr / base_lr) * cosine_decay

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    def training_step(self, batch):
        self.log("step", self.current_epoch)

        imgs, genus_labels, species_labels = batch
        logits = self(imgs)
        loss = self.loss_fn(logits, genus_labels, species_labels)
        species_preds = self.pred_fn(logits)

        genus_preds = torch.zeros_like(genus_labels)
        metrics = self.metric.process_train_batch(genus_preds, genus_labels, species_preds, species_labels)

        self.log_epoch(loss, "train_loss")
        self.log_epoch(metrics, "train_")

        lr = self.trainer.optimizers[0].param_groups[0]["lr"]

        self.log_epoch(lr, "lr")

        return loss

    def unfreeze_backbone(self):
        for param in self.model.model.parameters():
            param.requires_grad = True

    def on_train_epoch_start(self):
        if not self.freeze_backbone:
            if self.current_epoch == self.freeze_epochs:
                self.unfreeze_backbone()

    def validation_step(self, batch):
        imgs, genus_labels, species_labels = batch
        logits = self(imgs)
        species_preds = self.pred_fn(logits)

        genus_preds = torch.zeros_like(genus_labels)
        metrics = self.metric.process_valid_batch(genus_preds, genus_labels, species_preds, species_labels)

        self.log_epoch(metrics, "valid_")

    def on_validation_epoch_end(self):
        recall = self.metric.compute_recall()

        self.log_epoch(recall, "valid_recall_")
