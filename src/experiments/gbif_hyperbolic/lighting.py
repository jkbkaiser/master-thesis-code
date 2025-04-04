import math
import os
from pathlib import Path

import lightning as L
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from src.shared.datasets import Dataset, DatasetType
from src.shared.torch.backbones import (ViTAEv2_B, load_for_transfer_learning,
                                        t2t_vit_t_14)
from src.shared.torch.metric import Metric

from .models.uniform import Uniform

PRETRAINED_WEIGHTS_DIR = Path(os.getcwd()) / "pretrained_weights"
PRETRAINED_T2T_VITT_T14 = PRETRAINED_WEIGHTS_DIR / "81.7_T2T_ViTt_14.pth.tar"
PRETRAINED_VITAEv2 = PRETRAINED_WEIGHTS_DIR / "ViTAEv2-B.pth.tar"

torch.set_float32_matmul_precision("medium")


BACKBONE_DICT = {
    "t2t_vit": (t2t_vit_t_14, PRETRAINED_T2T_VITT_T14, 384),
    "vitaev2": (ViTAEv2_B, PRETRAINED_VITAEv2, 1024),
}

def get_prototypes(prototype_path: Path):
    curvature = 1
    scale_factor = 0.95

    prototypes = torch.from_numpy(np.load(prototype_path)).float()
    prototypes = F.normalize(prototypes, p=2, dim=1)
    prototypes = prototypes.cuda() * scale_factor / math.sqrt(curvature)
    return prototypes

def create_model(model_hparams, ds):
    prototypes = get_prototypes(model_hparams["prototypes"])
    model_hparams["prototypes"] = prototypes

    init, path_to_weights, out_features = BACKBONE_DICT[model_hparams["backbone_name"]]
    backbone = init(
        drop_rate=0.1,
        attn_drop_rate=0.1,
        drop_path_rate=0.1,
    )

    load_for_transfer_learning(
        backbone,
        path_to_weights,
        use_ema=True,
        strict=False,
    )

    model = Uniform(backbone, out_features, **model_hparams, ds=ds)

    print("Freezing backbone")
    if model_hparams["freeze_backbone"]:
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
        model_hparams,
        optimizer_name,
        optimizer_hparams,
        ds: Dataset,
    ):
        super().__init__()
        self.ds = ds
        self.optimizer_name = optimizer_name
        self.optimizer_hparams = optimizer_hparams
        self.model = create_model(model_hparams, ds)

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

        return [optimizer]

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

        return loss

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
