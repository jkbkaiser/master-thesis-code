import os
from pathlib import Path

import lightning as L
import torch
import torch.optim as optim

from src.experiments.gbif_baselines.models import HAC, MARG, MPLC, PLC
from src.shared.datasets import Dataset, DatasetType, DatasetVersion
from src.shared.torch.backbones import (ViTAEv2_B, load_for_transfer_learning,
                                        t2t_vit_t_14)
from src.shared.torch.metric import Metric

PRETRAINED_WEIGHTS_DIR = Path(os.getcwd()) / "pretrained_weights"
PRETRAINED_T2T_VITT_T14 = PRETRAINED_WEIGHTS_DIR / "81.7_T2T_ViTt_14.pth.tar"
PRETRAINED_VITAEv2 = PRETRAINED_WEIGHTS_DIR / "ViTAEv2-B.pth.tar"

BACKBONE_DICT = {
    "t2t_vit": (t2t_vit_t_14, PRETRAINED_T2T_VITT_T14, 384),
    "vitaev2": (ViTAEv2_B, PRETRAINED_VITAEv2, 1024),
}

MODEL_DICT = {"hac": HAC, "plc": PLC, "mplc": MPLC, "marg": MARG}

torch.set_float32_matmul_precision("medium")


def create_model(model_name, model_hparams, ds):
    if ds.version in [DatasetVersion.GBIF_GENUS_SPECIES_10K_EMBEDDINGS, DatasetVersion.GBIF_FLAT_10K_EMBEDDINGS]:
        out_features = 384
        backbone = None
    else:
        # Load pretrained weights
        init, path_to_weights, out_features = BACKBONE_DICT[model_hparams["backbone_name"]]
        backbone = init()
        load_for_transfer_learning(
            backbone,
            path_to_weights,
            use_ema=True,
            strict=False,
        )

    # Initialize model
    cls = MODEL_DICT[model_name]
    model = cls(backbone, out_features, **model_hparams, ds=ds)

    if ds.version in [DatasetVersion.GBIF_GENUS_SPECIES_10K_EMBEDDINGS, DatasetVersion.GBIF_FLAT_10K_EMBEDDINGS]:
        pass
    else:
        if model_hparams["freeze_backbone"]:
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
        genus_preds, species_preds = self.pred_fn(logits)
        metrics = self.metric.process_train_batch(genus_preds, genus_labels, species_preds, species_labels)

        self.log_epoch(loss, "train_loss")
        self.log_epoch(metrics, "train_")

        return loss

    def validation_step(self, batch):
        imgs, genus_labels, species_labels = batch
        logits = self(imgs)
        genus_preds, species_preds = self.pred_fn(logits)
        metrics = self.metric.process_valid_batch(genus_preds, genus_labels, species_preds, species_labels)

        self.log_epoch(metrics, "valid_")

    def on_validation_epoch_end(self):
        recall = self.metric.compute_recall()

        self.log_epoch(recall, "valid_recall_")
