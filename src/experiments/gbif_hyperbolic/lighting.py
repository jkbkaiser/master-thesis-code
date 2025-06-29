import math
import os
from pathlib import Path

import lightning as L
import torch

from src.experiments.gbif_hyperbolic.models.euclidean_genus_species import \
    GenusSpeciesEucldiean
from src.experiments.gbif_hyperbolic.models.hyperbolic_genus_species import \
    GenusSpeciesPoincare
from src.experiments.gbif_hyperbolic.models.hyperbolic_uniform import \
    HyperbolicUniform
from src.experiments.gbif_hyperbolic.models.hypersphere import Hyperspherical
from src.experiments.gbif_hyperbolic.models.single_classifier import \
    SingleClassifier
from src.shared.datasets import Dataset, DatasetVersion
from src.shared.prototypes import get_prototypes
from src.shared.torch.backbones import (ViTAEv2_B, load_for_transfer_learning,
                                        t2t_vit_t_14)
from src.shared.torch.metric import Metric

PRETRAINED_WEIGHTS_DIR = Path(os.getcwd()) / "pretrained_weights"
PRETRAINED_T2T_VITT_T14 = PRETRAINED_WEIGHTS_DIR / "81.7_T2T_ViTt_14.pth.tar"
PRETRAINED_VITAEv2 = PRETRAINED_WEIGHTS_DIR / "ViTAEv2-B.pth.tar"

torch.set_float32_matmul_precision("high")

MODEL_DICT = {
    "hyperspherical": Hyperspherical,
    "hyperbolic-uniform": HyperbolicUniform,
    "hyperbolic-genus-species": GenusSpeciesPoincare,
    "single": SingleClassifier,
    "euclidean": GenusSpeciesEucldiean,
}

BACKBONE_DICT = {
    "t2t_vit": (t2t_vit_t_14, PRETRAINED_T2T_VITT_T14, 384),
    "vitaev2": (ViTAEv2_B, PRETRAINED_VITAEv2, 1024),
}


def create_model(model_name, model_hparams, ds):
    prototypes = get_prototypes(model_hparams["prototypes"], ds.version.value, model_hparams["prototype_dim"])
    print("Intializing prototypes")
    model_hparams["prototypes"] = prototypes

    if ds.version in [DatasetVersion.GBIF_GENUS_SPECIES_10K_EMBEDDINGS]:
        out_features = 384
        backbone = None
    else:
        init, path_to_weights, out_features = BACKBONE_DICT[model_hparams["backbone_name"]]
        backbone = init(
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
    model = cls(backbone, out_features, **model_hparams, ds=ds)

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

        [self.num_classes_genus, self.num_classes_species] = ds.labelcount_per_level
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
        lr = self.optimizer_hparams["learning_rate"]
        weight_decay = self.optimizer_hparams["weight_decay"]

        params = filter(lambda p: p.requires_grad, self.model.parameters())
        optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

        return {"optimizer": optimizer}

    def training_step(self, batch):
        self.log("step", self.current_epoch)

        imgs, genus_labels, species_labels = batch
        logits = self(imgs)
        loss = self.loss_fn(logits, genus_labels, species_labels)
        preds = self.pred_fn(logits)

        if len(preds) == 2:
            [genus_preds, species_preds] = preds
            metrics = self.metric.process_train_batch(genus_preds, genus_labels, logits[1], species_preds, species_labels)
        else:
            species_preds = preds
            genus_preds = torch.zeros_like(genus_labels)
            metrics = self.metric.process_train_batch(genus_preds, genus_labels, logits, species_preds, species_labels)

        self.log_epoch(loss, "train_loss")
        self.log_epoch(metrics, "train_")

        lr = self.trainer.optimizers[0].param_groups[0]["lr"]

        self.log_epoch(lr, "lr")

        return loss

    def validation_step(self, batch):
        imgs, genus_labels, species_labels = batch
        logits = self(imgs)

        preds = self.pred_fn(logits)

        if len(preds) == 2:
            [genus_preds, species_preds] = preds
            metrics = self.metric.process_valid_batch(genus_preds, genus_labels, logits[1], species_preds, species_labels)
        else:
            species_preds = preds
            genus_preds = torch.zeros_like(genus_labels)
            metrics = self.metric.process_valid_batch(genus_preds, genus_labels, logits, species_preds, species_labels)

        self.log_epoch(metrics, "valid_")

    def on_validation_epoch_end(self):
        recall_species = self.metric.compute_recall(self.metric.valid_conf_m_species, self.metric.species_freq)
        recall_genus = self.metric.compute_recall(self.metric.valid_conf_m_genus, self.metric.genus_freq)

        self.log_epoch(recall_species, "valid_recall_species_")
        self.log_epoch(recall_genus, "valid_recall_genus_")
