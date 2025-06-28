import os
from pathlib import Path

import lightning as L
import torch
import torch.optim as optim

from src.experiments.clibdb_hyperbolic.models import HierarchicalPoincare
from src.shared.datasets import ClibdbDataset, DatasetVersion
from src.shared.prototypes import get_prototypes
from src.shared.torch.backbones import (ViTAEv2_B, load_for_transfer_learning,
                                        t2t_vit_t_14)
from src.shared.torch.hierarchical_metric import HierarchicalMetric

PRETRAINED_WEIGHTS_DIR = Path(os.getcwd()) / "pretrained_weights"
PRETRAINED_T2T_VITT_T14 = PRETRAINED_WEIGHTS_DIR / "81.7_T2T_ViTt_14.pth.tar"
PRETRAINED_VITAEv2 = PRETRAINED_WEIGHTS_DIR / "ViTAEv2-B.pth.tar"

BACKBONE_DICT = {
    "t2t_vit": (t2t_vit_t_14, PRETRAINED_T2T_VITT_T14, 384),
    "vitaev2": (ViTAEv2_B, PRETRAINED_VITAEv2, 1024),
}

torch.set_float32_matmul_precision("medium")


def create_model(model_hparams, ds):
    prototypes = get_prototypes(model_hparams["prototypes"], ds.version.value, 128)
    model_hparams["prototypes"] = prototypes

    if ds.version in [DatasetVersion.GBIF_GENUS_SPECIES_10K_EMBEDDINGS]:
        out_features = 384
        backbone = None
    else:
        # Load pretrained weights
        init, path_to_weights, out_features = BACKBONE_DICT[model_hparams["backbone_name"]]
        backbone = init(
            # drop_rate=0.2,
            # attn_drop_rate=0.1,
            # drop_path_rate=0.2,
        )
        load_for_transfer_learning(
            backbone,
            path_to_weights,
            use_ema=True,
            strict=False,
        )

    # Initialize model
    cls = HierarchicalPoincare
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



class LightningGBIF(L.LightningModule):
    def __init__(
        self,
        model_hparams,
        optimizer_name,
        optimizer_hparams,
        ds: ClibdbDataset,
    ):
        super().__init__()
        self.ds = ds
        self.optimizer_name = optimizer_name
        self.optimizer_hparams = optimizer_hparams

        self.model = create_model(model_hparams, ds)
        self.pred_fn = self.model.pred_fn
        self.loss_fn = self.model.loss_fn

        self.num_classes_per_level = ds.labelcount_per_level
        self.num_levels = len(self.num_classes_per_level)

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
        params = filter(lambda p: p.requires_grad, self.model.parameters())

        optim_dict = {
            "adam": optim.AdamW,
            "sgd": optim.SGD,
        }

        optimizer = optim_dict[self.optimizer_name](params, **self.optimizer_hparams)
        return [optimizer]

    def training_step(self, batch):
        self.log("step", self.current_epoch)

        imgs, *labels = batch  # variable-length labels

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
        imgs, *labels = batch
        logits = self(imgs)
        preds = self.pred_fn(logits)

        metrics = self.metric.process_batch(preds, labels, logits, split="valid")

        self.log_epoch(metrics, "valid_")

    def on_validation_epoch_end(self):
        recall_stats = self.metric.compute_recall()
        self.log_epoch(recall_stats, "valid_")

        lca = self.metric.compute_lca_stats()
        self.log_epoch(lca, "valid_")

    # def on_load_checkpoint(self, checkpoint):
    #     # Reconstruct list of level prototypes from registered buffers
    #     num_levels = len(self.model.level_sizes)
    #     self.model.level_prototypes = [
    #         getattr(self.model, f"level_{i}_prototypes") for i in range(num_levels)
    #     ]
