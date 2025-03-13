import os
from pathlib import Path

import lightning as L
import torch
import torch.optim as optim

from src.constants import DEVICE
from src.experiments.gbif_torch.models import HAC, MARG, MPLC, PLC
from src.shared.datasets import Dataset, DatasetType
from src.shared.torch.t2t_vit.t2t_vit import t2t_vit_t_14
from src.shared.torch.t2t_vit.utils import load_for_transfer_learning
from src.shared.torch.vitaev2.ViTAEv2 import ViTAEv2_B

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
        model_name,
        model_hparams,
        optimizer_name,
        optimizer_hparams,
        ds: Dataset,
        thresholds=[5, 10, 50, 100, 500, 1000],
    ):
        super().__init__()
        self.ds = ds
        self.optimizer_name = optimizer_name
        self.optimizer_hparams = optimizer_hparams
        self.model = create_model(model_name, model_hparams, ds)

        self.pred_fn = self.model.pred_fn
        self.loss_fn = self.model.loss_fn

        [self.num_classes_genus, self.num_classes_species] = get_num_classes(ds)
        print(self.num_classes_genus, self.num_classes_species)

        self.thresholds = thresholds

        self.valid_conf_m = torch.zeros(
            (self.num_classes_species, self.num_classes_species), dtype=torch.int64
        ).to(DEVICE)

    def compute_valid_conf_m(self, species_preds, species_labels):
        if self.ds.type == DatasetType.FLAT:
            species_preds = species_preds - self.ds.split
            species_labels = species_labels - self.ds.split

        self.valid_conf_m.index_add_(
            0,
            species_labels.view(-1),
            torch.eye(self.num_classes_species, device=DEVICE)[
                species_preds.view(-1)
            ].to(torch.int64),
        )

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
        self.log_epoch(loss, "train_loss")

        genus_preds, species_preds = self.pred_fn(logits)

        correct_genus = genus_preds == genus_labels
        acc_genus = correct_genus.float().mean().item()
        correct_species = species_preds == species_labels
        acc_species = correct_species.float().mean().item()
        correct_both = correct_species & correct_genus
        total_acc = correct_both.float().mean().item()

        metrics = {
            "accuracy_genus": acc_genus,
            "accuracy_species": acc_species,
            "accuracy_avg": (acc_genus + acc_species) / 2,
            "accuracy_total": total_acc,
        }

        self.log_epoch(metrics, "train_")
        return loss

    def validation_step(self, batch):
        imgs, genus_labels, species_labels = batch
        logits = self(imgs)

        genus_preds, species_preds = self.pred_fn(logits)

        correct_genus = genus_preds == genus_labels
        acc_genus = correct_genus.float().mean().item()
        correct_species = species_preds == species_labels
        acc_species = correct_species.float().mean().item()
        correct_both = correct_species & correct_genus
        total_acc = correct_both.float().mean().item()

        metrics = {
            "accuracy_genus": acc_genus,
            "accuracy_species": acc_species,
            "accuracy_avg": (acc_genus + acc_species) / 2,
            "accuracy_total": total_acc,
        }

        self.log_epoch(metrics, "valid_")

        self.compute_valid_conf_m(species_preds, species_labels)

    def on_validation_epoch_end(self):
        conf_m = self.valid_conf_m.float()
        tp = torch.diag(conf_m)
        fp = conf_m.sum(dim=0) - tp
        fn = conf_m.sum(dim=1) - tp

        precision_per_class = tp / (tp + fp + 1e-8)
        recall_per_class = tp / (tp + fn + 1e-8)
        f1_per_class = (
            2
            * precision_per_class
            * recall_per_class
            / (precision_per_class + recall_per_class + 1e-8)
        )

        support = conf_m.sum(dim=1)
        f1 = (f1_per_class * support).sum() / support.sum()

        appeared_classes = support > 0
        precision = precision_per_class[appeared_classes].mean()
        recall = recall_per_class[appeared_classes].mean()

        metrics = {
            "valid_precision_species": precision,
            "valid_recall_species": recall,
            "valid_f1_species": f1,
        }

        # for threshold_idx in range(self.freq.shape[0]):
        #     freq_mask = self.freq[threshold_idx]
        #     valid_freq_mask = freq_mask & appeared_classes
        #
        #     if valid_freq_mask.any():
        #         recall_low_freq = recall_per_class[valid_freq_mask].mean()
        #         precision_low_freq = precision_per_class[valid_freq_mask].mean()
        #         f1_low_freq = f1_per_class[valid_freq_mask].mean()
        #     else:
        #         recall_low_freq = torch.tensor(0.0, device=self.device)
        #         precision_low_freq = torch.tensor(0.0, device=self.device)
        #         f1_low_freq = torch.tensor(0.0, device=self.device)
        #
        #     metrics[
        #         f"valid_precision_species_less_freq_{self.thresholds[threshold_idx]}"
        #     ] = precision_low_freq
        #     metrics[
        #         f"valid_recall_species_less_freq_{self.thresholds[threshold_idx]}"
        #     ] = recall_low_freq
        #     metrics[f"valid_f1_species_less_freq_{self.thresholds[threshold_idx]}"] = (
        #         f1_low_freq
        #     )
        #
        self.log_dict(metrics)
        self.valid_conf_m.zero_()
