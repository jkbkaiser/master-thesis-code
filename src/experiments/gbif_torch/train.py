import argparse

import lightning as L
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.loggers import MLFlowLogger

from src.experiments.gbif_torch.lighting import LightningGBIF
from src.shared.datasets.gbif.dataset import load_gbif_dataloader
from src.shared.datasets.gbif.taxonomy import (get_genus_to_species_mask,
                                               get_species_to_genus_map)

MLFLOW_SERVER = "http://localhost:5000"
ARTIFACT_LOCATION = "./mlartifacts"

MODEL_TO_HYPERPARAMS = {
    "hac": ("flat", 439),
    "plc": ("hier", [2883, 6777]),
    "mplc": ("hier", [2883, 6777]),
    "marg": ("hier", 6777),
}

# torch.Size([2883, 6777])


def parse_args():
    parser = argparse.ArgumentParser(
        prog="Training script for gbif datasets",
        description="Train various models using different configurations using gbif datasets",
    )
    parser.add_argument(
        "-o",
        "--optimizer",
        default="adam",
        required=False,
        type=str,
        choices=["adam", "sgd"],
    )
    parser.add_argument(
        "-lr", "--learning-rate", default=1e-4, required=False, type=float
    )
    parser.add_argument(
        "-wd", "--weight-decay", default=1e-2, required=False, type=float
    )
    parser.add_argument(
        "-m",
        "--model",
        default="hac",
        required=False,
        type=str,
        choices=["hac", "plc", "mplc", "marg"],
    )
    parser.add_argument(
        "-fb", "--freeze-backbone", default=True, required=False, type=bool
    )
    parser.add_argument(
        "-b",
        "--backbone",
        default="t2t_vit",
        required=False,
        type=str,
        choices=["t2t_vit", "vitaev2"],
    )
    parser.add_argument("-ee", "--eval-every", default=2, required=False, type=int)
    parser.add_argument("-ne", "--num-epochs", default=50, required=False, type=int)
    parser.add_argument("-sd", "--seed", default=42, required=False, type=int)
    parser.add_argument("-bs", "--batch-size", default=16, required=False, type=int)
    parser.add_argument(
        "-lfc",
        "--load-from-checkpoint",
        default=None,
        required=False,
        type=str,
    )
    parser.add_argument(
        "-name", "--experiment-name", default="gbif_torch", required=False, type=str
    )

    return parser.parse_args()


def run(args):
    ds, num_classes = MODEL_TO_HYPERPARAMS[args.model]

    train_loader = load_gbif_dataloader(
        "train", batch_size=args.batch_size, version=ds, use_torch=True
    )
    valid_loader = load_gbif_dataloader(
        "valid", batch_size=args.batch_size, version=ds, use_torch=True
    )

    # t = get_genus_to_species_mask()
    # t1 = get_species_to_genus_map()

    # print(t.shape)
    # # torch.Size([2883, 6777])
    # print(t1.shape)

    mlf_logger = MLFlowLogger(
        experiment_name=args.experiment_name,
        tracking_uri=MLFLOW_SERVER,
        artifact_location=ARTIFACT_LOCATION,
        log_model="all",
    )

    general_hparams = {
        "model_name": args.model,
        "optim_name": args.optimizer,
        "batch_size": args.batch_size,
        "dataset": ds,
        "load_from_checkpoint": args.load_from_checkpoint,
    }
    model_hparams = {
        "backbone_name": args.backbone,
        "num_classes": num_classes,
        "freeze_backbone": args.freeze_backbone,
    }

    optim_hparams = {"lr": args.learning_rate, "weight_decay": args.weight_decay}

    if args.load_from_checkpoint:
        model = LightningGBIF.load_from_checkpoint(
            args.load_from_checkpoint,
            model_name=args.model,
            model_hparams=model_hparams,
            optimizer_name=args.optimizer,
            optimizer_hparams=optim_hparams,
        )
    else:
        model = LightningGBIF(
            model_name=args.model,
            model_hparams=model_hparams,
            optimizer_name=args.optimizer,
            optimizer_hparams=optim_hparams,
            num_classes_genus=2883,
            num_classes_species=6777,
        )

    mlf_logger.log_hyperparams(general_hparams)
    mlf_logger.log_hyperparams(model_hparams)
    mlf_logger.log_hyperparams(optim_hparams)

    early_stop_callback = EarlyStopping(
        monitor="train_loss",
        min_delta=0.00,
        patience=3,
        verbose=False,
        mode="min",
    )

    trainer = L.Trainer(
        max_epochs=args.num_epochs,
        log_every_n_steps=args.eval_every,
        logger=mlf_logger,
        callbacks=[early_stop_callback],
        profiler="simple",
    )

    trainer.fit(model, train_loader, valid_loader)
    # trainer.test(model, test_loader)


if __name__ == "__main__":
    args = parse_args()
    run(args)
