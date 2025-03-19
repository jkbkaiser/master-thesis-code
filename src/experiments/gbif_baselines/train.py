import argparse
import os

import lightning as L
from dotenv import load_dotenv
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import MLFlowLogger

from src.experiments.gbif_baselines.lighting import LightningGBIF
from src.shared.datasets import Dataset, DatasetType, DatasetVersion

load_dotenv()
MLFLOW_SERVER = os.environ["MLFLOW_SERVER"]
CHECKPOINT_DIR = os.environ["CHECKPOINT_DIR"]

def get_model_architecture(model, ds: Dataset):
    if model == "plc" or model == "mplc" or model == "hac":
        return ds.labelcount_per_level
    if model == "marg":
        return ds.labelcount_per_level[-1]

def parse_args():
    parser = argparse.ArgumentParser(
        prog="Training script for gbif datasets",
        description="Train various models using different configurations using gbif datasets",

    )

    parser.add_argument(
        "--experiment-name", default="gbif_baselines", required=False, type=str
    )

    parser.add_argument(
        "-d",
        "--dataset",
        required=True,
        type=DatasetVersion,
        choices=[v.value for v in DatasetVersion],
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
        "--backbone",
        default="t2t_vit",
        required=False,
        type=str,
        choices=["t2t_vit", "vitaev2"],
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
        "--weight-decay", default=1e-2, required=False, type=float
    )
    parser.add_argument(
        "--freeze-backbone", default=True, required=False, type=bool
    )
    parser.add_argument("--num-epochs", default=50, required=False, type=int)
    parser.add_argument("--batch-size", default=16, required=False, type=int)
    parser.add_argument("--eval-every", default=2, required=False, type=int)
    parser.add_argument("--seed", default=42, required=False, type=int)
    parser.add_argument("--reload", default=False, required=False, type=bool)

    return parser.parse_args()


def run(args):
    ds = Dataset(args.dataset)
    ds.load(batch_size=args.batch_size, use_torch=True, reload=args.reload)
    architecture = get_model_architecture(args.model, ds)

    mlf_logger = MLFlowLogger(
        experiment_name=args.experiment_name,
        tracking_uri=MLFLOW_SERVER,
        artifact_location="file:./logs/mlruns",
        log_model=True,
    )

    general_hparams = {
        "model_name": args.model,
        "optim_name": args.optimizer,
        "batch_size": args.batch_size,
        "dataset": args.dataset.value,
    }

    model_hparams = {
        "backbone_name": args.backbone,
        "architecture": architecture,
        "freeze_backbone": args.freeze_backbone,
    }

    if ds.type == DatasetType.FLAT:
        model_hparams.update({"split": ds.split})

    optim_hparams = {"lr": args.learning_rate, "weight_decay": args.weight_decay}

    model = LightningGBIF(
        model_name=args.model,
        model_hparams=model_hparams,
        optimizer_name=args.optimizer,
        optimizer_hparams=optim_hparams,
        ds=ds,
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

    checkpoint_callback = ModelCheckpoint(
        dirpath=f'{CHECKPOINT_DIR}/{mlf_logger.experiment_id}',
        filename='{epoch}'
    )

    trainer = L.Trainer(
        max_epochs=args.num_epochs,
        log_every_n_steps=args.eval_every,
        logger=mlf_logger,
        callbacks=[early_stop_callback, checkpoint_callback],
        profiler="simple",
    )

    trainer.fit(model, ds.train_dataloader, ds.valid_dataloader)


if __name__ == "__main__":
    args = parse_args()
    run(args)
