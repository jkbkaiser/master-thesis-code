import argparse
import logging
import os

import lightning as L
from dotenv import load_dotenv
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import MLFlowLogger

from src.experiments.gbif_hyperbolic.lighting import MODEL_DICT, LightningGBIF
from src.shared.datasets import DatasetVersion
from src.shared.datasets.gbif import Dataset
from src.shared.prototypes import PrototypeVersion

load_dotenv()
MLFLOW_SERVER = os.environ["MLFLOW_SERVER"]
CHECKPOINT_DIR = os.environ["CHECKPOINT_DIR"]


def get_model_architecture(model, ds: Dataset):
    if model in ["hyperbolic-genus-species", "single", "euclidean"]:
        return ds.labelcount_per_level
    else:
        return ds.labelcount_per_level[-1]


def run(args):
    ds = Dataset(args.dataset)
    ds.load(batch_size=args.batch_size, use_torch=True)
    architecture = get_model_architecture(args.model, ds)

    mlf_logger = MLFlowLogger(
        experiment_name=args.experiment_name,
        tracking_uri=MLFLOW_SERVER,
        log_model=True,
    )

    general_hparams = {
        "machine": args.machine,
        "model_name": args.model,
        "batch_size": args.batch_size,
        "dataset": args.dataset.value,
        "epochs": args.num_epochs,
    }

    model_hparams = {
        "backbone_name": args.backbone,
        "freeze_backbone": args.freeze_backbone,
        "prototypes": args.prototypes,
        "prototype_dim": args.prototype_dim,
        "architecture": architecture,
        "temp": args.temp,
    }

    if not args.freeze_backbone:
        model_hparams.update({
            "freeze_epochs": args.freeze_epochs
        })

    optim_hparams = {
        "learning_rate": args.learning_rate,
        "backbone_learning_rate": args.backbone_learning_rate,
        "weight_decay": args.weight_decay,
    }

    if args.optimizer == "sgd":
        optim_hparams.update({
            "momentum": args.momentum
        })

    mlf_logger.log_hyperparams(general_hparams)
    mlf_logger.log_hyperparams(model_hparams)
    mlf_logger.log_hyperparams(optim_hparams)


    model = LightningGBIF(
        model_name=args.model,
        model_hparams=model_hparams,
        optimizer_name=args.optimizer,
        optimizer_hparams=optim_hparams,
        ds=ds,
    )

    early_stop_callback = EarlyStopping(
        monitor="train_loss",
        min_delta=0.00,
        patience=3,
        verbose=False,
        mode="min",
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=f'{CHECKPOINT_DIR}/{mlf_logger.experiment_id}/{mlf_logger.run_id}',
        filename='{epoch}'
    )

    trainer = L.Trainer(
        max_epochs=args.num_epochs,
        log_every_n_steps=args.eval_every,
        logger=mlf_logger,
        callbacks=[early_stop_callback, checkpoint_callback],
        profiler="simple",
        enable_progress_bar=args.progress_bar,
    )

    logging.getLogger("lightning.pytorch").setLevel(logging.FATAL)

    print(f"Training for {args.num_epochs} epochs")

    trainer.fit(model, ds.train_dataloader, ds.valid_dataloader)


def parse_args():
    parser = argparse.ArgumentParser(
        prog="Hyperbolic training",
        description="Training script for training a hyperbolical and hyperspherical models with the gbif dataset",
    )

    parser.add_argument(
        "--experiment-name", default="gbif_hyperbolic", required=False, type=str
    )
    parser.add_argument(
        "-m",
        "--model",
        default="hyperspherical",
        required=False,
        type=str,
        choices=MODEL_DICT.keys(),
    )
    parser.add_argument(
        "--backbone",
        default="t2t_vit",
        required=False,
        type=str,
        choices=["t2t_vit", "vitaev2"],
    )
    parser.add_argument(
        "-d",
        "--dataset",
        required=True,
        type=DatasetVersion,
        choices=[v.value for v in DatasetVersion],
    )
    parser.add_argument("--batch-size", default=16, required=False, type=int)
    parser.add_argument('--learning-rate', default=0.05, type=float)
    parser.add_argument('--backbone-learning-rate', default=0.0001, type=float)
    parser.add_argument('--num-epochs', default=25, type=int)
    parser.add_argument("--eval-every", default=2, required=False, type=int)
    parser.add_argument(
        "--weight-decay", default=1e-2, required=False, type=float
    )
    parser.add_argument(
        "--momentum", default=0.9, required=False, type=float
    )
    parser.add_argument(
        "--prototypes",
        default=PrototypeVersion.HYPERSPHERE_UNIFORM.value,
        required=False,
        type=str,
        choices=[v.value for v in PrototypeVersion],
    )
    parser.add_argument(
        "--prototype_dim",
        default=128,
        required=False,
        type=int,
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
        "--freeze-backbone", action="store_true", help="Freeze backbone during training"
    )
    parser.add_argument(
        "--freeze-epochs", help="Unfreeze backbone during training", type=int, default=10, required=False
    )
    parser.add_argument(
        "--temp",
        type=float,
        default=0.07,
        help="Temp applied to the logits"
    )
    parser.add_argument(
        "--machine",
        type=str,
        default="local",
        help="Machine identifier (e.g., 'local', 'snellius', etc.)"
    )
    parser.add_argument(
        "--show-progress-bar",
        dest="progress_bar",
        action="store_true",
        help="Show progress bar"
    )

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    run(args)
