import argparse
import logging
import os

import lightning as L
from dotenv import load_dotenv
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import MLFlowLogger

from src.experiments.clibdb_hyperbolic.lighting import LightningGBIF
from src.shared.datasets import ClibdbDataset, DatasetVersion
from src.shared.prototypes import PrototypeVersion

load_dotenv()
MLFLOW_SERVER = os.environ["MLFLOW_SERVER"]
CHECKPOINT_DIR = os.environ["CHECKPOINT_DIR"]

def get_model_architecture(ds: ClibdbDataset):
    return ds.labelcount_per_level

def parse_args():
    parser = argparse.ArgumentParser(
        prog="Training script for gbif datasets",
        description="Train various models using different configurations using gbif datasets",

    )

    parser.add_argument(
        "--experiment-name", default="clibdb_hyperbolic", required=False, type=str
    )

    parser.add_argument(
        "-d",
        "--dataset",
        required=True,
        type=DatasetVersion,
        choices=[v.value for v in DatasetVersion],
    )

    # parser.add_argument(
    #     "-m",
    #     "--model",
    #     default="hac",
    #     required=False,
    #     type=str,
    #     choices=["hac", "plc", "mplc", "marg"],
    # )

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
        "--freeze-backbone", action="store_true", help="Freeze backbone during training"
    )
    parser.add_argument(
        "--no-freeze-backbone", dest="freeze_backbone", action="store_false", help="Unfreeze backbone"
    )
    parser.set_defaults(freeze_backbone=True)
    parser.add_argument("--num-epochs", default=50, required=False, type=int)
    parser.add_argument("--batch-size", default=16, required=False, type=int)
    parser.add_argument("--eval-every", default=2, required=False, type=int)
    parser.add_argument("--seed", default=42, required=False, type=int)
    parser.add_argument("--reload", action="store_true", default=False, required=False)
    parser.add_argument("--mplc-warmup", default=10, required=False, type=int)

    parser.add_argument(
        "--prototypes",
        required=True,
        type=str,
        choices=[v.value for v in PrototypeVersion],
    )

    parser.add_argument(
        "--show-progress-bar",
        dest="progress_bar",
        action="store_true",
        help="Show progress bar"
    )


    return parser.parse_args()


def run(args):
    ds = ClibdbDataset(args.dataset)
    ds.load(batch_size=args.batch_size, use_torch=True, reload=args.reload)
    architecture = get_model_architecture(ds)

    mlf_logger = MLFlowLogger(
        experiment_name=args.experiment_name,
        tracking_uri=MLFLOW_SERVER,
        log_model=True,
    )

    general_hparams = {
        # "model_name": args.model,
        "optim_name": args.optimizer,
        "batch_size": args.batch_size,
        "dataset": args.dataset.value,
        "epochs": args.num_epochs,
    }

    model_hparams = {
        "backbone_name": args.backbone,
        "prototypes": args.prototypes,
        "architecture": architecture,
        "freeze_backbone": args.freeze_backbone,
    }

    # if args.model == "mplc":
    #     model_hparams.update({"warmup": args.mplc_warmup})

    optim_hparams = {"lr": args.learning_rate, "weight_decay": args.weight_decay}

    model = LightningGBIF(
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


if __name__ == "__main__":
    args = parse_args()
    run(args)
