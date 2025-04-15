import argparse

import jax
from flax import nnx

from src.experiments.gbif_jax.trainer import Trainer
from src.experiments.gbif_jax.util import (MODEL_DICT, load_model,
                                           load_optimizer)
from src.shared.datasets import Dataset, DatasetVersion


def parse_args():
    parser = argparse.ArgumentParser(
        prog="Training script for models implemented using jax"
    )
    parser.add_argument(
        "--experiment-name",
        default="gbif_jax",
        dest="experiment_name",
        required=False,
        type=str
    )
    parser.add_argument(
        "--model",
        default="baseline",
        required=False,
        type=str,
        choices=[m for m in MODEL_DICT.keys()],
    )
    parser.add_argument(
        "--dataset",
        default=DatasetVersion.GBIF_GENUS_SPECIES_10K,
        required=False,
        type=DatasetVersion,
        choices=[v.value for v in DatasetVersion],
    )
    parser.add_argument(
        "--learning-rate",
        dest="learning_rate",
        default=1e-4,
        required=False,
        type=float
    )
    parser.add_argument(
        "--batch-size",
        dest="batch_size",
        default=16,
        required=False,
        type=int
    )
    return parser.parse_args()


def run(args):
    rng = jax.random.PRNGKey(0)

    rngs = nnx.Rngs(
        params=jax.random.split(rng, 1)[0],
        dropout=jax.random.split(rng, 2)[1],
        default=jax.random.split(rng, 3)[2],
    )

    ds = Dataset(args.dataset)
    ds.load(batch_size=args.batch_size, use_torch=False)

    model = load_model(args.model, rngs=rngs)
    # param_filter = nnx.All(nnx.Param, nnx.PathContains('head'))
    param_filter = None

    optimizer = load_optimizer(model, param_filter)

    trainer = Trainer(
        model=model,
        param_filter=param_filter,
        optimizer=optimizer,
    )

    trainer.train(
        train_dataloader=ds.train_dataloader,
        valid_dataloader=ds.valid_dataloader,
        num_epochs=5,
    )


if __name__ == "__main__":
    args = parse_args()
    run(args)
