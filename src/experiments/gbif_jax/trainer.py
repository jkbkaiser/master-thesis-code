import os
import warnings

import jax
import jax.numpy as jnp
import mlflow
from dotenv import load_dotenv
from flax import nnx
from tqdm import tqdm

load_dotenv()

warnings.filterwarnings(
    "ignore",
    category=RuntimeWarning,
    module="multiprocessing.popen_fork"
)

MLFLOW_SERVER = os.environ["MLFLOW_SERVER"]


class Trainer:
    def __init__(
        self,
        model,
        param_filter,
        optimizer,
        show_progress_bar = True,
    ):
        self.model = model
        self.param_filter = param_filter
        self.optimizer = optimizer
        self.show_pb = show_progress_bar

        mlflow.set_tracking_uri(MLFLOW_SERVER)
        self.mlflow_experiment_name = "jax-experiment"
        self.mlflow_run = None

        self.current_step = 0

    def start_mlflow_run(self, num_epochs):
        mlflow.set_experiment(self.mlflow_experiment_name)
        self.mlflow_run = mlflow.start_run()

        mlflow.log_param("num_epochs", num_epochs)
        mlflow.log_param("model_type", self.model.__class__.__name__)

    def train_step(self, epoch):
        @nnx.jit
        def train_iteration(model, optimizer, batch):
            imgs, genus_labels, species_labels = jax.device_put(batch)

            def loss_fn(model):
                logits = model(imgs)
                loss = model.loss(logits, genus_labels, species_labels)
                return loss, logits

            (loss, logits), grads = nnx.value_and_grad(loss_fn, wrt=self.param_filter, has_aux=True)(model)
            optimizer.update(grads)

            predictions = jnp.argmax(logits, axis=-1)
            acc = jnp.sum(predictions == genus_labels) / genus_labels.size

            return loss, acc

        dataloader = self.train_dataloader
        if self.show_pb:
            dataloader = tqdm(dataloader, desc=f"Epoch {epoch+1}", leave=False)

        avg_acc = 0

        for batch in dataloader:
            loss, acc = train_iteration(self.model, self.optimizer, batch)
            avg_acc += acc

            mlflow.log_metric("train_loss", loss, step=self.current_step)
            self.current_step += 1

        mlflow.log_metric("acc", avg_acc / len(dataloader), step=self.current_step)

        if self.show_pb:
            dataloader.close()

    def train(
        self,
        train_dataloader,
        valid_dataloader,
        num_epochs,
    ):
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader

        self.start_mlflow_run(num_epochs)

        for epoch in range(num_epochs):
            self.train_step(epoch)

        if self.mlflow_run:
            mlflow.end_run()
