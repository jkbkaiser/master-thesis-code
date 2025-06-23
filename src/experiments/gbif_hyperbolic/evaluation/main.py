import ast
import os

import mlflow
import torch
from dotenv import load_dotenv

from src.experiments.gbif_hyperbolic.lighting import LightningGBIF
from src.shared.datasets import Dataset, DatasetVersion

load_dotenv()

MLFLOW_SERVER = os.environ["MLFLOW_SERVER"]
CHECKPOINT_DIR = os.environ["CHECKPOINT_DIR"]

mlflow.set_tracking_uri(MLFLOW_SERVER)
client = mlflow.tracking.MlflowClient()

run_id = "97647cdb6c1643c69d920c9194c09b51"
artifact_path = "epoch=9/epoch=9.ckpt"

local_ckpt_path = mlflow.artifacts.download_artifacts(
    run_id=run_id,
    artifact_path=artifact_path
)

run = client.get_run(run_id)

params = run.data.params


ds = Dataset(DatasetVersion.GBIF_GENUS_SPECIES_100K)
ds.load(batch_size=16, use_torch=True)


def get_model_architecture(model, ds: Dataset):
    if model in ["hyperbolic-genus-species", "single"]:
        return ds.labelcount_per_level
    else:
        return ds.labelcount_per_level[-1]

architecture = get_model_architecture(params["model_name"], ds)

general_hparams = {
    "machine": "local",
    "model_name": params["model_name"],
    "batch_size": int(params["batch_size"]),
    "dataset": params["dataset"],
    "epochs": int(params["epochs"]),
}

model_hparams = {
    "backbone_name": params["backbone_name"],
    "freeze_backbone": params["freeze_backbone"],
    "prototypes": params["prototypes"],
    "prototype_dim": int(params["prototype_dim"]),
    "architecture": ast.literal_eval(params["architecture"]),
    "temp": float(params["temp"]),
}

model = LightningGBIF.load_from_checkpoint(
    checkpoint_path=local_ckpt_path,
    model_name=params["model_name"],
    model_hparams=model_hparams,
    optimizer_name=None,
    optimizer_hparams=None,
    ds=ds,
).to(torch.device("cuda"))

[imgs, genus, species] = next(iter(ds.train_dataloader))

imgs = imgs.to(model.device)

out = model(imgs)

print(out)

# Initialize trainer (no need to re-enable logging etc.)
# trainer = L.Trainer(logger=False, enable_progress_bar=True)
#
# # Evaluate on validation/test set
# trainer.test(model, dataloaders=test_loader)  # or `trainer.validate(...)` if you're using a validation set
