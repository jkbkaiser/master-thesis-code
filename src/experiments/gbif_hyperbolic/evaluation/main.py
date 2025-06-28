import ast
import os

import lightning as L
import mlflow
from dotenv import load_dotenv

from src.experiments.clibdb_hyperbolic.lighting import LightningGBIF
from src.shared.datasets import ClibdbDataset, Dataset, DatasetVersion

load_dotenv()

MLFLOW_SERVER = os.environ["MLFLOW_SERVER"]
CHECKPOINT_DIR = os.environ["CHECKPOINT_DIR"]

mlflow.set_tracking_uri(MLFLOW_SERVER)
client = mlflow.tracking.MlflowClient()

run_id = "f1c9fbb2f05c4bb6904354eaa773abab"
prototypes = "genus_species_poincare"

artifacts = mlflow.artifacts.list_artifacts(run_id=run_id)

for artifact in artifacts:
    print(artifact.path)

artifact_path = "epoch=0/epoch=0.ckpt"

local_ckpt_path = mlflow.artifacts.download_artifacts(
    run_id=run_id,
    artifact_path=artifact_path
)

print(local_ckpt_path)

import torch

ckpt = torch.load(local_ckpt_path, map_location="cpu")
print(ckpt["state_dict"].keys())

run = client.get_run(run_id)

params = run.data.params

ds = ClibdbDataset(DatasetVersion.CLIBDB)
ds.load(batch_size=16, use_torch=True)

# def get_model_architecture(model, ds: Dataset):
#     if model in ["hyperbolic-genus-species", "single"]:
#         return ds.labelcount_per_level
#     else:
#         return ds.labelcount_per_level[-1]
#
# architecture = get_model_architecture(params["model_name"], ds)

architecture = ds.labelcount_per_level

general_hparams = {
    "machine": "local",
    # "model_name": params["model_name"],
    "batch_size": int(params["batch_size"]),
    "dataset": params["dataset"],
    "epochs": int(params["epochs"]),
}

model_hparams = {
    "backbone_name": params["backbone_name"],
    "freeze_backbone": params["freeze_backbone"],
    "prototypes": prototypes,
    # "prototype_dim": int(params["prototype_dim"]),
    "architecture": ast.literal_eval(params["architecture"]),
    # "temp": float(params["temp"]),
}

model = LightningGBIF.load_from_checkpoint(
    checkpoint_path=local_ckpt_path,
    # model_name=params["model_name"],
    model_hparams=model_hparams,
    optimizer_name=None,
    optimizer_hparams=None,
    ds=ds,
)
trainer = L.Trainer(logger=True, enable_progress_bar=True)

trainer.validate(model, dataloaders=ds.test_dataloader)
