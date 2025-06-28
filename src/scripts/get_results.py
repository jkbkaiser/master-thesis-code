import argparse
import os
from pathlib import Path

import mlflow
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dotenv import load_dotenv

from src.constants import DEVICE
from src.shared.datasets import ClibdbDataset, Dataset, DatasetVersion

load_dotenv()

MLFLOW_SERVER = os.environ["MLFLOW_SERVER"]
CHECKPOINT_DIR = os.environ["CHECKPOINT_DIR"]

mlflow.set_tracking_uri(MLFLOW_SERVER)
mlflow.set_experiment("gbif_hyperbolic")


def get_metrics_for_run(run_id, metric_keys=None):
    if metric_keys is None:
        metric_keys = ["valid_accuracy_species", "valid_accuracy_genus", "valid_top5_accuracy", "valid_recall_species_support_weighted_recall_all", "valid_recall_species_macro_recall_all", "valid_recall_species_micro_recall_all"]

    client = mlflow.tracking.MlflowClient()
    run = client.get_run(run_id)

    metrics = run.data.metrics
    result = {}

    for key in metric_keys:
        if key in metrics:
            result[key] = metrics[key]
        else:
            result[key] = None  # or raise an error/warning if preferred

    return result


if __name__ == "__main__":
    run_id = input("Enter the MLflow run ID: ").strip()
    metrics = get_metrics_for_run(run_id)
    print("Metrics for run", run_id)
    for key, value in metrics.items():
        print(f"{key}: {value}")
