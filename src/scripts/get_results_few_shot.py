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
        metric_keys = [
            "ap",
            "acc",
        ]

    client = mlflow.tracking.MlflowClient()
    result = {}

    for key in metric_keys:
        # Get full metric history (all steps)
        history = client.get_metric_history(run_id, key)
        values = [m.value for m in history]

        if values:
            mean = np.mean(values)
            std = np.std(values)
            result[key] = {"mean": mean, "std": std}
        else:
            result[key] = {"mean": None, "std": None}

    return result


if __name__ == "__main__":
    run_id = input("Enter the MLflow run ID: ").strip()
    metrics = get_metrics_for_run(run_id)
    print("Metrics for run", run_id)
    for key, value in metrics.items():
        print(f"{key}: {value}")
