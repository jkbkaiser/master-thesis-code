import os

import mlflow
import numpy as np
from dotenv import load_dotenv

load_dotenv()

MLFLOW_SERVER = os.environ["MLFLOW_SERVER"]
CHECKPOINT_DIR = os.environ["CHECKPOINT_DIR"]

mlflow.set_tracking_uri(MLFLOW_SERVER)
mlflow.set_experiment("few-shot")

def get_metrics_for_run(run_id, metric_keys=None):
    if metric_keys is None:
        metric_keys = [
            "acc",
            "ap",
        ]

    client = mlflow.tracking.MlflowClient()
    result = {}

    for key in metric_keys:
        history = client.get_metric_history(run_id, key)
        if not history:
            result[key] = None
            continue

        values = np.array([m.value for m in history])
        final = values[-1]
        avg = values.mean()
        std = values.std()

        result[key] = {
            "final": final,
            "avg": avg,
            "std": std,
            "n": len(values),
        }

    return result

if __name__ == "__main__":
    run_id = input("Enter the MLflow run ID: ").strip()
    metrics = get_metrics_for_run(run_id)

    print(f"\n📊 Metrics for run {run_id}\n")
    for key, stats in metrics.items():
        if stats is None:
            print(f"{key}: ❌ Not logged.")
        else:
            print(f"{key}:")
            print(f"  Final: {stats['final']:.4f}")
            print(f"  Mean:  {stats['avg']:.4f}")
            print(f"  Std:   {stats['std']:.4f}")
            print(f"  N:     {stats['n']}")

