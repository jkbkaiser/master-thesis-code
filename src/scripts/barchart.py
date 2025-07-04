import os

import matplotlib.pyplot as plt
import mlflow
import pandas as pd
import seaborn as sns
from dotenv import load_dotenv

load_dotenv()
MLFLOW_SERVER = os.environ["MLFLOW_SERVER"]
mlflow.set_tracking_uri(MLFLOW_SERVER)

experiment_name = "gbif_hyperbolic"
experiment = mlflow.get_experiment_by_name(experiment_name)
if experiment is None:
    raise ValueError(f"Experiment '{experiment_name}' not found.")

run_ids_to_compare = [
    "e8ef4cc6785547d88c2cc04ec6470e0d",
    "3be829d7736f4825bf4660474078d801",
    "f973eec0a8a94e998ff414fd2d5f7cd6",
    "3e3fd198ae4b457ea329d0931961a671",
    "9bd0a8062061402fb37c632c34ff5cd2",
]

names_map = {
    "genus_species_poincare": "Poincaré",
    "entailment_cones": "Entailment Cones",
    "avg_genus": "Aggregated",
    "plc": "PLC",
    "marg": "MARG",
}

frequencies = ["10", "50", "100", "all"]
metric_prefix = "valid_recall_species_support_weighted_recall"

records = []
for run_id in run_ids_to_compare:
    try:
        run = mlflow.get_run(run_id)
        print(run)
        run_name = run.data.tags.get("mlflow.runName", run_id)
        model_name = run.data.params.get("prototypes", run.data.params.get("model_name", "unknown"))

        for freq in frequencies:
            metric_key = f"{metric_prefix}_{freq}"
            value = run.data.metrics.get(metric_key)
            if value is not None:
                records.append({
                    "Run Name": run_name,
                    "Model Name": names_map.get(model_name, model_name),
                    "Frequency": freq,
                    "Recall": value
                })
    except Exception as e:
        print(f"Failed to retrieve run {run_id}: {e}")

df = pd.DataFrame(records)

sns.set(style="whitegrid", font_scale=1.1)
plt.figure(figsize=(10, 6))

ax = sns.barplot(
    data=df,
    x="Frequency",
    y="Recall",
    hue="Model Name",
    palette="deep",
    edgecolor="black"
)
sns.despine(top=True, right=False, left=True, bottom=False)

ax.set_ylabel("Weighted Recall")
ax.set_xlabel("Frequency Threshold")
plt.legend(title="", loc="upper left")
plt.tight_layout()
plt.savefig("baselines_barchart.png", dpi=600)
plt.show()
