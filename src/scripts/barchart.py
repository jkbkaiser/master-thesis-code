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
    "86fab3a2574843c592a9e3e783452ca4",
    "423d8e5d629f4ef0acc5fdd2b6b1c013",
    "dd0caaa80d14416b8a3e094cfaad4916",
    "8115459ea38545378430c491c8de53dd"
]

frequencies = ["10", "50", "100", "all"]
metric_prefix = "valid_recall_species_support_weighted_recall"

records = []
for run_id in run_ids_to_compare:
    try:
        run = mlflow.get_run(run_id)
        run_name = run.data.tags.get("mlflow.runName", run_id)
        model_name = run.data.params.get("model_name", "unknown")

        for freq in frequencies:
            metric_key = f"{metric_prefix}_{freq}"
            value = run.data.metrics.get(metric_key)
            if value is not None:
                records.append({
                    "Run Name": run_name,
                    "Model Name": model_name.upper(),
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

ax.set_ylabel("Macro Recall")
ax.set_xlabel("Frequency Threshold")
plt.legend(title="Model", loc="upper left")
plt.tight_layout()
plt.savefig("baselines_barchart.png", dpi=600)
plt.show()

