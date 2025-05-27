import os

import matplotlib.pyplot as plt
import mlflow
import pandas as pd
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
MLFLOW_SERVER = os.environ["MLFLOW_SERVER"]
mlflow.set_tracking_uri(MLFLOW_SERVER)

# Set experiment name
experiment_name = "gbif_hyperbolic"
experiment = mlflow.get_experiment_by_name(experiment_name)
if experiment is None:
    raise ValueError(f"Experiment '{experiment_name}' not found.")

# List the run names or IDs you want to compare
# You can use run IDs, or if you prefer to filter by tag/name, adjust the logic below
run_ids_to_compare = [
    "86fab3a2574843c592a9e3e783452ca4",
    "423d8e5d629f4ef0acc5fdd2b6b1c013",
    "dd0caaa80d14416b8a3e094cfaad4916",
    "8115459ea38545378430c491c8de53dd"
]

# Frequencies to evaluate
frequencies = ["10", "50", "100", "all"]
metric_prefix = "valid_recall_species_support_weighted_recall"

# Collect metrics for each run
results = []

for run_id in run_ids_to_compare:
    try:
        run = mlflow.get_run(run_id)
        row = {
            "Run ID": run_id,
            "Run Name": run.data.tags.get("mlflow.runName", run_id)  # fallback to ID if no name
        }
        for freq in frequencies:
            metric_key = f"{metric_prefix}_{freq}"
            value = run.data.metrics.get(metric_key, None)
            row[freq] = value
        results.append(row)
    except Exception as e:
        print(f"Failed to retrieve run {run_id}: {e}")

# Convert to DataFrame
df = pd.DataFrame(results)

# Plotting
fig, ax = plt.subplots(figsize=(10, 6))
bar_width = 0.2
x = range(len(frequencies))

for i, row in df.iterrows():
    values = [row[freq] for freq in frequencies]
    ax.bar(
        [p + i * bar_width for p in x],
        values,
        width=bar_width,
        label=row["Run Name"]
    )

# Axis formatting
ax.set_xlabel("Frequency Threshold")
ax.set_ylabel("Macro Recall")
ax.set_title("Macro Recall per Frequency Threshold by Run")
ax.set_xticks([p + (bar_width * (len(df) - 1)) / 2 for p in x])
ax.set_xticklabels(frequencies)
ax.legend()

plt.tight_layout()
plt.show()

