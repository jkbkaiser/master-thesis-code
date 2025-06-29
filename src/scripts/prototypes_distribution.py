import geoopt
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

from src.shared.datasets import DatasetVersion, get_hierarchy

labels = True

prototype_name = "genus_species_poincare"
dimensionality = 128

name_map = {
    "genus_species_poincare": "Poincaré",
    "entailment_cones": "Entailment Cones",
    "distortion": "Distortion"
}

ball = geoopt.PoincareBallExact(c=1.5)
hierarchy_levels = get_hierarchy(DatasetVersion.BIOSCAN.value)

prototype_names = ["genus_species_poincare", "entailment_cones", "distortion"]
colors = cm.tab10.colors

level_names = ["class", "order", "family", "subfamily", "genus", "species"]
level_sizes = [level.shape[0] for level in hierarchy_levels]
level_sizes.append(hierarchy_levels[-1].shape[1])

offsets = [0]
for size in level_sizes:
    offsets.append(offsets[-1] + size)


fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 9), sharex=True)

for idx, prototype_name in enumerate(prototype_names):
    prototypes = np.load(f"./prototypes/clibdb/{prototype_name}/{dimensionality}.npy")
    prototypes_t = torch.tensor(prototypes)
    per_level_norms = {}

    offsets = [0]
    for size in level_sizes:
        offsets.append(offsets[-1] + size)

    for i, name in enumerate(level_names):
        start, end = offsets[i], offsets[i + 1]
        level_protos = prototypes_t[start:end]
        dists = ball.dist0(level_protos).detach().cpu().numpy()
        per_level_norms[name] = dists

    ax = axes[idx]
    for i, (name, norms) in enumerate(per_level_norms.items()):
        ax.hist(norms, bins=30, alpha=0.5, label=name if idx == 0 else "", color=colors[i])

    ax.set_ylabel(name_map.get(prototype_name, prototype_name), fontsize=12)
    ax.grid(True)

handles, labels_ = axes[0].get_legend_handles_labels()
fig.legend(handles, labels_, loc="lower center", ncol=6, bbox_to_anchor=(0.5, 0), fontsize=12)
plt.tight_layout(rect=[0, 0.07, 1, 1])  # Make sure bottom (0.15) is large enough
sns.despine()
plt.savefig("./output_results/hyperbolic_norms_per_level_all.png", dpi=600, bbox_inches="tight")
plt.show()
