import matplotlib.pyplot as plt
import numpy as np
import torch

# Load saved prototypes
prototypes = np.load("./prototypes/gbif_genus_species_10k/prototypes-2-gbif_genus_species_10k.npy")

# Ensure normalization
# prototypes = prototypes / np.linalg.norm(prototypes, axis=1, keepdims=True)

# Plot
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim(-1.1, 1.1)
ax.set_ylim(-1.1, 1.1)
ax.set_aspect("equal")
ax.axhline(0, color='gray', linewidth=0.5)
ax.axvline(0, color='gray', linewidth=0.5)
circle = plt.Circle((0, 0), 1, color='black', fill=False, linestyle='dashed')
ax.add_patch(circle)

# Scatter plot of prototypes
ax.scatter(prototypes[:, 0], prototypes[:, 1], color='red')

# # Annotate prototypes with indices
# for i, (x, y) in enumerate(prototypes):
#     ax.text(x, y, str(i), fontsize=8, ha='right', color='blue')

plt.title("Hyperspherical Prototypes in 2D")
plt.show()
