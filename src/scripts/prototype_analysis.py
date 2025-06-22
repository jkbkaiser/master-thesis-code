import os
import uuid

import geoopt
import matplotlib.pyplot as plt
import numpy as np
import requests
import torch
from sklearn.manifold import TSNE

from src.constants import CACHE_DIR, GOOGLE_BUCKET_URL
from src.shared.datasets import DatasetVersion


# Function to get hierarchy
def get_hierarchy(dataset_version, reload: bool = False):
    path = CACHE_DIR / f"{dataset_version}/hierarchy.npz"
    if reload or not path.is_file():
        url = f"{GOOGLE_BUCKET_URL}/{dataset_version}/hierarchy.npz?id={uuid.uuid4()}"
        response = requests.get(url)

        if response.status_code != 200:
            raise Exception(f"Could not retrieve metadata for {dataset_version}")

        with open(path, "wb") as f:
            f.write(response.content)

    hierarchy = np.load(path)["data"]
    return hierarchy.squeeze()


# Define directory for saving results and images
prototype_name = "entailment_cones"  # Example prototype name, adjust as needed
dimensionality = 128  # Example dimensionality, adjust as needed

# Create directory named after the prototype and dimensionality
output_dir = f'./output_results/{prototype_name}_{dimensionality}'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Prepare output file for storing results
results_file = os.path.join(output_dir, "distances_results.txt")
with open(results_file, "w") as file:
    file.write(f"Statistics {prototype_name} {dimensionality}:\n")

# Load data
hierarchy = get_hierarchy(DatasetVersion.GBIF_GENUS_SPECIES_100K.value)
prototypes = np.load(f"./prototypes/gbif_genus_species_100k/{prototype_name}/{dimensionality}.npy")
# prototypes = prototypes * 0.95 / 3
prototypes_t = torch.tensor(prototypes)

print("prototypes_shape", prototypes.shape)

# Separate species and genus prototypes based on the hierarchy
num_genus = hierarchy.shape[0]
num_species = hierarchy.shape[1]

print(num_genus + num_species)

species_prototypes = prototypes_t[num_genus:]  # Assuming species prototypes start after genus
genus_prototypes = prototypes_t[:num_genus]    # Genus prototypes are at the beginning

# Calculate and save norm statistics for species and genus separately
species_norms = species_prototypes.norm(dim=1)
genus_norms = genus_prototypes.norm(dim=1)

mean_species_norm = species_norms.mean().item()
max_species_norm = species_norms.max().item()
min_species_norm = species_norms.min().item()

mean_genus_norm = genus_norms.mean().item()
max_genus_norm = genus_norms.max().item()
min_genus_norm = genus_norms.min().item()

print(f"Species Norms - Mean: {mean_species_norm}, Max: {max_species_norm}, Min: {min_species_norm}")
print(f"Genus Norms - Mean: {mean_genus_norm}, Max: {max_genus_norm}, Min: {min_genus_norm}")

# Save the species and genus norm statistics in the result file
with open(results_file, "a") as file:
    file.write(f"Species Norms:\n")
    file.write(f"Mean Norm: {mean_species_norm:.4f}\n")
    file.write(f"Max Norm: {max_species_norm:.4f}\n")
    file.write(f"Min Norm: {min_species_norm:.4f}\n")

    file.write(f"Genus Norms:\n")
    file.write(f"Mean Norm: {mean_genus_norm:.4f}\n")
    file.write(f"Max Norm: {max_genus_norm:.4f}\n")
    file.write(f"Min Norm: {min_genus_norm:.4f}\n")

# Plot histograms of norms for species and genus separately
plt.hist(species_norms.numpy(), bins=20, color='skyblue', edgecolor='black')
plt.title("Species Norms Histogram")
plt.xlabel("Norm")
plt.ylabel("Frequency")
plt.savefig(os.path.join(output_dir, "species_norms_histogram.png"))
plt.close()

plt.hist(genus_norms.numpy(), bins=20, color='orange', edgecolor='black')
plt.title("Genus Norms Histogram")
plt.xlabel("Norm")
plt.ylabel("Frequency")
plt.savefig(os.path.join(output_dir, "genus_norms_histogram.png"))
plt.close()

# Continue with the rest of the original code for distance calculations and statistics

# Create GeoOpt Poincaré Ball instance
c = 3
ball = geoopt.PoincareBallExact(c=c)

# Initialize lists to store distances
within_genus_distances_euclidean = []
within_genus_distances_poincare = []
cross_genus_distances_euclidean = []
cross_genus_distances_poincare = []
species_to_genus_distances_euclidean = []  # For storing species-to-genus prototype distances (Euclidean)
species_to_genus_distances_poincare = []  # For storing species-to-genus prototype distances (Poincaré)

# Compute the t-SNE projection for visualization
proj = TSNE(n_components=2).fit_transform(prototypes)
proj -= proj.min(axis=0)
proj /= proj.max(axis=0)
proj = proj * 2 - 1

num_genus = hierarchy.shape[0]
num_species = hierarchy.shape[1]
t = hierarchy.sum(axis=1)
chosen_genus_indices = (-t).argsort()[:10]

# Scatter plot for genus and species prototypes
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim(-1.1, 1.1)
ax.set_ylim(-1.1, 1.1)
ax.set_aspect("equal")
ax.axhline(0, color='gray', linewidth=0.5)
ax.axvline(0, color='gray', linewidth=0.5)
circle = plt.Circle((0, 0), 1, color='black', fill=False, linestyle='dashed')
ax.add_patch(circle)

colors = plt.cm.get_cmap("tab10", 10)

for idx, genus_idx in enumerate(chosen_genus_indices):
    genus_proto = proj[genus_idx]

    species_indices = np.where(hierarchy[genus_idx] == 1)[0]
    for species_rel_idx in species_indices:
        species_proto = proj[2883 + species_rel_idx]
        ax.scatter(*species_proto, color=colors(idx), alpha=0.5, s=30, zorder=2)

    ax.scatter(*genus_proto, color=colors(idx), label=f"Genus {genus_idx}", edgecolor='black')

plt.legend()
plt.title("t-SNE of Prototypes")
plt.savefig(os.path.join(output_dir, "tsne_scatter_plot_top_10.png"))
plt.close()

fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim(-1.1, 1.1)
ax.set_ylim(-1.1, 1.1)
ax.set_aspect("equal")
ax.axhline(0, color='gray', linewidth=0.5)
ax.axvline(0, color='gray', linewidth=0.5)
circle = plt.Circle((0, 0), 1, color='black', fill=False, linestyle='dashed')
ax.add_patch(circle)

# Plot species prototypes first
for genus_idx in range(num_genus):
    species_indices = np.where(hierarchy[genus_idx] == 1)[0]
    species_proj = proj[num_genus + species_indices]
    ax.scatter(species_proj[:, 0], species_proj[:, 1], 
               color="blue", alpha=0.4, s=10, label=f"Genus {genus_idx}" if genus_idx < 10 else None)

# Plot genus prototypes last (on top)
genus_proj = proj[:num_genus]
ax.scatter(genus_proj[:, 0], genus_proj[:, 1], 
           color="red", edgecolors='black', s=10, marker='o')

ax.set_title("t-SNE of Genus and Species Prototypes")
plt.savefig(os.path.join(output_dir, "tsne_scatter_plot_genus_vs_species.png"))
plt.close()

# Create GeoOpt Poincaré Ball instance
c = 3
ball = geoopt.PoincareBallExact(c=c)

# Initialize lists to store distances
within_genus_distances_euclidean = []
within_genus_distances_poincare = []
cross_genus_distances_euclidean = []
cross_genus_distances_poincare = []
species_to_genus_distances_euclidean = []  # For storing species-to-genus prototype distances (Euclidean)
species_to_genus_distances_poincare = []  # For storing species-to-genus prototype distances (Poincaré)

# Calculate the distances
for genus_idx in chosen_genus_indices:
    species_indices = np.where(hierarchy[genus_idx] == 1)[0]  # Get species in the genus
    genus_proto = prototypes[genus_idx]

    # Calculate distances between species within the same genus
    for i in range(len(species_indices)):
        species_i_idx = species_indices[i]
        species_proto_i = prototypes[num_genus + species_i_idx]

        euclidean_distance_to_genus = np.linalg.norm(species_proto_i - genus_proto)
        species_to_genus_distances_euclidean.append(euclidean_distance_to_genus)

        species_proto_poincare = torch.tensor(species_proto_i).unsqueeze(0)  # Add batch dimension
        genus_proto_poincare = torch.tensor(genus_proto).unsqueeze(0)  # Add batch dimension

        poincare_distance_to_genus = ball.dist(species_proto_poincare, genus_proto_poincare).item()
        species_to_genus_distances_poincare.append(poincare_distance_to_genus)


        for j in range(i + 1, len(species_indices)):
            species_j_idx = species_indices[j]

            species_proto_j = prototypes[num_genus + species_j_idx]

            # Euclidean distance within genus
            euclidean_distance = np.linalg.norm(species_proto_i - species_proto_j)
            within_genus_distances_euclidean.append(euclidean_distance)

            # Poincaré distance within genus
            species_proto_i_poincare = torch.tensor(species_proto_i).unsqueeze(0)  # Add batch dimension
            species_proto_j_poincare = torch.tensor(species_proto_j).unsqueeze(0)  # Add batch dimension

            poincare_distance = ball.dist(species_proto_i_poincare, species_proto_j_poincare).item()
            within_genus_distances_poincare.append(poincare_distance)

    # Calculate distances from species to all other genus prototypes (excluding same genus)
    for species_idx in species_indices:
        species_proto = prototypes[num_genus + species_idx]

        # Compute Poincaré distance from species to the genus prototype

        for other_genus_idx in range(num_genus):
            if other_genus_idx == genus_idx:
                continue  # Skip the same genus

            genus_proto = prototypes[other_genus_idx]

            # Euclidean distance from species to another genus prototype
            euclidean_distance = np.linalg.norm(species_proto - genus_proto)
            cross_genus_distances_euclidean.append(euclidean_distance)

            # Poincaré distance from species to another genus prototype
            species_proto_poincare = torch.tensor(species_proto).unsqueeze(0)  # Add batch dimension
            genus_proto_poincare = torch.tensor(genus_proto).unsqueeze(0)  # Add batch dimension

            poincare_distance = ball.dist(species_proto_poincare, genus_proto_poincare).item()
            cross_genus_distances_poincare.append(poincare_distance)

# Compute statistics for within-genus distances
avg_within_genus_euclidean = np.mean(within_genus_distances_euclidean)
min_within_genus_euclidean = np.min(within_genus_distances_euclidean)
max_within_genus_euclidean = np.max(within_genus_distances_euclidean)

avg_within_genus_poincare = np.mean(within_genus_distances_poincare)
min_within_genus_poincare = np.min(within_genus_distances_poincare)
max_within_genus_poincare = np.max(within_genus_distances_poincare)

# Compute statistics for cross-genus distances
avg_cross_genus_euclidean = np.mean(cross_genus_distances_euclidean)
min_cross_genus_euclidean = np.min(cross_genus_distances_euclidean)
max_cross_genus_euclidean = np.max(cross_genus_distances_euclidean)

avg_cross_genus_poincare = np.mean(cross_genus_distances_poincare)
min_cross_genus_poincare = np.min(cross_genus_distances_poincare)
max_cross_genus_poincare = np.max(cross_genus_distances_poincare)

# Compute statistics for species-to-genus distances
avg_species_to_genus_euclidean = np.mean(species_to_genus_distances_euclidean)
min_species_to_genus_euclidean = np.min(species_to_genus_distances_euclidean)
max_species_to_genus_euclidean = np.max(species_to_genus_distances_euclidean)

avg_species_to_genus_poincare = np.mean(species_to_genus_distances_poincare)
min_species_to_genus_poincare = np.min(species_to_genus_distances_poincare)
max_species_to_genus_poincare = np.max(species_to_genus_distances_poincare)

# Print the statistics
print(f"Within Genus Euclidean Average Distance: {avg_within_genus_euclidean:.4f}")
print(f"Within Genus Euclidean Min Distance: {min_within_genus_euclidean:.4f}")
print(f"Within Genus Euclidean Max Distance: {max_within_genus_euclidean:.4f}")
print(f"Within Genus Poincaré Average Distance: {avg_within_genus_poincare:.4f}")
print(f"Within Genus Poincaré Min Distance: {min_within_genus_poincare:.4f}")
print(f"Within Genus Poincaré Max Distance: {max_within_genus_poincare:.4f}")
print(f"Cross Genus Euclidean Average Distance: {avg_cross_genus_euclidean:.4f}")
print(f"Cross Genus Euclidean Min Distance: {min_cross_genus_euclidean:.4f}")
print(f"Cross Genus Euclidean Max Distance: {max_cross_genus_euclidean:.4f}")
print(f"Cross Genus Poincaré Average Distance: {avg_cross_genus_poincare:.4f}")
print(f"Cross Genus Poincaré Min Distance: {min_cross_genus_poincare:.4f}")
print(f"Cross Genus Poincaré Max Distance: {max_cross_genus_poincare:.4f}")
print(f"Species to Genus Euclidean Average Distance: {avg_species_to_genus_euclidean:.4f}")
print(f"Species to Genus Euclidean Min Distance: {min_species_to_genus_euclidean:.4f}")
print(f"Species to Genus Euclidean Max Distance: {max_species_to_genus_euclidean:.4f}")
print(f"Species to Genus Poincaré Average Distance: {avg_species_to_genus_poincare:.4f}")
print(f"Species to Genus Poincaré Min Distance: {min_species_to_genus_poincare:.4f}")
print(f"Species to Genus Poincaré Max Distance: {max_species_to_genus_poincare:.4f}")

# Write statistics to the results file
with open(results_file, "a") as file:
    file.write(f"Within Genus Euclidean Distance Statistics:\n")
    file.write(f"Average Distance: {avg_within_genus_euclidean:.4f}\n")
    file.write(f"Min Distance: {min_within_genus_euclidean:.4f}\n")
    file.write(f"Max Distance: {max_within_genus_euclidean:.4f}\n")
    file.write(f"Within Genus Poincaré Distance Statistics:\n")
    file.write(f"Average Distance: {avg_within_genus_poincare:.4f}\n")
    file.write(f"Min Distance: {min_within_genus_poincare:.4f}\n")
    file.write(f"Max Distance: {max_within_genus_poincare:.4f}\n")
    file.write(f"Cross Genus Euclidean Distance Statistics:\n")
    file.write(f"Average Distance: {avg_cross_genus_euclidean:.4f}\n")
    file.write(f"Min Distance: {min_cross_genus_euclidean:.4f}\n")
    file.write(f"Max Distance: {max_cross_genus_euclidean:.4f}\n")
    file.write(f"Cross Genus Poincaré Distance Statistics:\n")
    file.write(f"Average Distance: {avg_cross_genus_poincare:.4f}\n")
    file.write(f"Min Distance: {min_cross_genus_poincare:.4f}\n")
    file.write(f"Max Distance: {max_cross_genus_poincare:.4f}\n")
    file.write(f"Species to Genus Euclidean Distance Statistics:\n")
    file.write(f"Average Distance: {avg_species_to_genus_euclidean:.4f}\n")
    file.write(f"Min Distance: {min_species_to_genus_euclidean:.4f}\n")
    file.write(f"Max Distance: {max_species_to_genus_euclidean:.4f}\n")
    file.write(f"Species to Genus Poincaré Distance Statistics:\n")
    file.write(f"Average Distance: {avg_species_to_genus_poincare:.4f}\n")
    file.write(f"Min Distance: {min_species_to_genus_poincare:.4f}\n")
    file.write(f"Max Distance: {max_species_to_genus_poincare:.4f}\n")

# Plot the distributions of distances and save them
plt.hist(within_genus_distances_euclidean, bins=20, color='skyblue', edgecolor='black')
plt.title("Within Genus Euclidean Distances")
plt.xlabel("Distance")
plt.ylabel("Frequency")
plt.savefig(os.path.join(output_dir, "within_genus_euclidean_distances.png"))
plt.close()

plt.hist(within_genus_distances_poincare, bins=20, color='skyblue', edgecolor='black')
plt.title("Within Genus Poincaré Distances")
plt.xlabel("Distance")
plt.ylabel("Frequency")
plt.savefig(os.path.join(output_dir, "within_genus_poincare_distances.png"))
plt.close()

plt.hist(cross_genus_distances_euclidean, bins=20, color='skyblue', edgecolor='black')
plt.title("Cross Genus Euclidean Distances")
plt.xlabel("Distance")
plt.ylabel("Frequency")
plt.savefig(os.path.join(output_dir, "cross_genus_euclidean_distances.png"))
plt.close()

plt.hist(cross_genus_distances_poincare, bins=20, color='skyblue', edgecolor='black')
plt.title("Cross Genus Poincaré Distances")
plt.xlabel("Distance")
plt.ylabel("Frequency")
plt.savefig(os.path.join(output_dir, "cross_genus_poincare_distances.png"))
plt.close()

plt.hist(species_to_genus_distances_euclidean, bins=20, color='skyblue', edgecolor='black')
plt.title("Species to Genus Euclidean Distances")
plt.xlabel("Distance")
plt.ylabel("Frequency")
plt.savefig(os.path.join(output_dir, "species_to_genus_euclidean_distances.png"))
plt.close()

plt.hist(species_to_genus_distances_poincare, bins=20, color='skyblue', edgecolor='black')
plt.title("Species to Genus Poincaré Distances")
plt.xlabel("Distance")
plt.ylabel("Frequency")
plt.savefig(os.path.join(output_dir, "species_to_genus_poincare_distances.png"))
plt.close()

