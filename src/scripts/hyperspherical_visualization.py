from typing import cast

import datasets
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from mpl_toolkits.mplot3d.proj3d import proj_transform

from src.constants import NUM_PROC
from src.shared.datasets import DatasetVersion

VERSION = DatasetVersion.GBIF_GENUS_SPECIES_100K
path = f"jkbkaiser/{VERSION.value}"
dataset_dict = cast(datasets.DatasetDict, datasets.load_dataset(path, num_proc=NUM_PROC))
ds = dataset_dict["train"].select(range(10000))
img = ds[2558]

# Unit sphere surface
phi, theta = np.mgrid[0:np.pi:200j, 0:2*np.pi:200j]
x = np.sin(phi) * np.cos(theta) * 2
y = np.sin(phi) * np.sin(theta) * 2
z = np.cos(phi) * 2

# Prototypes with higher z, lower x and y
v1 = np.array([0.8, -0.1, 0.5])
v1 = v1 / np.linalg.norm(v1) * 2
v2 = np.array([0.2, 0.7, 0.9])
v2 = v2 / np.linalg.norm(v2) * 2
v3 = np.array([0.4, 0.5, -0.2])
v3 = v3 / np.linalg.norm(v3) * 2

# Setup plot
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(x, y, z, color='lightgray', alpha=0.3, edgecolor='none')

# Prototypes and their vectors
ax.scatter(*v1, color='#7091C6', s=100)
ax.scatter(*v3, color='#3C834C', s=100)

ax.plot([0, v1[0]], [0, v1[1]], [0, v1[2]], color='#7091C6', linewidth=2)
ax.plot([0, v2[0]], [0, v2[1]], [0, v2[2]], color='gray', linewidth=2)
ax.plot([0, v3[0]], [0, v3[1]], [0, v3[2]], color='#3C834C', linewidth=2)

# Coordinate axes
ax.quiver(0, 0, 0, 2, 0, 0, color='black', arrow_length_ratio=0.05)
ax.quiver(0, 0, 0, 0, 2, 0, color='black', arrow_length_ratio=0.05)
ax.quiver(0, 0, 0, 0, 0, 2, color='black', arrow_length_ratio=0.05)

ax.text(2.2, 0, 0, 'X', color='black', fontsize=12)
ax.text(0, 2.05, 0, 'Y', color='black', fontsize=12)
ax.text(0, 0, 2.05, 'Z', color='black', fontsize=12)

# SLERP arc between vectors
def slerp(p0, p1, t_array, radius=2.0):
    # Normalize to unit vectors first
    u0 = p0 / np.linalg.norm(p0)
    u1 = p1 / np.linalg.norm(p1)

    # Compute the angle between the vectors
    omega = np.arccos(np.clip(np.dot(u0, u1), -1, 1))
    so = np.sin(omega)

    # SLERP in unit sphere, then scale up
    points = [
        (np.sin((1.0 - t) * omega) / so) * u0 + (np.sin(t * omega) / so) * u1
        for t in t_array
    ]
    return np.array(points) * radius


# Small circular angle arc near origin
angle_arc = np.array(slerp(v1, v2, np.linspace(0, 1, 100))) * 0.3
ax.plot(angle_arc[:, 0], angle_arc[:, 1], angle_arc[:, 2], color='#7091C6', linewidth=2)

# Label for angle
label_pos = angle_arc[0] + 0.4
ax.text(*label_pos, r'$\theta_1$', fontsize=16, color='#7091C6')

# Small circular angle arc near origin
angle_arc = np.array(slerp(v2, v3, np.linspace(0, 1, 20))) * 0.3
ax.plot(angle_arc[:, 0], angle_arc[:, 1], angle_arc[:, 2], color='#3C834C', linewidth=2)

# Label for angle
label_pos = angle_arc[len(angle_arc)//2] + 0.10
ax.text(*label_pos, r'$\theta_2$', fontsize=16, color='#3C834C')

# View and layout
ax.set_xlim([-2.3, 2.3])
ax.set_ylim([-2.3, 2.3])
ax.set_zlim([-2.3, 2.3])
ax.set_box_aspect([1, 1, 1])
ax.set_axis_off()
ax.view_init(elev=25, azim=35)

# Convert to uint8 for safety
img_arr = img["image"]

if not isinstance(img_arr, np.ndarray):
    img_arr = np.array(img_arr)

if img_arr.dtype != np.uint8:
    img_arr = (img_arr * 255).clip(0, 255).astype(np.uint8)
if img_arr.ndim == 2:
    img_arr = np.stack([img_arr]*3, axis=-1)
elif img_arr.shape[2] == 1:
    img_arr = np.repeat(img_arr, 3, axis=-1)

# Prepare the image box
zoom_factor = 0.2
imagebox = OffsetImage(img_arr, zoom=zoom_factor)

# Project 3D to 2D display coords
x2d, y2d, _ = proj_transform(v2[0], v2[1], v2[2], ax.get_proj())

# Slightly offset the image placement (in screen space)
offset_pixels = (5, 5)  # (x, y) in pixels
ab = AnnotationBbox(
    imagebox,
    (x2d, y2d),
    xybox=offset_pixels,
    xycoords='data',
    boxcoords="offset points",
    frameon=True,
    bboxprops=dict(edgecolor='white', linewidth=1),
    # arrowprops=dict(arrowstyle="->", color="black")
)

# Add the annotation
ax.add_artist(ab)

ax.text(v1[0], v1[1] - 0.5, v1[2], 'Harmonia axyridis', color='black', fontsize=10)
ax.text(v3[0] + 0.6, v3[1] - 0.2, v3[2], 'Anantis ocellata', color='black', fontsize=10)

plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
fig.savefig("your_figure.png", dpi=600, bbox_inches='tight', pad_inches=0.1)


from PIL import Image

# Load the image
input_path = "your_figure.png"
output_path = "your_figure_cropped.png"

with Image.open(input_path) as img:
    width, height = img.size
    crop = 1100
    print(width, height)

    # Define the crop box (left, upper, right, lower)
    crop_box = (crop, crop, width - crop, height - crop)

    # Perform the crop
    cropped_img = img.crop(crop_box)

    # Save the result
    cropped_img.save(output_path)

print(f"Cropped image saved to {output_path}")
