import geoopt
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from matplotlib.patches import Circle

sns.set_theme("talk")
sns.set_palette("deep")

ball = geoopt.PoincareBall(c=1.0)
fig, ax = plt.subplots(figsize=(6,6))
circle = Circle((0, 0), 1, color='b')
ax.add_artist(circle)
ax.set_xlim(( -1.1, 1.1 ))
ax.set_ylim(( -1.1, 1.1 ))
ax.set_aspect('equal', 'box')
ax.set_title("Poincaré Disk model")
ax.set_xticks([])
ax.set_yticks([])


def plot_points(points, color="r"):
    num_samples = 25
    n = len(points)
    points_on_manifold = ball.expmap0(points)

    timesteps = torch.linspace(0, 1, num_samples)
    duplicated_timesteps = timesteps.repeat_interleave(2)
    timesteps_pairs = duplicated_timesteps.view(-1, 2)

    for i in range(n):
        a = points_on_manifold[i]
        b = points_on_manifold[(i + 1) % n]

        points = np.array([
            ball.geodesic(t, a, b)
            for t in timesteps_pairs
        ])

        # ax.plot(points[:, 0], points[:, 1], 'ro')

        for i in range(len(points) - 1):
            ax.plot([points[i, 0], points[i + 1, 0]], [points[i, 1], points[i + 1, 1]], color)


def main():
    points = torch.tensor([
        [0.0, 0.5],
        [-1.2, 0],
        [-1.2, -0.6],
        [1.2, -0.8],
    ])

    plot_points(points, color="r")

    points = torch.tensor([
        [0.2, 0.8],
        [-0.5, 0.7],
        [-0.3, -0.7],
    ])

    plot_points(points, color="g")

    points = torch.tensor([
        [0.2, -0.2],
        [1.0, -0.2],
        [1.0, -1.0],
        [0.2, -1.0],
    ])

    plot_points(points, color="y")

    plt.show()


if __name__ == "__main__":
    main()
