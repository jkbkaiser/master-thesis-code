import geoopt
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Circle

DEEP_BLUE = '#4C72B0'
DEEP_ORANGE = '#DD8452'
DEEP_GREEN = '#55A868'
DEEP_RED = '#C44E52'
DEEP_PURPLE = '#8172B3'
DEEP_BROWN = '#937860'


sphere_size = 0.95

ball = geoopt.PoincareBall(c=1.0)
fig, ax = plt.subplots(figsize=(6,6))
circle = Circle((0, 0.04), sphere_size, color='lightgray')
ax.add_artist(circle)
ax.set_xlim(( -(sphere_size + 0.1), sphere_size + 0.1 ))
ax.set_ylim(( -(sphere_size + 0.1), sphere_size + 0.1 ))
ax.set_aspect('equal', 'box')
ax.set_xticks([])
ax.set_yticks([])


def plot_points(points, color="r", num_samples = 25):
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

        for i in range(len(points) - 1):
            ax.plot([points[i, 0], points[i + 1, 0]], [points[i, 1], points[i + 1, 1]], color)


def main():
    points = torch.tensor([
        [0.2, 0.8],
        [-0.5, 0.7],
        [-0.6, -0.3],
    ])

    plot_points(points, color=DEEP_ORANGE)

    points = torch.tensor([
        [1.8, 0.0],
        [0.0, -1.5],
    ])

    plot_points(points, color=DEEP_GREEN, num_samples=200)

    points = torch.tensor([
        [-0.3 * 0.93, -1.6 * 0.93],
        [-0.5 * 1.6, -1.0 * 1.3],
    ])

    plot_points(points, color=DEEP_PURPLE, num_samples=200)


    plt.show()


if __name__ == "__main__":
    main()
