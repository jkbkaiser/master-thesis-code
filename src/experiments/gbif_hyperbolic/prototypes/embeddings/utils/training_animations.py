import os

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.animation import FuncAnimation, PillowWriter

import seaborn as sns

import torch

from geoopt.manifolds import PoincareBallExact
from geoopt.tensor import ManifoldParameter


DISTANCE_CMAP = sns.color_palette("rocket", as_cmap=True)
DISTORTION_CMAP = sns.color_palette("vlag", as_cmap=True)


def plot_losses(ax: Axes, losses: list[float], idx: int) -> None:
    ax.clear()
    ax.set_xlim([0, len(losses)])
    minl = min(losses)
    maxl = max(losses)
    margin = (maxl - minl) * 0.1
    ax.set_ylim([minl - margin, maxl + margin])
    ax.plot(range(idx + 1), losses[:idx + 1])


def plot_embeddings(ax: Axes, embeddings: torch.Tensor) -> None:
    ax.clear()
    ax.scatter(x=embeddings[:, 0], y=embeddings[:, 1], s=16)
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])


def plot_distance_mat(
    ax: Axes, distance_mat: torch.Tensor, node_permutation: list[int], first_iter: bool
) -> None:
    ax.clear()
    pos = ax.imshow(
        X=distance_mat[:len(node_permutation), :len(node_permutation)],
        cmap=DISTANCE_CMAP,
        vmin=0,
        vmax=6,
    )
    if first_iter:
        plt.colorbar(pos, ax=ax)


def plot_distortion_mat(
    ax: Axes, distortion_mat: torch.Tensor, node_permutation: list[int], first_iter: bool
) -> None:
    ax.clear()
    pos = ax.imshow(
        X=distortion_mat,
        cmap=DISTORTION_CMAP,
        vmin=-1,
        vmax=1,
    )
    if first_iter:
        plt.colorbar(pos, ax=ax)
    

def animate_training(
    embeddings: list[ManifoldParameter],
    losses: list[float],
    graph_dist_mat: torch.Tensor,
    node_permutation: list[int],
    ball: PoincareBallExact,
    output_dir: str = "",
):
    fig, axs = plt.subplots(2, 2)
    fig.set_size_inches(12, 10)

    def animate(idx: int):
        print(idx)
        cur_embeddings = embeddings[idx]
        perm_cur_embeddings = embeddings[idx][node_permutation]

        plot_losses(axs[0, 0], losses, idx)
        plot_embeddings(axs[0, 1], cur_embeddings)

        embeddings_dist_mat = ball.dist(perm_cur_embeddings[:, None, :], perm_cur_embeddings)
        distortion_mat = (embeddings_dist_mat - graph_dist_mat) / graph_dist_mat

        first_iter = not idx

        plot_distance_mat(axs[1, 0], embeddings_dist_mat, node_permutation, first_iter)
        plot_distortion_mat(axs[1, 1], distortion_mat, node_permutation, first_iter)

        return [fig]

    anim = FuncAnimation(
        fig=fig,
        func=animate,
        frames=range(0, len(losses), 100),
        init_func=lambda: [],
        interval=300,
        blit=True,
    )

    writergif = PillowWriter(fps=10)
    anim.save(os.path.join(output_dir, "training_animation.gif"), writer=writergif)
