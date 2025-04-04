import os

import matplotlib.pyplot as plt

from geoopt.manifolds import PoincareBallExact

import pandas as pd
import seaborn as sn

import torch


def create_dist_mat_plots(
    embeddings: torch.Tensor,
    node_permutation: list[int],
    ball: PoincareBallExact,
    output_dir: str = "",
) -> torch.Tensor:
    if node_permutation is not None:
        embeddings = embeddings[node_permutation]

    dist_mat = ball.dist(embeddings[:, None, :], embeddings)

    df_cm = pd.DataFrame(
        dist_mat[:len(node_permutation), :len(node_permutation)],
        node_permutation,
        node_permutation,
    )

    fig = plt.figure(figsize=(10, 8))
    sn.set(font_scale=0.5)
    sn.heatmap(df_cm)
    fig.savefig(os.path.join(output_dir, f"prototype_edge_distances.png"))
    plt.clf()

    return dist_mat


def create_dist_diff_mat_plots(
    embeddings_dist_mat: torch.Tensor,
    graph_dist_mat: torch.Tensor,
    node_permutation: list[int],
    output_dir: str = "",
) -> None:
    diff_mat = (embeddings_dist_mat - graph_dist_mat) / graph_dist_mat
    df_cm = pd.DataFrame(
        diff_mat[:len(node_permutation), :len(node_permutation)],
        node_permutation,
        node_permutation,
    )

    fig = plt.figure(figsize=(10, 8))
    sn.set(font_scale=0.5)
    sn.heatmap(df_cm)
    fig.savefig(os.path.join(output_dir, f"prototype_edge_distortions.png"))
    plt.clf()
