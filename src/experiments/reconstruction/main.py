import argparse
import os

import mlflow
import networkx as nx
import torch
from dotenv import load_dotenv
from geoopt import ManifoldParameter, ManifoldTensor, PoincareBallExact
from torch.utils.data import DataLoader

from src.constants import DEVICE
from src.experiments.gbif_hyperbolic.prototypes.embeddings.poincare_embedding import \
    PoincareEmbedding
from src.experiments.gbif_hyperbolic.prototypes.utils.hierarchy_embedding_dataset import \
    HierarchyEmbeddingDataset
from src.shared.datasets import DatasetVersion, get_hierarchy, get_metadata
from src.shared.prototypes import PrototypeVersion, get_prototypes

load_dotenv()

MLFLOW_SERVER = os.environ["MLFLOW_SERVER"]
CHECKPOINT_DIR = os.environ["CHECKPOINT_DIR"]


def build_hierarchical_graph(hierarchy_matrices, level_names):
    G = nx.DiGraph()
    node_index = 0
    level_offsets = []
    node_labels = {}

    for names in level_names:
        level_offsets.append(node_index)
        for name in names:
            G.add_node(node_index, label=name)
            node_labels[name] = node_index
            node_index += 1

    root_index = node_index
    G.add_node(root_index, label="root")

    top_level_offset = level_offsets[0]
    num_top_level = len(level_names[0])
    for i in range(num_top_level):
        G.add_edge(root_index, top_level_offset + i)

    for level in range(len(hierarchy_matrices)):
        mat = hierarchy_matrices[level]
        parent_offset = level_offsets[level]
        child_offset = level_offsets[level + 1]

        num_parents, num_children = mat.shape
        for i in range(num_parents):
            for j in range(num_children):
                if mat[i, j] == 1:
                    parent_idx = parent_offset + i
                    child_idx = child_offset + j
                    G.add_edge(parent_idx, child_idx)

    return G, root_index


def compute_map_score(dists: torch.Tensor):
    ranks = torch.argsort(dists, dim=1)
    pos_ranks = (ranks == 0).nonzero(as_tuple=False)[:, 1]
    ap = 1.0 / (pos_ranks + 1).float()

    return ap.mean().item(), pos_ranks.float().mean().item()


def run(args):
    hierarchy = get_hierarchy(args.dataset)
    metadata = get_metadata(args.dataset)

    id2label_dict = {
        rank: {int(k): v for k, v in per_level["id2label"].items()}
        for rank, per_level in zip(["class", "order", "family", "subfamily", "genus", "species"], metadata["per_level"])
    }

    level_names = [list(id2label_dict[rank].values()) for rank in ["class", "order", "family", "subfamily", "genus", "species"] if rank in id2label_dict]

    graph, root_id = build_hierarchical_graph(
        hierarchy_matrices=hierarchy,
        level_names=level_names
    )

    print("Node count:", len(graph.nodes))

    mlflow.set_tracking_uri(MLFLOW_SERVER)
    mlflow.set_experiment("reconstruction")

    dataset = HierarchyEmbeddingDataset(
        hierarchy=graph,
        root_id=root_id,
        num_negs=10,
        edge_sample_from="both",
        edge_sample_strat="uniform",
        dist_sample_strat="shortest_path",
    )

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=True,
    )

    ball = PoincareBallExact(c=args.curvature)
    model = PoincareEmbedding(
        num_embeddings=len(graph.nodes),
        embedding_dim=args.dims,
        ball=ball,
    )

    prototypes = get_prototypes(PrototypeVersion.DISTORTION.value, args.dataset.value, args.dims)

    model.weight = ManifoldParameter(
        data=ManifoldTensor(prototypes, manifold=ball).to(DEVICE)
    )

    mean_map = 0
    mean_rank = 0
    i = 0

    while i < 1000:
        for batch in dataloader:
            edges = batch["edges"].to(model.weight.device)

            dists = model(edges)
            map, mrank = compute_map_score(dists)

            mean_map += map
            mean_rank += mrank

            i += 1

            if i >= 1000:
                break

    print(i)
    print(mean_map / i)
    print(mean_rank / i)


def parse_args():
    parser = argparse.ArgumentParser(
        prog="Hyperbolic embeddings",
        description="Training script for embedding genus and species in hyperbolic space",
    )
    parser.add_argument("--batch-size", default=16, required=False, type=int)
    parser.add_argument(
        "-d",
        "--dataset",
        required=True,
        type=DatasetVersion,
        choices=[v.value for v in DatasetVersion],
    )
    parser.add_argument('--curvature', default=1.5, type=float)
    parser.add_argument('--dims', default=128, type=int)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args)
