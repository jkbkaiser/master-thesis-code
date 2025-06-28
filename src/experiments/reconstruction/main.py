import argparse
import json
import os
import uuid
from pathlib import Path
from re import DEBUG

import mlflow
import networkx as nx
import numpy as np
import requests
import torch
from dotenv import load_dotenv
from geoopt import ManifoldParameter, ManifoldTensor, PoincareBallExact
from geoopt.optim import RiemannianSGD
from torch.utils.data import DataLoader

from src.constants import CACHE_DIR, DEVICE, GOOGLE_BUCKET_URL
from src.experiments.gbif_hyperbolic.prototypes.embeddings.base import \
    BaseEmbedding
from src.experiments.gbif_hyperbolic.prototypes.embeddings.entailment_cones import \
    EntailmentConeEmbedding
from src.experiments.gbif_hyperbolic.prototypes.embeddings.poincare_embedding import \
    PoincareEmbedding
from src.experiments.gbif_hyperbolic.prototypes.utils.hierarchy_embedding_dataset import \
    HierarchyEmbeddingDataset
from src.shared.datasets import DatasetVersion
from src.shared.prototypes import PrototypeVersion, get_prototypes

load_dotenv()

MLFLOW_SERVER = os.environ["MLFLOW_SERVER"]
CHECKPOINT_DIR = os.environ["CHECKPOINT_DIR"]


def build_hierarchical_graph(hierarchy_matrices, level_names):
    G = nx.DiGraph()
    node_index = 0
    level_offsets = []
    node_labels = {}  # index → name

    # Calculate offsets for index uniqueness across levels
    for names in level_names:
        level_offsets.append(node_index)
        for name in names:
            G.add_node(node_index, label=name)
            node_labels[name] = node_index
            node_index += 1

    # Add a virtual root
    root_index = -1
    G.add_node(root_index, label="root")
    for i in range(len(level_names[0])):
        G.add_edge(root_index, level_offsets[0] + i)

    # Add edges per level
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

    return G


def get_hierarchy(dataset_version, reload: bool = False):
    path = CACHE_DIR / f"{dataset_version}/hierarchy.npz"

    if reload or not path.is_file():
        url = f"{GOOGLE_BUCKET_URL}/{dataset_version}/hierarchy.npz?id={uuid.uuid4()}"
        response = requests.get(url)

        if response.status_code != 200:
            raise Exception(f"Could not retrieve hierarchy for {dataset_version}")

        with open(path, "wb") as f:
            f.write(response.content)

    data = np.load(path)

    if "data" in data:
        # Single hierarchy matrix
        return [data["data"].squeeze()]

    # Otherwise, assume multiple levels: level_0, level_1, ...
    hierarchy = [data[key] for key in sorted(data.files, key=lambda k: int(k.split("_")[1]))]
    return hierarchy


def get_metadata(dataset_version, reload: bool = False):
    directory = CACHE_DIR / dataset_version
    path = directory / "metadata.json"

    if reload or not path.is_file():
        url = f"{GOOGLE_BUCKET_URL}/{dataset_version}/metadata.json?id={uuid.uuid4()}"

        response = requests.get(url)
        if response.status_code != 200:
            raise Exception(f"Could not retrieve metadata for {dataset_version}, status: {response.status_code}")
        metadata = response.json()

        if not directory.exists():
            directory.mkdir(parents=True)

        with open(path, "w") as f:
            json.dump(metadata, f)

        return metadata

    with open(path, "r") as f:
        return json.load(f)


def compute_map_score(dists: torch.Tensor):
    ranks = torch.argsort(dists, dim=1)
    pos_ranks = (ranks == 0).nonzero(as_tuple=False)[:, 1]
    ap = 1.0 / (pos_ranks + 1).float()

    return ap.mean().item(), pos_ranks.float().mean().item()



def run(args):
    hierarchy = get_hierarchy(args.dataset)
    metadata = get_metadata(args.dataset)

    # Build id2label dict for all ranks
    id2label_dict = {
        rank: {int(k): v for k, v in per_level["id2label"].items()}
        for rank, per_level in zip(["class", "order", "family", "subfamily", "genus", "species"], metadata["per_level"])
    }

    # Construct level_names for the graph
    level_names = [list(id2label_dict[rank].values()) for rank in ["class", "order", "family", "subfamily", "genus", "species"] if rank in id2label_dict]

    graph = build_hierarchical_graph(
        hierarchy_matrices=hierarchy,
        level_names=level_names
    )

    print("Node count:", len(graph.nodes))

    mlflow.set_tracking_uri(MLFLOW_SERVER)
    mlflow.set_experiment("reconstruction")

    dataset = HierarchyEmbeddingDataset(
        hierarchy=graph,
        root_id=-1,
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

    prototypes = get_prototypes(PrototypeVersion.AVG_MULTI.value, args.dataset.value, args.dims)

    print(prototypes.shape)

    model.weight = ManifoldParameter(
        data=ManifoldTensor(prototypes, manifold=ball).to(DEVICE)
    )

    mean_map = 0
    mean_rank = 0
    i = 0

    while i < 1000:
        for batch in dataloader:
            edges = batch["edges"].to(model.weight.device)
            mask = batch["mask"].to(model.weight.device)
            edge_label_targets = batch["edge_label_targets"].to(model.weight.device)

            dists = model(edges)

            # source = edges[:,:,0]
            # dest = edges[:,:,1]
            #
            # source_prototypes = model(source)
            # dest_prototypes = model(dest)
            #
            # dists = ball.dist(source_prototypes, dest_prototypes)
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
