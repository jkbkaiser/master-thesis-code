import argparse
import json
import os
import uuid
from pathlib import Path

import mlflow
import networkx as nx
import numpy as np
import requests
from dotenv import load_dotenv
from geoopt import PoincareBallExact
from geoopt.optim import RiemannianSGD
from torch.utils.data import DataLoader

from src.constants import CACHE_DIR, GOOGLE_BUCKET_URL
from src.experiments.gbif_hyperbolic.prototypes.embeddings.distortion_embedding import \
    DistortionEmbedding
from src.experiments.gbif_hyperbolic.prototypes.utils.hierarchy_embedding_dataset import \
    HierarchyEmbeddingDataset
from src.shared.datasets import DatasetVersion

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

    mlflow.set_tracking_uri(MLFLOW_SERVER)
    mlflow.set_experiment("prototypes")

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
    model = DistortionEmbedding(
        num_embeddings=len(graph.nodes),
        embedding_dim=args.dims,
        ball=ball,
    )

    lr = 1.0
    burn_in_lr_mult = 1 / 10
    epochs = 100
    burn_in_epochs = 10
    momentum = 0.9
    weight_decay = 0.0005

    optimizer = RiemannianSGD(
        params=model.parameters(),
        lr=lr,
        momentum=momentum,
        dampening=0,
        weight_decay=weight_decay,
        nesterov=True,
        stabilize=500
    )

    with mlflow.start_run():
        mlflow.log_params({
            "nodes": len(graph.nodes),
            "type": "distortion",
            "curvature": args.curvature,
            "distortion_lr": lr,
            "distortion_epochs": epochs,
            "distortion_burn_in_lr_mult": burn_in_lr_mult,
            "distortion_burn_in_epochs": burn_in_epochs,
            "distortion_momentum": momentum,
            "distortion_weight_decay": weight_decay,
            "distortion_optimizer": "riemannian sgd"
        })

        model.train_model(
            dataloader=dataloader,
            epochs=epochs,
            optimizer=optimizer,
            burn_in_epochs=burn_in_epochs,
            burn_in_lr_mult=burn_in_lr_mult,
            store_losses=True,
        )

    base = Path(f"./prototypes/{args.dataset}/distortion")

    if not base.is_dir():
        base.mkdir(parents=True)

    print(model.weight.data.cpu().numpy().shape)

    path = f"{base}/{args.dims}.npy"
    print(path)
    np.save(path, model.weight.data.cpu().numpy())

def parse_args():
    parser = argparse.ArgumentParser(
        prog="Hyperbolic embeddings",
        description="Training script for embedding genus and species in hyperbolic space",
    )
    parser.add_argument("--batch-size", default=256, required=False, type=int)
    parser.add_argument(
        "-d",
        "--dataset",
        required=True,
        type=DatasetVersion,
        choices=[v.value for v in DatasetVersion],
    )
    parser.add_argument(
        "--dims",
        default=2,
        required=True,
        type=int,
    )
    parser.add_argument('--lr', dest="learning_rate", default=0.1, type=float)
    parser.add_argument("--reload", action="store_true", default=False, required=False)
    parser.add_argument('--momentum', dest="momentum", default=0.9, type=float)
    parser.add_argument('--epochs', dest="epochs", default=10000, type=int,)
    parser.add_argument('--resdir', default="./prototypes", type=Path)
    parser.add_argument("--curvature", default=3., required=False, type=float)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args)
