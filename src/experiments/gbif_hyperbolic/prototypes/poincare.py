import argparse
import json
import os
import uuid
from pathlib import Path

import mlflow
import networkx as nx
import numpy as np
import requests
import torch
from dotenv import load_dotenv
from geoopt import PoincareBallExact
from geoopt.optim import RiemannianSGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from src.constants import CACHE_DIR, GOOGLE_BUCKET_URL
from src.experiments.gbif_hyperbolic.prototypes.embeddings.poincare_embedding import \
    PoincareEmbedding
from src.experiments.gbif_hyperbolic.prototypes.utils.hierarchy_embedding_dataset import \
    HierarchyEmbeddingDataset
from src.shared.datasets import DatasetVersion

torch.set_float32_matmul_precision("high")
load_dotenv()

MLFLOW_SERVER = os.environ["MLFLOW_SERVER"]
CHECKPOINT_DIR = os.environ["CHECKPOINT_DIR"]


def build_genus_species_graph(genus_species_matrix, genus_names=None, species_names=None):
    num_genus, num_species = genus_species_matrix.shape

    if genus_names is None:
        genus_names = [i for i in range(num_genus)]
    if species_names is None:
        species_names = [num_genus + j for j in range(num_species)]

    G = nx.DiGraph()

    # Add root node with index -1
    G.add_node(-1, index=-1)

    # Add genus nodes with index 0 .. num_genus-1
    for i, genus in enumerate(genus_names):
        G.add_node(genus, index=i)
        G.add_edge(-1, genus)

    # Add species nodes with index offset to avoid collision
    for j, species in enumerate(species_names):
        G.add_node(species, index=num_genus + j)

    # Add edges from genus to species based on the matrix
    for i in range(num_genus):
        for j in range(num_species):
            if genus_species_matrix[i, j] == 1:
                G.add_edge(genus_names[i], species_names[j])

    return G


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
    genusid_2_label = metadata["per_level"][0]["id2label"]
    speciesid_2_label = metadata["per_level"][1]["id2label"]

    id2lable = {}

    for id, label in genusid_2_label.items():
        id2lable[int(id)] = label

    for id, label in speciesid_2_label.items():
        id2lable[int(id) + len(genusid_2_label)] = label

    graph = build_genus_species_graph(hierarchy)

    print("Node count:", len(graph.nodes))
    print(len(id2lable))

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

    ball = PoincareBallExact(c=3.0)
    model = PoincareEmbedding(
        num_embeddings=len(id2lable),
        embedding_dim=args.dims,
        ball=ball,
    )

    lr = 0.1
    burn_in_lr_mult = 1 / 10
    epochs = 200
    burn_in_epochs = 10
    momentum = 0.9
    weight_decay = 0.0005

    mlflow.set_tracking_uri(MLFLOW_SERVER)
    mlflow.set_experiment("prototypes")

    optimizer = RiemannianSGD(
        params=model.parameters(),
        lr=lr,
        momentum=momentum,
        dampening=0,
        weight_decay=weight_decay,
        nesterov=True,
        stabilize=500
    )

    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-10)

    with mlflow.start_run():
        mlflow.log_params({
            "nodes": len(graph.nodes),
            "type": "poincare",
            "lr": lr,
            "epochs": epochs,
            "momentum": momentum,
            "weight_decay": weight_decay,
            "scheduler": "cosine annealing",
            "dims": args.dims,
        })

        model.train_model(
            dataloader=dataloader,
            epochs=epochs,
            optimizer=optimizer,
            burn_in_epochs=burn_in_epochs,
            burn_in_lr_mult=burn_in_lr_mult,
            scheduler=scheduler
        )

    base = Path("./prototypes/gbif_genus_species_100k/genus_species_poincare")

    if not base.is_dir():
        base.mkdir(parents=True)

    np.save(f"{base}/{args.dims}.npy", model.weight.data.cpu().numpy())

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
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args)
