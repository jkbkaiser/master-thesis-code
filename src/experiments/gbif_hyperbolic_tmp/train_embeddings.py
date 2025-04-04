import argparse
import time

import matplotlib.pyplot as plt
import networkx as nx
from geoopt import PoincareBallExact
from geoopt.optim import RiemannianSGD
from torch.utils.data import DataLoader

from src.constants import DEVICE
from src.experiments.gbif_hyperbolic.dataset import HierarchyEmbeddingDataset
from src.experiments.gbif_hyperbolic.embeddings.poincare_embeddings.embedding import \
    PoincareEmbedding
from src.shared.datasets import Dataset, DatasetType, DatasetVersion

from .embeddings.distortion.embedding import DistortionEmbedding


def visualize_graph(G):
    plt.figure(figsize=(8, 5))

    # Use Graphviz layout for a hierarchical structure
    pos = nx.nx_agraph.graphviz_layout(G, prog="dot")  

    nx.draw(
        G, pos, with_labels=True, node_size=2000, 
        node_color="lightblue", edge_color="gray", font_size=10
    )

    plt.title("Genus-Species Hierarchy")
    plt.show()


def create_hierarchy_graph(genus_species_matrix, genus_names, species_names):
    G = nx.DiGraph()

    # Add root node with standardized properties
    root_id = 0
    G.add_node(root_id, label="root")

    num_genera, num_species = genus_species_matrix.shape

    # Add genus nodes with auto-incrementing IDs
    for genus_idx in range(num_genera):
        genus_id = genus_idx + 1
        G.add_node(genus_id, label=genus_names[genus_idx])
        G.add_edge(root_id, genus_id)

    # Add species nodes with offset IDs
    species_id_offset = num_genera + 1
    for species_idx in range(num_species):
        species_id = species_id_offset + species_idx
        species_label = species_names[species_idx]

        # Only add species if it exists in at least one genus
        if genus_species_matrix[:, species_idx].any():
            G.add_node(species_id, label=species_label)

            # Connect to all parent genera
            for genus_idx in range(num_genera):
                if genus_species_matrix[genus_idx, species_idx] == 1:
                    parent_id = genus_idx + 1
                    G.add_edge(parent_id, species_id)

    return root_id, G

def run(args):
    ds = Dataset(args.dataset)
    ds.load(batch_size=args.batch_size, use_torch=True, reload=args.reload)

    genus_species_matrix = ds.hierarchy[0]

    genus_names = ds.id2label_per_level[0]
    species_names = ds.id2label_per_level[1]

    root_id, graph = create_hierarchy_graph(genus_species_matrix, genus_names, species_names)
    # visualize_graph(graph)


    lr=1e0
    burn_in_lr_mult=1/10
    epochs=10000
    burn_in_epochs= 10
    embedding_dim=64
    save_prototypes=True

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
        batch_size=256,
        shuffle=True,
    )

    ball = PoincareBallExact(c=1.0)

    model = PoincareEmbedding(
        num_embeddings=graph.number_of_nodes() + 1,
        embedding_dim=embedding_dim,
        ball=ball,
    )

    model.to(DEVICE)

    optimizer = RiemannianSGD(
        params=model.parameters(),
        lr=lr,
        momentum=0.9,
        dampening=0,
        weight_decay=0.0005,
        nesterov=True,
        stabilize=500
    )

    # Train the model
    start = time.time()

    losses, _ = model.train(
        dataloader=dataloader,
        epochs=epochs,
        optimizer=optimizer,
        burn_in_epochs=burn_in_epochs,
        burn_in_lr_mult=burn_in_lr_mult,
        store_losses=True,
    )

    print(f"Elapsed training time: {time.time() - start:.3f} seconds")




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
    parser.add_argument("--reload", action="store_true", default=False, required=False)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args)
