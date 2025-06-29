from functools import partial
from typing import Literal, Optional

import networkx as nx
import numpy as np
import torch

# TODO: Use a decorator to register samplers together with a parser to select from library
from .sample_funcs import (dist_sample_shortest_path,
                           edge_corrupt_both_sampler,
                           edge_corrupt_source_sampler,
                           edge_corrupt_target_sampler,
                           edge_sample_prioritize_siblings,
                           edge_sample_uniform)


class EdgeSampler:
    edge_sample_from_dict = {
        "both": edge_corrupt_both_sampler,
        "source": edge_corrupt_source_sampler,
        "target": edge_corrupt_target_sampler,
    }

    edge_sample_strat_dict = {
        "uniform": edge_sample_uniform,
        "siblings": edge_sample_prioritize_siblings,
    }

    dist_sample_strat_dict = {
        "shortest_path": dist_sample_shortest_path,
    }

    def __init__(
        self,
        # ood_node_list: np.array,
        hierarchy: nx.DiGraph,
        num_negs: int,
        root_id: int,
        edge_sample_from: Literal["both", "source", "target"] = "both",
        edge_sample_strat: Literal["uniform", "siblings"] = "uniform",
        dist_sample_strat: Optional[Literal["shortest_path"]] = None,
    ) -> None:
        # self.ood_node_list = ood_node_list
        self.hierarchy = hierarchy
        self.root_id = root_id
        self.num_negs = num_negs
        self.edge_sample_from = edge_sample_from
        self.edge_sample_strat = edge_sample_strat
        self.dist_sample_strat = dist_sample_strat

        if dist_sample_strat is not None:
            self.dist_sample_strat_fn = self.dist_sample_strat_dict[dist_sample_strat]
            self.undirected_hierarchy = self.hierarchy.to_undirected()
            n = self.undirected_hierarchy.number_of_nodes()
            self.dist_matrix = torch.empty([n, n])

            # TODO: explain this stuff
            for dist_tuple in nx.shortest_path_length(self.undirected_hierarchy):

                i = dist_tuple[0]
                dists = dist_tuple[1]
                if len(dists) < self.dist_matrix.shape[1]:
                    print(f"Node {i} missing distances to some nodes")

                distances_sorted_by_node_id = [d for n, d in sorted(dist_tuple[1].items())]
                self.dist_matrix[dist_tuple[0], :] = torch.tensor(distances_sorted_by_node_id)

            t = (self.dist_matrix == 0).sum()
            print("Total zero dist:", t)

            zero_mask = self.dist_matrix == 0
            off_diagonal_mask = ~torch.eye(self.dist_matrix.shape[0], dtype=torch.bool)
            violations = torch.nonzero(zero_mask & off_diagonal_mask, as_tuple=False)

            if len(violations) > 0:
                print("Off-diagonal zero distances found at:")
                for i, j in violations:
                    print(f"dist_matrix[{i}, {j}] = 0")
            else:
                print("All zero distances are on the diagonal (as expected).")

        self.hierarchy.remove_node(root_id)

        self.edge_sample_fn = partial(
            self.edge_sample_from_dict[edge_sample_from],
            hierarchy=self.hierarchy,
            num_negs=self.num_negs,
            sample_strat=self.edge_sample_strat_dict[edge_sample_strat],
        )


    def sample(self, rel: tuple[int, int]) -> dict[str, torch.Tensor]:
        edges, edge_label_targets = self.edge_sample_fn(rel=rel)
        edges = edges.to(dtype=torch.long)

        sample = {
            "edges": edges,
            "edge_label_targets": edge_label_targets,
        }


        if self.dist_sample_strat is not None:
            dist_targets = self.dist_sample_strat_fn(edges=edges, dist_matrix=self.dist_matrix)
            sample["dist_targets"] = dist_targets

        return sample
