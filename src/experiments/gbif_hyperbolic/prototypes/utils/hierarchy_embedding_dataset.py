from typing import Literal, Optional

import networkx as nx
import torch
from torch.utils.data import Dataset

from src.experiments.gbif_hyperbolic.prototypes.utils.edge_sampler import \
    EdgeSampler


class HierarchyEmbeddingDataset(Dataset):
    def __init__(
        self,
        hierarchy: nx.DiGraph,
        root_id: int,
        num_negs: int = 10,
        edge_sample_from: Literal["both", "source", "target"] = "both",
        edge_sample_strat: Literal["uniform", "siblings"] = "uniform",
        dist_sample_strat: Optional[str] = None,
    ):
        super(HierarchyEmbeddingDataset, self).__init__()
        self.hierarchy = hierarchy
        self.root_id = root_id
        self.num_negs = num_negs
        self.edge_sample_from = edge_sample_from
        self.edge_sample_strat = edge_sample_strat
        self.dist_sample_strat = dist_sample_strat

        self.sampler = EdgeSampler(
            hierarchy=self.hierarchy,
            root_id=self.root_id,
            num_negs=self.num_negs,
            edge_sample_from=edge_sample_from,
            edge_sample_strat=edge_sample_strat,
            dist_sample_strat=dist_sample_strat
        )

        self.edges_list = list(hierarchy.edges())

    def __len__(self) -> int:
        return len(self.edges_list)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        rel = self.edges_list[idx]
        sample = self.sampler.sample(rel=rel)

        # Ensure each tensor in the sample has length == 1 + num_negs
        max_len = 1 + self.num_negs

        for key in sample:
            if sample[key].shape[0] > max_len:
                sample[key] = sample[key][:max_len]
            elif sample[key].shape[0] < max_len:
                # Optional: pad with zeros or raise error if sample too short
                raise ValueError(f"Sample too short for key '{key}': got {sample[key].shape[0]}, expected {max_len}")

        return sample

    # def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
    #     print("start sample")
    #     rel = self.edges_list[idx]
    #     sample = self.sampler.sample(rel=rel)
    #
    #     for key, value in sample.items():
    #         if value.shape[0] != 11:
    #             print("wrong shape")
    #             print(key, value.shape)
    #
    #     print("return sample")
    #     return sample
