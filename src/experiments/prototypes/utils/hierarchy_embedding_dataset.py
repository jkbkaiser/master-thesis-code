from typing import Literal, Optional

import networkx as nx
import torch
from torch.utils.data import Dataset

from src.experiments.prototypes.utils.edge_sampler import EdgeSampler


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
        if 0 in rel:
            return self.__getitem__((idx + 1) % len(self))
        sample = self.sampler.sample(rel=rel)

        return sample
