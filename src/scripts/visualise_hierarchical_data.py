import json

import numpy as np

from src.shared.datasets import DatasetVersion, get_hierarchy, get_metadata

hierarchy = get_hierarchy(DatasetVersion.BIOSCAN)
metadata = get_metadata(DatasetVersion.BIOSCAN)
label_maps = [lvl["id2label"] for lvl in metadata["per_level"]]

def build_node(level, index):
    label = label_maps[level][str(index)]
    node = {"name": label}

    if level >= len(hierarchy):
        return node

    row = hierarchy[level][index]
    child_indices = np.where(row == 1)[0]

    if len(child_indices) > 0:
        node["children"] = [build_node(level + 1, i) for i in child_indices]

    return node

root = build_node(0, 0)

with open("tree_data.json", "w") as f:
    json.dump(root, f, indent=2)
