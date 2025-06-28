import math

import matplotlib.pyplot as plt
import networkx as nx


def radial_layout(G, root, width=2 * math.pi):
    def _layout(node, depth, pos, parent_angle, radius_step):
        children = list(G.successors(node))
        if not children:
            return
        angle_step = width / len(children)
        for i, child in enumerate(children):
            angle = parent_angle - width / 2 + angle_step * (i + 0.5)
            x = (depth + 1) * math.cos(angle)
            y = (depth + 1) * math.sin(angle)
            pos[child] = (x, y)
            _layout(child, depth + 1, pos, angle, radius_step)
    pos = {root: (0, 0)}
    _layout(root, 0, pos, 0, 1)
    return pos

G = nx.DiGraph()
edges = [("Root", "A"), ("Root", "B"), ("A", "A1"), ("A", "A2"), ("B", "B1"), ("B", "B2")]
G.add_edges_from(edges)

pos = radial_layout(G, "Root")
nx.draw(G, pos, with_labels=True, node_size=500)
plt.show()

