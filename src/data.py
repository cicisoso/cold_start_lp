"""
ColdCL: Uncertainty-Aware Contrastive Learning for Cold-Start Link Prediction

Dataset loading and edge-split utilities.
"""
import os
import torch
import numpy as np
from torch_geometric.datasets import Planetoid, Amazon, Coauthor
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.utils import degree, to_undirected


DATASETS = {
    "Cora": lambda root: Planetoid(root=root, name="Cora"),
    "CiteSeer": lambda root: Planetoid(root=root, name="CiteSeer"),
    "PubMed": lambda root: Planetoid(root=root, name="PubMed"),
    "Photo": lambda root: Amazon(root=root, name="Photo"),
    "CS": lambda root: Coauthor(root=root, name="CS"),
}


def load_dataset(name, root=None, val_ratio=0.05, test_ratio=0.1, seed=42):
    """Load dataset and create LP train/val/test splits."""
    if root is None:
        root = os.path.join(os.path.dirname(__file__), "..", "data")

    if name not in DATASETS:
        raise ValueError(f"Unknown dataset: {name}. Choose from {list(DATASETS.keys())}")

    dataset = DATASETS[name](root)
    data = dataset[0]

    # Ensure undirected
    data.edge_index = to_undirected(data.edge_index)

    transform = RandomLinkSplit(
        num_val=val_ratio,
        num_test=test_ratio,
        is_undirected=True,
        add_negative_train_samples=True,
        split_labels=True,
    )

    # Set seed for reproducible splits
    torch.manual_seed(seed)
    np.random.seed(seed)
    train_data, val_data, test_data = transform(data)

    return data, train_data, val_data, test_data


def get_node_degrees(data):
    """Compute undirected degree for each node."""
    deg = degree(data.edge_index[0], num_nodes=data.num_nodes)
    deg = deg + degree(data.edge_index[1], num_nodes=data.num_nodes)
    return deg


def get_edge_degree_bins(original_data, edge_label_index, thresholds=(1, 5, 20)):
    """Bin edges by minimum endpoint degree in the original graph."""
    deg = get_node_degrees(original_data)
    src, dst = edge_label_index[0], edge_label_index[1]
    min_deg = torch.minimum(deg[src], deg[dst])

    lo, mid, hi = thresholds
    bins = {
        f"isolated (0-{lo})": min_deg <= lo,
        f"cold ({lo+1}-{mid})": (min_deg > lo) & (min_deg <= mid),
        f"warm ({mid+1}-{hi})": (min_deg > mid) & (min_deg <= hi),
        f"hot (>{hi})": min_deg > hi,
    }
    return bins, deg


def dataset_stats(data, name=""):
    """Print dataset statistics."""
    deg = get_node_degrees(data)
    n_edges = data.edge_index.size(1) // 2  # undirected
    print(f"Dataset: {name}")
    print(f"  Nodes: {data.num_nodes}, Edges: {n_edges}, Features: {data.num_features}")
    print(f"  Degree — mean: {deg.mean():.1f}, median: {deg.median():.1f}, "
          f"min: {deg.min():.0f}, max: {deg.max():.0f}")
    cold_frac = (deg <= 5).float().mean()
    print(f"  Cold nodes (deg≤5): {cold_frac:.1%}")
    return deg
