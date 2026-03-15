"""Common utilities for cold-start LP pilots."""
import os
import json
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.nn import GCNConv, SAGEConv
from torch_geometric.utils import degree, to_undirected
from sklearn.metrics import roc_auc_score, average_precision_score


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_dataset(name="Cora"):
    """Load a Planetoid dataset and split edges for LP."""
    path = os.path.join(os.path.dirname(__file__), "..", "data")
    dataset = Planetoid(root=path, name=name)
    data = dataset[0]

    transform = RandomLinkSplit(
        num_val=0.05,
        num_test=0.1,
        is_undirected=True,
        add_negative_train_samples=True,
        split_labels=True,
    )
    train_data, val_data, test_data = transform(data)
    return data, train_data, val_data, test_data


def get_degree_bins(data, edge_label_index):
    """Bin test edges by minimum endpoint degree."""
    deg = degree(data.edge_index[0], num_nodes=data.num_nodes)
    deg = deg + degree(data.edge_index[1], num_nodes=data.num_nodes)
    src, dst = edge_label_index
    min_deg = torch.minimum(deg[src], deg[dst])
    bins = {
        "isolated (0-1)": min_deg <= 1,
        "cold (2-5)": (min_deg >= 2) & (min_deg <= 5),
        "warm (6-20)": (min_deg >= 6) & (min_deg <= 20),
        "hot (>20)": min_deg > 20,
    }
    return bins, deg


class GCNLinkPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels=128):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)

    def encode(self, x, edge_index):
        h = self.conv1(x, edge_index).relu()
        h = self.conv2(h, edge_index)
        return h

    def decode(self, z, edge_label_index):
        src, dst = edge_label_index
        return (z[src] * z[dst]).sum(dim=-1)

    def forward(self, x, edge_index, edge_label_index):
        z = self.encode(x, edge_index)
        return self.decode(z, edge_label_index)


class SAGELinkPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels=128):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)

    def encode(self, x, edge_index):
        h = self.conv1(x, edge_index).relu()
        h = self.conv2(h, edge_index)
        return h

    def decode(self, z, edge_label_index):
        src, dst = edge_label_index
        return (z[src] * z[dst]).sum(dim=-1)

    def forward(self, x, edge_index, edge_label_index):
        z = self.encode(x, edge_index)
        return self.decode(z, edge_label_index)


def train_lp_model(model, train_data, val_data, epochs=200, lr=0.01, device="cuda"):
    """Standard LP training loop."""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)

    train_data = train_data.to(device)
    val_data = val_data.to(device)

    best_val_auc = 0
    best_state = None
    patience = 30
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        pos_edge = train_data.pos_edge_label_index
        neg_edge = train_data.neg_edge_label_index
        edge_label_index = torch.cat([pos_edge, neg_edge], dim=1)
        labels = torch.cat([
            torch.ones(pos_edge.size(1)),
            torch.zeros(neg_edge.size(1)),
        ]).to(device)

        out = model(train_data.x, train_data.edge_index, edge_label_index)
        loss = F.binary_cross_entropy_with_logits(out, labels)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            val_auc = eval_lp(model, val_data, device)
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
            if no_improve >= patience // 10:
                break

    if best_state:
        model.load_state_dict(best_state)
    return model, best_val_auc


def eval_lp(model, data, device="cuda"):
    """Evaluate LP model, return AUC."""
    model.eval()
    data = data.to(device)
    with torch.no_grad():
        pos_edge = data.pos_edge_label_index
        neg_edge = data.neg_edge_label_index
        edge_label_index = torch.cat([pos_edge, neg_edge], dim=1)
        labels = torch.cat([
            torch.ones(pos_edge.size(1)),
            torch.zeros(neg_edge.size(1)),
        ])

        out = model(data.x, data.edge_index, edge_label_index)
        pred = torch.sigmoid(out).cpu().numpy()
        auc = roc_auc_score(labels.numpy(), pred)
    return auc


def eval_lp_by_degree(model, data, original_data, device="cuda"):
    """Evaluate LP model stratified by degree bins."""
    model.eval()
    data = data.to(device)
    results = {}

    with torch.no_grad():
        pos_edge = data.pos_edge_label_index
        neg_edge = data.neg_edge_label_index
        edge_label_index = torch.cat([pos_edge, neg_edge], dim=1)
        labels = torch.cat([
            torch.ones(pos_edge.size(1)),
            torch.zeros(neg_edge.size(1)),
        ])
        out = model(data.x, data.edge_index, edge_label_index)
        pred = torch.sigmoid(out).cpu()

        bins, deg = get_degree_bins(original_data, edge_label_index.cpu())
        overall_auc = roc_auc_score(labels.numpy(), pred.numpy())
        results["overall"] = {"auc": overall_auc, "count": len(labels)}

        for bin_name, mask in bins.items():
            if mask.sum() > 10:
                bin_labels = labels[mask].numpy()
                bin_pred = pred[mask].numpy()
                if len(np.unique(bin_labels)) > 1:
                    auc = roc_auc_score(bin_labels, bin_pred)
                else:
                    auc = float("nan")
                results[bin_name] = {"auc": auc, "count": int(mask.sum())}
            else:
                results[bin_name] = {"auc": float("nan"), "count": int(mask.sum())}

    return results


def save_results(results, filename):
    """Save results to JSON."""
    path = os.path.join(os.path.dirname(__file__), "..", "results", filename)
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Results saved to {path}")
