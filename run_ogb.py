"""
OGB-scale benchmarks for tail-node LP analysis.

Datasets: ogbl-collab, ogbl-citation2
Tests: Vanilla, GlobalCL, AugOnly + degree-stratified analysis
"""
import os
import sys
import json
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import degree, to_undirected
from sklearn.metrics import roc_auc_score
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))
from src.models import GCNEncoder, SAGEEncoder, ENCODERS


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# OGB dataset loading
# ============================================================

def load_ogb_dataset(name, root=None):
    """Load OGB LP dataset with its official split."""
    from ogb.linkproppred import LinkPropPredDataset

    if root is None:
        root = os.path.join(os.path.dirname(__file__), "data")

    dataset = LinkPropPredDataset(name=name, root=root)
    data = dataset[0]

    # Convert to PyG-style
    edge_index = torch.from_numpy(data['edge_index']) if not isinstance(data['edge_index'], torch.Tensor) else data['edge_index']

    # Get node features if available
    if 'x' in data and data['x'] is not None:
        x = torch.FloatTensor(data['x']) if not isinstance(data['x'], torch.Tensor) else data['x'].float()
    elif 'node_feat' in data and data['node_feat'] is not None:
        x = torch.FloatTensor(data['node_feat']) if not isinstance(data['node_feat'], torch.Tensor) else data['node_feat'].float()
    else:
        # No features — use degree-based features
        num_nodes = data['num_nodes']
        x = None  # Will compute later

    num_nodes = data['num_nodes']

    # Get split
    split = dataset.get_edge_split()

    return edge_index, x, num_nodes, split, dataset


def prepare_ogb_data(name, max_nodes=None, max_train_edges=None, seed=42):
    """
    Prepare OGB data for our experiments.
    For large graphs, subsample to fit in GPU memory.
    """
    set_seed(seed)
    edge_index, x, num_nodes, split, dataset = load_ogb_dataset(name)

    train_edge = torch.from_numpy(split['train']['edge']) if not isinstance(split['train']['edge'], torch.Tensor) else split['train']['edge']
    valid_edge = torch.from_numpy(split['valid']['edge']) if not isinstance(split['valid']['edge'], torch.Tensor) else split['valid']['edge']
    test_edge = torch.from_numpy(split['test']['edge']) if not isinstance(split['test']['edge'], torch.Tensor) else split['test']['edge']

    # For citation2: edges are directed, make undirected for message passing
    if name == 'ogbl-citation2':
        # Subsample for memory — citation2 has 2.9M nodes
        if max_nodes and num_nodes > max_nodes:
            print(f"  Subsampling {name} from {num_nodes} to {max_nodes} nodes...")
            # Keep nodes with most edges (core subgraph)
            deg = torch.zeros(num_nodes, dtype=torch.long)
            deg.scatter_add_(0, train_edge[:, 0], torch.ones(train_edge.size(0), dtype=torch.long))
            deg.scatter_add_(0, train_edge[:, 1], torch.ones(train_edge.size(0), dtype=torch.long))
            _, top_nodes = torch.topk(deg, max_nodes)
            node_mask = torch.zeros(num_nodes, dtype=torch.bool)
            node_mask[top_nodes] = True

            # Remap nodes
            node_map = torch.full((num_nodes,), -1, dtype=torch.long)
            node_map[top_nodes] = torch.arange(max_nodes)

            # Filter edges
            def filter_edges(edges):
                mask = node_mask[edges[:, 0]] & node_mask[edges[:, 1]]
                filtered = edges[mask]
                return torch.stack([node_map[filtered[:, 0]], node_map[filtered[:, 1]]], dim=1)

            train_edge = filter_edges(train_edge)
            valid_edge = filter_edges(valid_edge)
            test_edge = filter_edges(test_edge)

            if x is not None:
                x = x[top_nodes]
            num_nodes = max_nodes

    # Limit training edges if needed
    if max_train_edges and len(train_edge) > max_train_edges:
        perm = torch.randperm(len(train_edge))[:max_train_edges]
        train_edge = train_edge[perm]

    # Build undirected training edge index for message passing
    train_ei = torch.stack([train_edge[:, 0], train_edge[:, 1]], dim=0)
    train_ei = to_undirected(train_ei)

    # Create node features if missing
    if x is None:
        deg = degree(train_ei[0], num_nodes=num_nodes) + degree(train_ei[1], num_nodes=num_nodes)
        # Simple degree + random features
        x = torch.zeros(num_nodes, 128)
        x[:, 0] = deg / deg.max()
        x[:, 1:] = torch.randn(num_nodes, 127) * 0.01

    # Negative sampling for val/test
    def sample_negatives(pos_edges, n_neg):
        neg_src = pos_edges[:n_neg, 0]
        neg_dst = torch.randint(0, num_nodes, (n_neg,))
        return torch.stack([neg_src, neg_dst], dim=1)

    valid_neg = sample_negatives(valid_edge, min(len(valid_edge), 50000))
    test_neg = sample_negatives(test_edge, min(len(test_edge), 50000))
    train_neg = sample_negatives(train_edge, min(len(train_edge), 100000))

    # Limit pos edges for val/test
    max_eval = 50000
    if len(valid_edge) > max_eval:
        valid_edge = valid_edge[torch.randperm(len(valid_edge))[:max_eval]]
        valid_neg = valid_neg[:max_eval]
    if len(test_edge) > max_eval:
        test_edge = test_edge[torch.randperm(len(test_edge))[:max_eval]]
        test_neg = test_neg[:max_eval]

    info = {
        "name": name,
        "num_nodes": num_nodes,
        "num_train_edges": train_ei.size(1) // 2,
        "num_features": x.size(1),
    }

    return {
        "x": x,
        "train_ei": train_ei,
        "train_pos": train_edge[:min(len(train_edge), 100000)],
        "train_neg": train_neg,
        "valid_pos": valid_edge,
        "valid_neg": valid_neg,
        "test_pos": test_edge,
        "test_neg": test_neg,
        "num_nodes": num_nodes,
        "info": info,
    }


# ============================================================
# Models (simplified for OGB scale)
# ============================================================

class SimpleLP(nn.Module):
    def __init__(self, in_ch, hidden=256, dropout=0.0):
        super().__init__()
        from torch_geometric.nn import SAGEConv
        self.conv1 = SAGEConv(in_ch, hidden)
        self.conv2 = SAGEConv(hidden, hidden)
        self.dropout = nn.Dropout(dropout)

    def encode(self, x, edge_index):
        h = self.conv1(x, edge_index).relu()
        h = self.dropout(h)
        h = self.conv2(h, edge_index)
        return h

    def decode(self, z, edges):
        src, dst = edges[:, 0], edges[:, 1]
        return (z[src] * z[dst]).sum(dim=-1)

    def forward(self, x, edge_index, edges):
        z = self.encode(x, edge_index)
        return self.decode(z, edges)


class SimpleCL(nn.Module):
    def __init__(self, in_ch, hidden=256, proj_dim=64, dropout=0.0,
                 edge_drop=0.3, feat_noise=0.1, temperature=0.5):
        super().__init__()
        from torch_geometric.nn import SAGEConv
        self.conv1 = SAGEConv(in_ch, hidden)
        self.conv2 = SAGEConv(hidden, hidden)
        self.dropout = nn.Dropout(dropout)
        self.projector = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, proj_dim))
        self.edge_drop = edge_drop
        self.feat_noise = feat_noise
        self.temperature = temperature

    def encode(self, x, edge_index):
        h = self.conv1(x, edge_index).relu()
        h = self.dropout(h)
        h = self.conv2(h, edge_index)
        return h

    def decode(self, z, edges):
        src, dst = edges[:, 0], edges[:, 1]
        return (z[src] * z[dst]).sum(dim=-1)

    def forward(self, x, edge_index, edges):
        z = self.encode(x, edge_index)
        return self.decode(z, edges)

    def augment(self, x, edge_index):
        mask = torch.bernoulli(torch.full((edge_index.size(1),),
                               1 - self.edge_drop, device=edge_index.device)).bool()
        aug_ei = edge_index[:, mask]
        aug_x = x + torch.randn_like(x) * self.feat_noise
        return aug_x, aug_ei

    def cl_loss(self, x, edge_index, max_nodes=1024):
        z1 = self.encode(x, edge_index)
        ax, aei = self.augment(x, edge_index)
        z2 = self.encode(ax, aei)
        h1 = self.projector(z1)
        h2 = self.projector(z2)
        if h1.size(0) > max_nodes:
            idx = torch.randperm(h1.size(0), device=h1.device)[:max_nodes]
            h1, h2 = h1[idx], h2[idx]
        h1 = F.normalize(h1, dim=1)
        h2 = F.normalize(h2, dim=1)
        sim = torch.mm(h1, h2.t()) / self.temperature
        labels = torch.arange(h1.size(0), device=sim.device)
        return F.cross_entropy(sim, labels)


# ============================================================
# Training
# ============================================================

def train_ogb(model, d, method="vanilla", cl_weight=0.5,
              epochs=200, lr=0.001, device="cuda"):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    x = d["x"].to(device)
    train_ei = d["train_ei"].to(device)
    train_pos = d["train_pos"].to(device)
    train_neg = d["train_neg"].to(device)

    best_val = 0
    best_state = None
    patience_count = 0

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        # LP loss on batch
        batch_size = min(32768, train_pos.size(0))
        perm = torch.randperm(train_pos.size(0), device=device)[:batch_size]
        perm_neg = torch.randperm(train_neg.size(0), device=device)[:batch_size]

        pos_out = model(x, train_ei, train_pos[perm])
        neg_out = model(x, train_ei, train_neg[perm_neg])

        out = torch.cat([pos_out, neg_out])
        labels = torch.cat([torch.ones_like(pos_out), torch.zeros_like(neg_out)])
        lp_loss = F.binary_cross_entropy_with_logits(out, labels)

        if method == "cl" and hasattr(model, "cl_loss"):
            cl_loss = model.cl_loss(x, train_ei)
            loss = lp_loss + cl_weight * cl_loss
        elif method == "augonly":
            # Train on augmented graph
            ax, aei = model.augment(x, train_ei)
            pos_out2 = model(ax, aei, train_pos[perm])
            neg_out2 = model(ax, aei, train_neg[perm_neg])
            out2 = torch.cat([pos_out2, neg_out2])
            aug_loss = F.binary_cross_entropy_with_logits(out2, labels)
            loss = (lp_loss + aug_loss) / 2
        else:
            loss = lp_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            val_auc = eval_ogb(model, d, "valid", device)
            if val_auc > best_val:
                best_val = val_auc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_count = 0
            else:
                patience_count += 1
            if patience_count >= 5:
                break

    if best_state:
        model.load_state_dict(best_state)
    return model, best_val


@torch.no_grad()
def eval_ogb(model, d, split="test", device="cuda"):
    model.eval()
    x = d["x"].to(device)
    train_ei = d["train_ei"].to(device)
    pos = d[f"{split}_pos"].to(device)
    neg = d[f"{split}_neg"].to(device)

    pos_out = torch.sigmoid(model(x, train_ei, pos)).cpu()
    neg_out = torch.sigmoid(model(x, train_ei, neg)).cpu()

    labels = torch.cat([torch.ones(pos_out.size(0)), torch.zeros(neg_out.size(0))]).numpy()
    preds = torch.cat([pos_out, neg_out]).numpy()

    if len(np.unique(labels)) <= 1:
        return 0.5
    return roc_auc_score(labels, preds)


@torch.no_grad()
def eval_ogb_by_degree(model, d, device="cuda"):
    """Degree-stratified evaluation."""
    model.eval()
    x = d["x"].to(device)
    train_ei = d["train_ei"].to(device)

    # Compute train-graph degree
    deg = degree(train_ei[0], num_nodes=d["num_nodes"])
    deg = deg + degree(train_ei[1], num_nodes=d["num_nodes"])
    deg = deg.cpu()

    pos = d["test_pos"].to(device)
    neg = d["test_neg"].to(device)

    pos_out = torch.sigmoid(model(x, train_ei, pos)).cpu().numpy()
    neg_out = torch.sigmoid(model(x, train_ei, neg)).cpu().numpy()

    all_edges = torch.cat([d["test_pos"], d["test_neg"]], dim=0)
    all_preds = np.concatenate([pos_out, neg_out])
    all_labels = np.concatenate([np.ones(len(pos_out)), np.zeros(len(neg_out))])

    min_deg = torch.minimum(deg[all_edges[:, 0]], deg[all_edges[:, 1]]).numpy()

    bins = [(0, 5), (5, 10), (10, 20), (20, 50), (50, 100), (100, 10000)]
    results = {}

    for lo, hi in bins:
        mask = (min_deg >= lo) & (min_deg < hi)
        n = mask.sum()
        if n < 20 or len(np.unique(all_labels[mask])) <= 1:
            continue
        auc = roc_auc_score(all_labels[mask], all_preds[mask])
        results[f"deg_{lo}-{hi}"] = {"auc": auc, "count": int(n)}

    overall = roc_auc_score(all_labels, all_preds)
    results["overall"] = {"auc": overall, "count": len(all_labels)}
    return results


# ============================================================
# Main
# ============================================================

def run_ogb_benchmark(name, max_nodes=50000, max_train_edges=200000, seeds=[0, 1, 2]):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'#'*70}")
    print(f"# OGB Benchmark: {name}")
    print(f"{'#'*70}")

    methods = ["vanilla", "cl", "augonly"]
    all_results = {}

    for method in methods:
        seed_results = []
        for seed in seeds:
            set_seed(seed)
            print(f"\n  [{name}/{method}/seed={seed}]", end=" ")

            d = prepare_ogb_data(name, max_nodes=max_nodes,
                                 max_train_edges=max_train_edges, seed=seed)
            info = d["info"]
            if seed == seeds[0] and method == methods[0]:
                print(f"\n    Nodes: {info['num_nodes']}, "
                      f"Train edges: {info['num_train_edges']}, "
                      f"Features: {info['num_features']}")

            in_ch = d["x"].size(1)
            t0 = time.time()

            if method == "vanilla":
                model = SimpleLP(in_ch, hidden=256)
                model, val_auc = train_ogb(model, d, method="vanilla", device=device)
            elif method == "cl":
                model = SimpleCL(in_ch, hidden=256)
                model, val_auc = train_ogb(model, d, method="cl", device=device)
            elif method == "augonly":
                model = SimpleCL(in_ch, hidden=256)
                model, val_auc = train_ogb(model, d, method="augonly", device=device)

            test_auc = eval_ogb(model, d, "test", device)
            deg_results = eval_ogb_by_degree(model, d, device)
            elapsed = time.time() - t0

            result = {"test_auc": test_auc, "val_auc": val_auc,
                      "by_degree": deg_results, "time_sec": elapsed}
            seed_results.append(result)

            # Print cold vs warm
            cold_auc = deg_results.get("deg_0-5", {}).get("auc", float("nan"))
            warm_auc = deg_results.get("deg_20-50", {}).get("auc", float("nan"))
            print(f"AUC={test_auc:.4f} cold={cold_auc:.4f} warm={warm_auc:.4f} ({elapsed:.1f}s)")

        # Aggregate
        test_aucs = [r["test_auc"] for r in seed_results]
        cold_aucs = [r["by_degree"].get("deg_0-5", {}).get("auc", float("nan")) for r in seed_results]
        cold_aucs = [x for x in cold_aucs if not np.isnan(x)]

        all_results[method] = {
            "per_seed": seed_results,
            "test_auc_mean": np.mean(test_aucs),
            "test_auc_std": np.std(test_aucs),
            "cold_auc_mean": np.mean(cold_aucs) if cold_aucs else float("nan"),
            "cold_auc_std": np.std(cold_aucs) if cold_aucs else float("nan"),
        }
        print(f"  >> {method}: AUC={np.mean(test_aucs):.4f}±{np.std(test_aucs):.4f}, "
              f"cold={np.mean(cold_aucs):.4f}±{np.std(cold_aucs):.4f}" if cold_aucs else "")

    return all_results


def main():
    all_results = {}

    # ogbl-collab: 235K nodes, 1.3M edges — manageable
    all_results["ogbl-collab"] = run_ogb_benchmark(
        "ogbl-collab", max_nodes=None, max_train_edges=300000, seeds=[0, 1, 2]
    )

    # ogbl-citation2: 2.9M nodes — need to subsample
    all_results["ogbl-citation2"] = run_ogb_benchmark(
        "ogbl-citation2", max_nodes=50000, max_train_edges=200000, seeds=[0, 1, 2]
    )

    # Print summary
    print(f"\n{'='*80}")
    print("OGB SUMMARY")
    print(f"{'='*80}")
    for ds, ds_res in all_results.items():
        print(f"\n--- {ds} ---")
        for method, m_res in ds_res.items():
            print(f"  {method:>10}: AUC={m_res['test_auc_mean']:.4f}±{m_res['test_auc_std']:.4f}, "
                  f"cold(0-5)={m_res['cold_auc_mean']:.4f}±{m_res['cold_auc_std']:.4f}")

        # Degree-stratified gain table
        if "vanilla" in ds_res and "cl" in ds_res:
            print(f"\n  Degree-stratified CL gain (seed 0):")
            v_deg = ds_res["vanilla"]["per_seed"][0]["by_degree"]
            c_deg = ds_res["cl"]["per_seed"][0]["by_degree"]
            a_deg = ds_res["augonly"]["per_seed"][0]["by_degree"]
            for key in sorted(v_deg.keys()):
                if key == "overall":
                    continue
                v = v_deg.get(key, {}).get("auc", float("nan"))
                c = c_deg.get(key, {}).get("auc", float("nan"))
                a = a_deg.get(key, {}).get("auc", float("nan"))
                n = v_deg.get(key, {}).get("count", 0)
                if not np.isnan(v) and not np.isnan(c):
                    print(f"    {key}: n={n}, vanilla={v:.4f}, CL={c:.4f}(Δ={c-v:+.4f}), "
                          f"AugOnly={a:.4f}(Δ={a-v:+.4f})")

    # Save
    output = "results/full/ogb_results.json"
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved to {output}")


if __name__ == "__main__":
    main()
