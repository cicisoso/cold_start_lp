"""
Pilot 5: Cold-Only Contrastive Learning (Method)

Hypothesis: Applying contrastive augmentation ONLY to low-degree ego-nets
(not globally) should improve cold-start LP without degrading dense regions.

Method: Degree-gated edge-drop + feature perturbation around low-degree nodes,
with a consistency loss on embeddings.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.nn.functional as F
import numpy as np
from torch_geometric.utils import degree, subgraph
from common import (
    set_seed, load_dataset, GCNLinkPredictor,
    train_lp_model, eval_lp_by_degree, save_results
)


def cold_only_augment(x, edge_index, num_nodes, deg_threshold=5, edge_drop_rate=0.3, feat_noise=0.1):
    """Augment ONLY the ego-nets of low-degree nodes."""
    deg = degree(edge_index[0], num_nodes=num_nodes) + degree(edge_index[1], num_nodes=num_nodes)
    cold_nodes = torch.where(deg <= deg_threshold)[0]

    if len(cold_nodes) == 0:
        return x.clone(), edge_index.clone()

    # Identify edges incident to cold nodes
    cold_set = set(cold_nodes.tolist())
    src, dst = edge_index
    incident_mask = torch.zeros(edge_index.size(1), dtype=torch.bool)
    for i in range(edge_index.size(1)):
        if src[i].item() in cold_set or dst[i].item() in cold_set:
            incident_mask[i] = True

    # Edge dropout: only drop cold-incident edges
    keep_mask = torch.ones(edge_index.size(1), dtype=torch.bool)
    cold_edges = torch.where(incident_mask)[0]
    if len(cold_edges) > 0:
        n_drop = int(len(cold_edges) * edge_drop_rate)
        perm = torch.randperm(len(cold_edges))[:n_drop]
        keep_mask[cold_edges[perm]] = False

    aug_edge_index = edge_index[:, keep_mask]

    # Feature noise: only perturb cold nodes
    aug_x = x.clone()
    noise = torch.randn_like(aug_x[cold_nodes]) * feat_noise
    aug_x[cold_nodes] = aug_x[cold_nodes] + noise

    return aug_x, aug_edge_index


def global_augment(x, edge_index, edge_drop_rate=0.3, feat_noise=0.1):
    """Standard global augmentation (baseline)."""
    n_edges = edge_index.size(1)
    n_drop = int(n_edges * edge_drop_rate)
    keep_mask = torch.ones(n_edges, dtype=torch.bool)
    perm = torch.randperm(n_edges)[:n_drop]
    keep_mask[perm] = False
    aug_edge_index = edge_index[:, keep_mask]

    aug_x = x + torch.randn_like(x) * feat_noise
    return aug_x, aug_edge_index


class CLLinkPredictor(torch.nn.Module):
    """GCN LP model with optional contrastive loss."""

    def __init__(self, in_channels, hidden_channels=128):
        super().__init__()
        self.encoder = GCNLinkPredictor(in_channels, hidden_channels)
        self.projector = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, 64),
        )

    def encode(self, x, edge_index):
        return self.encoder.encode(x, edge_index)

    def decode(self, z, edge_label_index):
        return self.encoder.decode(z, edge_label_index)

    def forward(self, x, edge_index, edge_label_index):
        z = self.encode(x, edge_index)
        return self.decode(z, edge_label_index)

    def contrastive_loss(self, z1, z2, node_mask=None, temperature=0.5):
        """InfoNCE loss on selected nodes."""
        h1 = self.projector(z1)
        h2 = self.projector(z2)

        if node_mask is not None and node_mask.sum() > 0:
            h1 = h1[node_mask]
            h2 = h2[node_mask]

        if h1.size(0) == 0:
            return torch.tensor(0.0, device=z1.device)

        # Limit to avoid memory issues
        if h1.size(0) > 512:
            idx = torch.randperm(h1.size(0))[:512]
            h1 = h1[idx]
            h2 = h2[idx]

        h1 = F.normalize(h1, dim=1)
        h2 = F.normalize(h2, dim=1)

        sim = torch.mm(h1, h2.t()) / temperature
        labels = torch.arange(h1.size(0), device=sim.device)
        loss = F.cross_entropy(sim, labels)
        return loss


def train_with_cl(model, train_data, val_data, augment_fn, cl_weight=0.5,
                  epochs=200, lr=0.01, device="cuda"):
    """Training with contrastive loss."""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    train_data = train_data.to(device)
    val_data = val_data.to(device)

    deg = degree(train_data.edge_index[0], num_nodes=train_data.num_nodes)
    deg = deg + degree(train_data.edge_index[1], num_nodes=train_data.num_nodes)
    cold_mask = deg <= 5

    best_val_auc = 0
    best_state = None
    patience = 30
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        # LP loss
        pos_edge = train_data.pos_edge_label_index
        neg_edge = train_data.neg_edge_label_index
        edge_label_index = torch.cat([pos_edge, neg_edge], dim=1)
        labels = torch.cat([
            torch.ones(pos_edge.size(1)),
            torch.zeros(neg_edge.size(1)),
        ]).to(device)

        out = model(train_data.x, train_data.edge_index, edge_label_index)
        lp_loss = F.binary_cross_entropy_with_logits(out, labels)

        # Contrastive loss
        z_orig = model.encode(train_data.x, train_data.edge_index)
        aug_x, aug_ei = augment_fn(
            train_data.x, train_data.edge_index, train_data.num_nodes
        ) if augment_fn == cold_only_augment else augment_fn(
            train_data.x, train_data.edge_index
        )
        aug_x, aug_ei = aug_x.to(device), aug_ei.to(device)
        z_aug = model.encode(aug_x, aug_ei)

        cl_loss = model.contrastive_loss(z_orig, z_aug, cold_mask.to(device))
        loss = lp_loss + cl_weight * cl_loss
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            from common import eval_lp
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


def run_pilot(dataset_name="Cora", seed=42):
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*60}")
    print(f"Pilot 5: Cold-Only Contrastive Learning — {dataset_name}")
    print(f"{'='*60}")

    data, train_data, val_data, test_data = load_dataset(dataset_name)
    in_channels = data.num_features
    results = {"dataset": dataset_name, "seed": seed}

    # === No CL baseline ===
    print("\n[A] No CL baseline...")
    model_a = GCNLinkPredictor(in_channels)
    model_a, _ = train_lp_model(model_a, train_data, val_data, device=device)
    res_a = eval_lp_by_degree(model_a, test_data, data, device)
    results["no_cl"] = res_a
    print(f"  Overall: {res_a['overall']['auc']:.4f}")
    for k, v in res_a.items():
        if k != "overall":
            print(f"  {k}: {v['auc']:.4f} (n={v['count']})")

    # === Global CL ===
    print("\n[B] Global CL...")
    model_b = CLLinkPredictor(in_channels)
    model_b, _ = train_with_cl(model_b, train_data, val_data,
                                augment_fn=global_augment, device=device)
    res_b = eval_lp_by_degree(model_b, test_data, data, device)
    results["global_cl"] = res_b
    print(f"  Overall: {res_b['overall']['auc']:.4f}")
    for k, v in res_b.items():
        if k != "overall":
            print(f"  {k}: {v['auc']:.4f} (n={v['count']})")

    # === Cold-Only CL (ours) ===
    print("\n[C] Cold-Only CL (ours)...")
    model_c = CLLinkPredictor(in_channels)

    def cold_aug_wrapper(x, edge_index, num_nodes=None):
        if num_nodes is None:
            num_nodes = x.size(0)
        return cold_only_augment(x, edge_index, num_nodes)

    model_c, _ = train_with_cl(model_c, train_data, val_data,
                                augment_fn=cold_aug_wrapper, device=device)
    res_c = eval_lp_by_degree(model_c, test_data, data, device)
    results["cold_only_cl"] = res_c
    print(f"  Overall: {res_c['overall']['auc']:.4f}")
    for k, v in res_c.items():
        if k != "overall":
            print(f"  {k}: {v['auc']:.4f} (n={v['count']})")

    # === Analysis ===
    print("\n[Analysis] Cold-bin improvement:")
    for bin_name in ["isolated (0-1)", "cold (2-5)"]:
        a = res_a.get(bin_name, {}).get("auc", float("nan"))
        b = res_b.get(bin_name, {}).get("auc", float("nan"))
        c = res_c.get(bin_name, {}).get("auc", float("nan"))
        if not np.isnan(c):
            print(f"  {bin_name}: NoCL={a:.4f}, GlobalCL={b:.4f}, ColdCL={c:.4f}")

    # Check if cold-only CL improves cold without hurting warm
    cold_improved = (
        res_c.get("cold (2-5)", {}).get("auc", 0) >
        res_a.get("cold (2-5)", {}).get("auc", 0)
    )
    warm_maintained = (
        res_c.get("warm (6-20)", {}).get("auc", 0) >=
        res_a.get("warm (6-20)", {}).get("auc", 0) - 0.02
    )
    results["signal"] = "POSITIVE" if (cold_improved and warm_maintained) else (
        "WEAK POSITIVE" if cold_improved else "NEGATIVE"
    )
    print(f"\n  Signal: {results['signal']}")
    return results


if __name__ == "__main__":
    all_results = {}
    for ds in ["Cora", "CiteSeer"]:
        all_results[ds] = run_pilot(ds)
    save_results(all_results, "pilot5_cold_only_cl.json")
