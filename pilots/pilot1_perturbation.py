"""
Pilot 1: Relative Perturbation Law (Diagnostic)

Hypothesis: If we force high-degree nodes to lose the same FRACTION of local
structural evidence as low-degree nodes, the LP performance gap should shrink.
If it doesn't, degree has an independent effect beyond relative evidence loss.

Method: For high-degree nodes, randomly mask additional non-target edges until
their fractional evidence loss matches that of low-degree nodes.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import torch
import numpy as np
from torch_geometric.utils import degree
from common import (
    set_seed, load_dataset, GCNLinkPredictor, SAGELinkPredictor,
    train_lp_model, eval_lp_by_degree, save_results
)


def compute_relative_evidence_loss(train_data, original_data):
    """Compute fractional evidence loss per node from edge splitting."""
    # Ensure everything on CPU for this computation
    orig_ei = original_data.edge_index.cpu()
    train_ei = train_data.edge_index.cpu()

    orig_deg = degree(orig_ei[0], num_nodes=original_data.num_nodes)
    orig_deg = orig_deg + degree(orig_ei[1], num_nodes=original_data.num_nodes)

    train_deg = degree(train_ei[0], num_nodes=train_data.num_nodes)
    train_deg = train_deg + degree(train_ei[1], num_nodes=train_data.num_nodes)

    frac_loss = (orig_deg - train_deg) / (orig_deg + 1e-8)
    return frac_loss, orig_deg, train_deg


def equalize_evidence(train_data, original_data, target_frac=None):
    """Mask extra edges from high-degree nodes to equalize fractional loss."""
    frac_loss, orig_deg, train_deg = compute_relative_evidence_loss(train_data, original_data)

    # Target: median fractional loss of low-degree nodes (deg <= 5)
    low_deg_mask = orig_deg <= 5
    if target_frac is None:
        if low_deg_mask.sum() > 0:
            target_frac = frac_loss[low_deg_mask].median().item()
        else:
            target_frac = frac_loss.median().item()

    print(f"  Target fractional evidence loss: {target_frac:.3f}")
    print(f"  Low-deg mean frac loss: {frac_loss[low_deg_mask].mean():.3f}")
    high_deg_mask = orig_deg > 10
    print(f"  High-deg mean frac loss (before): {frac_loss[high_deg_mask].mean():.3f}")

    # For each high-degree node, compute how many extra edges to mask
    edge_index = train_data.edge_index.cpu().clone()
    edges_to_keep = torch.ones(edge_index.size(1), dtype=torch.bool)

    for node_idx in torch.where(high_deg_mask)[0]:
        current_frac = frac_loss[node_idx].item()
        if current_frac >= target_frac:
            continue

        orig_d = orig_deg[node_idx].item()
        # target_frac = (orig_d - new_d) / orig_d => new_d = orig_d * (1 - target_frac)
        target_deg = max(1, int(orig_d * (1 - target_frac)))
        current_deg = int(train_deg[node_idx].item())
        edges_to_drop = current_deg - target_deg

        if edges_to_drop <= 0:
            continue

        # Find edges incident to this node
        incident = (edge_index[0] == node_idx) | (edge_index[1] == node_idx)
        incident_indices = torch.where(incident & edges_to_keep)[0]

        if len(incident_indices) > 1:  # keep at least 1 edge
            n_drop = min(edges_to_drop, len(incident_indices) - 1)
            perm = torch.randperm(len(incident_indices))[:n_drop]
            edges_to_keep[incident_indices[perm]] = False

    new_edge_index = edge_index[:, edges_to_keep]

    # Recompute stats
    new_train_deg = degree(new_edge_index[0], num_nodes=original_data.num_nodes)
    new_train_deg = new_train_deg + degree(new_edge_index[1], num_nodes=original_data.num_nodes)
    new_frac_loss = (orig_deg - new_train_deg) / (orig_deg + 1e-8)
    print(f"  High-deg mean frac loss (after): {new_frac_loss[high_deg_mask].mean():.3f}")
    print(f"  Edges: {edge_index.size(1)} -> {new_edge_index.size(1)}")

    return new_edge_index


def run_pilot(dataset_name="Cora", seed=42):
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*60}")
    print(f"Pilot 1: Relative Perturbation Law — {dataset_name}")
    print(f"{'='*60}")

    data, train_data, val_data, test_data = load_dataset(dataset_name)
    in_channels = data.num_features

    results = {"dataset": dataset_name, "seed": seed}

    # === Baseline: standard training ===
    print("\n[Baseline] Standard GCN training...")
    model = GCNLinkPredictor(in_channels)
    model, val_auc = train_lp_model(model, train_data, val_data, device=device)
    baseline_results = eval_lp_by_degree(model, test_data, data, device)
    results["baseline"] = baseline_results
    print(f"  Overall AUC: {baseline_results['overall']['auc']:.4f}")
    for k, v in baseline_results.items():
        if k != "overall":
            print(f"  {k}: AUC={v['auc']:.4f} (n={v['count']})")

    # === Treatment: equalized evidence ===
    print("\n[Treatment] Equalizing evidence loss across degree bins...")
    frac_loss, orig_deg, train_deg = compute_relative_evidence_loss(train_data, data)
    new_edge_index = equalize_evidence(train_data, data)

    # Create modified training data
    train_data_eq = train_data.clone()
    train_data_eq.edge_index = new_edge_index

    model_eq = GCNLinkPredictor(in_channels)
    model_eq, val_auc_eq = train_lp_model(model_eq, train_data_eq, val_data, device=device)
    eq_results = eval_lp_by_degree(model_eq, test_data, data, device)
    results["equalized"] = eq_results
    print(f"  Overall AUC: {eq_results['overall']['auc']:.4f}")
    for k, v in eq_results.items():
        if k != "overall":
            print(f"  {k}: AUC={v['auc']:.4f} (n={v['count']})")

    # === Analysis: gap change ===
    print("\n[Analysis]")
    for bin_name in ["cold (2-5)", "warm (6-20)", "hot (>20)"]:
        if bin_name in baseline_results and bin_name in eq_results:
            b = baseline_results[bin_name]["auc"]
            e = eq_results[bin_name]["auc"]
            if not (np.isnan(b) or np.isnan(e)):
                print(f"  {bin_name}: {b:.4f} -> {e:.4f} (Δ={e-b:+.4f})")

    # Gap between cold and hot
    for label, res in [("Baseline", baseline_results), ("Equalized", eq_results)]:
        cold_auc = res.get("cold (2-5)", {}).get("auc", float("nan"))
        hot_auc = res.get("hot (>20)", {}).get("auc", float("nan"))
        if not (np.isnan(cold_auc) or np.isnan(hot_auc)):
            print(f"  {label} cold-hot gap: {hot_auc - cold_auc:.4f}")

    results["signal"] = "POSITIVE" if (
        eq_results.get("cold (2-5)", {}).get("auc", 0) > baseline_results.get("cold (2-5)", {}).get("auc", 0)
        or (baseline_results.get("hot (>20)", {}).get("auc", 0) - baseline_results.get("cold (2-5)", {}).get("auc", 0)) >
           (eq_results.get("hot (>20)", {}).get("auc", 0) - eq_results.get("cold (2-5)", {}).get("auc", 0))
    ) else "NEGATIVE"

    print(f"\n  Signal: {results['signal']}")
    return results


if __name__ == "__main__":
    all_results = {}
    for ds in ["Cora", "CiteSeer"]:
        all_results[ds] = run_pilot(ds)
    save_results(all_results, "pilot1_relative_perturbation.json")
