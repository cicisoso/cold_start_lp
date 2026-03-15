"""
Pilot 8: Calibrated Cold-Start LP (Empirical)

Hypothesis: A large part of cold-start LP failure is overconfidence.
Uncertainty-aware prediction (MC dropout) should outperform raw ranking
on cold regions under risk-coverage evaluation.

Method: Add MC dropout on top of a fixed LP backbone and measure
ECE, Brier score, and selective prediction quality by degree bin.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.nn.functional as F
import numpy as np
from torch_geometric.utils import degree
from sklearn.metrics import roc_auc_score, brier_score_loss
from common import (
    set_seed, load_dataset, GCNLinkPredictor,
    train_lp_model, get_degree_bins, save_results
)


class MCDropoutLinkPredictor(torch.nn.Module):
    """GCN LP with MC dropout for uncertainty estimation."""

    def __init__(self, in_channels, hidden_channels=128, dropout=0.3):
        super().__init__()
        self.encoder = GCNLinkPredictor(in_channels, hidden_channels)
        self.dropout = torch.nn.Dropout(p=dropout)
        self.drop_rate = dropout

    def encode(self, x, edge_index):
        h = self.encoder.conv1(x, edge_index).relu()
        h = self.dropout(h)
        h = self.encoder.conv2(h, edge_index)
        return h

    def decode(self, z, edge_label_index):
        return self.encoder.decode(z, edge_label_index)

    def forward(self, x, edge_index, edge_label_index):
        z = self.encode(x, edge_index)
        return self.decode(z, edge_label_index)

    def mc_predict(self, x, edge_index, edge_label_index, n_samples=30):
        """MC dropout prediction: run n_samples forward passes with dropout enabled."""
        self.train()  # Keep dropout active
        preds = []
        for _ in range(n_samples):
            with torch.no_grad():
                out = self.forward(x, edge_index, edge_label_index)
                preds.append(torch.sigmoid(out))
        preds = torch.stack(preds)
        mean_pred = preds.mean(dim=0)
        std_pred = preds.std(dim=0)  # Uncertainty
        return mean_pred, std_pred


def expected_calibration_error(y_true, y_prob, n_bins=10):
    """Compute ECE."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (y_prob >= bin_boundaries[i]) & (y_prob < bin_boundaries[i + 1])
        if mask.sum() > 0:
            avg_confidence = y_prob[mask].mean()
            avg_accuracy = y_true[mask].mean()
            ece += mask.sum() * abs(avg_confidence - avg_accuracy)
    return ece / len(y_true)


def selective_auc(y_true, y_pred, uncertainty, coverage_levels=[0.5, 0.7, 0.9, 1.0]):
    """AUC at different coverage levels (reject uncertain predictions)."""
    results = {}
    for cov in coverage_levels:
        n_keep = int(len(y_true) * cov)
        if n_keep < 10:
            results[f"cov_{cov}"] = float("nan")
            continue
        # Keep lowest-uncertainty predictions
        sorted_idx = np.argsort(uncertainty)[:n_keep]
        sel_true = y_true[sorted_idx]
        sel_pred = y_pred[sorted_idx]
        if len(np.unique(sel_true)) > 1:
            results[f"cov_{cov}"] = roc_auc_score(sel_true, sel_pred)
        else:
            results[f"cov_{cov}"] = float("nan")
    return results


def run_pilot(dataset_name="Cora", seed=42):
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*60}")
    print(f"Pilot 8: Calibrated Cold-Start LP — {dataset_name}")
    print(f"{'='*60}")

    data, train_data, val_data, test_data = load_dataset(dataset_name)
    in_channels = data.num_features
    results = {"dataset": dataset_name, "seed": seed}

    # Train model with MC dropout
    print("\n[1] Training MC Dropout model...")
    model = MCDropoutLinkPredictor(in_channels, dropout=0.3)
    model, val_auc = train_lp_model(model, train_data, val_data, device=device)
    print(f"  Val AUC: {val_auc:.4f}")

    # Get test predictions with uncertainty
    print("\n[2] MC Dropout inference (30 samples)...")
    test_data_d = test_data.to(device)
    pos_edge = test_data_d.pos_edge_label_index
    neg_edge = test_data_d.neg_edge_label_index
    edge_label_index = torch.cat([pos_edge, neg_edge], dim=1)
    labels = torch.cat([
        torch.ones(pos_edge.size(1)),
        torch.zeros(neg_edge.size(1)),
    ]).cpu().numpy()

    mean_pred, std_pred = model.mc_predict(
        test_data_d.x, test_data_d.edge_index, edge_label_index, n_samples=30
    )
    mean_pred = mean_pred.cpu().numpy()
    std_pred = std_pred.cpu().numpy()

    # Also get deterministic prediction
    model.eval()
    with torch.no_grad():
        det_pred = torch.sigmoid(model(
            test_data_d.x, test_data_d.edge_index, edge_label_index
        )).cpu().numpy()

    # Overall metrics
    print("\n[3] Overall metrics:")
    det_auc = roc_auc_score(labels, det_pred)
    mc_auc = roc_auc_score(labels, mean_pred)
    det_ece = expected_calibration_error(labels, det_pred)
    mc_ece = expected_calibration_error(labels, mean_pred)
    det_brier = brier_score_loss(labels, det_pred)
    mc_brier = brier_score_loss(labels, mean_pred)

    print(f"  Deterministic — AUC: {det_auc:.4f}, ECE: {det_ece:.4f}, Brier: {det_brier:.4f}")
    print(f"  MC Dropout    — AUC: {mc_auc:.4f}, ECE: {mc_ece:.4f}, Brier: {mc_brier:.4f}")

    results["overall"] = {
        "det": {"auc": det_auc, "ece": det_ece, "brier": det_brier},
        "mc": {"auc": mc_auc, "ece": mc_ece, "brier": mc_brier},
    }

    # Degree-binned analysis
    print("\n[4] Degree-binned analysis:")
    bins, deg = get_degree_bins(data, edge_label_index.cpu())
    results["by_degree"] = {}

    for bin_name, mask in bins.items():
        mask = mask.numpy()
        if mask.sum() < 10:
            print(f"  {bin_name}: too few samples ({mask.sum()})")
            continue

        bin_labels = labels[mask]
        bin_det = det_pred[mask]
        bin_mc = mean_pred[mask]
        bin_std = std_pred[mask]

        if len(np.unique(bin_labels)) <= 1:
            print(f"  {bin_name}: single class, skip")
            continue

        bin_det_auc = roc_auc_score(bin_labels, bin_det)
        bin_mc_auc = roc_auc_score(bin_labels, bin_mc)
        bin_det_ece = expected_calibration_error(bin_labels, bin_det)
        bin_mc_ece = expected_calibration_error(bin_labels, bin_mc)
        bin_mean_unc = bin_std.mean()

        print(f"  {bin_name} (n={mask.sum()}):")
        print(f"    Det AUC={bin_det_auc:.4f}, ECE={bin_det_ece:.4f}")
        print(f"    MC  AUC={bin_mc_auc:.4f}, ECE={bin_mc_ece:.4f}, mean_unc={bin_mean_unc:.4f}")

        results["by_degree"][bin_name] = {
            "count": int(mask.sum()),
            "det_auc": bin_det_auc, "det_ece": bin_det_ece,
            "mc_auc": bin_mc_auc, "mc_ece": bin_mc_ece,
            "mean_uncertainty": float(bin_mean_unc),
        }

    # Selective prediction
    print("\n[5] Selective prediction (reject uncertain):")
    sel_results = selective_auc(labels, mean_pred, std_pred)
    results["selective"] = sel_results
    for k, v in sel_results.items():
        print(f"  {k}: AUC={v:.4f}")

    # Signal assessment
    cold_ece_improved = (
        results.get("by_degree", {}).get("cold (2-5)", {}).get("mc_ece", 1.0) <
        results.get("by_degree", {}).get("cold (2-5)", {}).get("det_ece", 1.0)
    )
    cold_more_uncertain = (
        results.get("by_degree", {}).get("cold (2-5)", {}).get("mean_uncertainty", 0) >
        results.get("by_degree", {}).get("warm (6-20)", {}).get("mean_uncertainty", 0)
    )
    selective_helps = sel_results.get("cov_0.7", 0) > sel_results.get("cov_1.0", 0)

    if cold_ece_improved and cold_more_uncertain and selective_helps:
        results["signal"] = "POSITIVE"
    elif cold_more_uncertain or selective_helps:
        results["signal"] = "WEAK POSITIVE"
    else:
        results["signal"] = "NEGATIVE"

    print(f"\n  Cold more uncertain: {cold_more_uncertain}")
    print(f"  Cold ECE improved by MC: {cold_ece_improved}")
    print(f"  Selective prediction helps: {selective_helps}")
    print(f"  Signal: {results['signal']}")
    return results


if __name__ == "__main__":
    all_results = {}
    for ds in ["Cora", "CiteSeer"]:
        all_results[ds] = run_pilot(ds)
    save_results(all_results, "pilot8_calibration.json")
