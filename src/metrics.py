"""
ColdCL: Evaluation metrics.

Includes standard LP metrics (AUC, AP) and calibration metrics (ECE, Brier),
all stratified by degree bins.
"""
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from src.data import get_edge_degree_bins


def hits_at_k(y_true, y_pred, k=20):
    """Hits@K: fraction of positive edges ranked in top-K among all candidates."""
    n_pos = int(y_true.sum())
    if n_pos == 0:
        return float("nan")
    # Rank all predictions descending
    sorted_idx = np.argsort(-y_pred)
    top_k_labels = y_true[sorted_idx[:k]]
    return float(top_k_labels.sum()) / min(n_pos, k)


def mrr(y_true, y_pred):
    """Mean Reciprocal Rank of positive edges."""
    n_pos = int(y_true.sum())
    if n_pos == 0:
        return float("nan")
    sorted_idx = np.argsort(-y_pred)
    ranks = np.where(y_true[sorted_idx] == 1)[0] + 1  # 1-indexed
    if len(ranks) == 0:
        return 0.0
    return float(np.mean(1.0 / ranks))


def expected_calibration_error(y_true, y_prob, n_bins=15):
    """Expected Calibration Error."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total = len(y_true)
    for i in range(n_bins):
        mask = (y_prob >= bin_boundaries[i]) & (y_prob < bin_boundaries[i + 1])
        if mask.sum() > 0:
            avg_confidence = y_prob[mask].mean()
            avg_accuracy = y_true[mask].mean()
            ece += mask.sum() * abs(avg_confidence - avg_accuracy)
    return ece / total if total > 0 else 0.0


def selective_auc(y_true, y_pred, uncertainty, coverages=(0.5, 0.7, 0.9, 1.0)):
    """AUC at different coverage levels (reject high-uncertainty predictions)."""
    results = {}
    sorted_idx = np.argsort(uncertainty)
    for cov in coverages:
        n_keep = max(10, int(len(y_true) * cov))
        sel_idx = sorted_idx[:n_keep]
        sel_true = y_true[sel_idx]
        sel_pred = y_pred[sel_idx]
        if len(np.unique(sel_true)) > 1:
            results[f"sel_auc@{cov}"] = roc_auc_score(sel_true, sel_pred)
        else:
            results[f"sel_auc@{cov}"] = float("nan")
    return results


@torch.no_grad()
def evaluate_lp(model, data, original_data, device="cuda", mc_samples=0):
    """
    Full LP evaluation with degree-stratified metrics.

    Args:
        model: trained LP model
        data: test split data
        original_data: original full graph (for degree computation)
        device: torch device
        mc_samples: if > 0, use MC dropout for uncertainty estimation

    Returns:
        dict with overall and per-bin metrics
    """
    model.eval()
    data_d = data.to(device)

    pos_edge = data_d.pos_edge_label_index
    neg_edge = data_d.neg_edge_label_index
    edge_label_index = torch.cat([pos_edge, neg_edge], dim=1)
    labels = torch.cat([
        torch.ones(pos_edge.size(1)),
        torch.zeros(neg_edge.size(1)),
    ]).cpu().numpy()

    # Get predictions
    if mc_samples > 0 and hasattr(model, "mc_predict"):
        mean_pred, std_pred = model.mc_predict(
            data_d.x, data_d.edge_index, edge_label_index, n_samples=mc_samples
        )
        pred = mean_pred.cpu().numpy()
        uncertainty = std_pred.cpu().numpy()
    else:
        out = model(data_d.x, data_d.edge_index, edge_label_index)
        pred = torch.sigmoid(out).cpu().numpy()
        uncertainty = None

    results = {}

    # Overall metrics
    overall_auc = roc_auc_score(labels, pred)
    overall_ap = average_precision_score(labels, pred)
    overall_ece = expected_calibration_error(labels, pred)
    overall_brier = brier_score_loss(labels, pred)

    results["overall"] = {
        "auc": overall_auc,
        "ap": overall_ap,
        "hits20": hits_at_k(labels, pred, k=20),
        "hits50": hits_at_k(labels, pred, k=50),
        "mrr": mrr(labels, pred),
        "ece": overall_ece,
        "brier": overall_brier,
        "count": len(labels),
    }

    if uncertainty is not None:
        results["overall"]["mean_uncertainty"] = float(uncertainty.mean())
        sel = selective_auc(labels, pred, uncertainty)
        results["overall"].update(sel)

    # Degree-binned metrics
    bins, deg = get_edge_degree_bins(original_data, edge_label_index.cpu())

    for bin_name, mask in bins.items():
        mask_np = mask.numpy()
        n = mask_np.sum()
        if n < 10:
            results[bin_name] = {"count": int(n), "auc": float("nan"), "ap": float("nan")}
            continue

        bin_labels = labels[mask_np]
        bin_pred = pred[mask_np]

        if len(np.unique(bin_labels)) <= 1:
            results[bin_name] = {"count": int(n), "auc": float("nan"), "ap": float("nan")}
            continue

        bin_auc = roc_auc_score(bin_labels, bin_pred)
        bin_ap = average_precision_score(bin_labels, bin_pred)
        bin_ece = expected_calibration_error(bin_labels, bin_pred)
        bin_brier = brier_score_loss(bin_labels, bin_pred)

        results[bin_name] = {
            "count": int(n),
            "auc": bin_auc,
            "ap": bin_ap,
            "hits20": hits_at_k(bin_labels, bin_pred, k=20),
            "hits50": hits_at_k(bin_labels, bin_pred, k=50),
            "ece": bin_ece,
            "brier": bin_brier,
        }

        if uncertainty is not None:
            bin_unc = uncertainty[mask_np]
            results[bin_name]["mean_uncertainty"] = float(bin_unc.mean())

    return results


def format_results(results, name=""):
    """Pretty-print evaluation results."""
    lines = [f"\n{'='*60}", f"  {name}", f"{'='*60}"]

    overall = results.get("overall", {})
    lines.append(f"  Overall — AUC: {overall.get('auc', 0):.4f}, "
                 f"AP: {overall.get('ap', 0):.4f}, "
                 f"ECE: {overall.get('ece', 0):.4f}, "
                 f"Brier: {overall.get('brier', 0):.4f}")

    if "mean_uncertainty" in overall:
        lines.append(f"  Mean uncertainty: {overall['mean_uncertainty']:.4f}")

    for key in ["sel_auc@0.5", "sel_auc@0.7", "sel_auc@0.9", "sel_auc@1.0"]:
        if key in overall:
            lines.append(f"  {key}: {overall[key]:.4f}")

    for bin_name in ["isolated (0-1)", "cold (2-5)", "warm (6-20)", "hot (>20)"]:
        if bin_name in results:
            b = results[bin_name]
            auc_str = f"{b['auc']:.4f}" if not np.isnan(b.get('auc', float('nan'))) else "N/A"
            ap_str = f"{b['ap']:.4f}" if not np.isnan(b.get('ap', float('nan'))) else "N/A"
            unc_str = f", unc={b['mean_uncertainty']:.4f}" if "mean_uncertainty" in b else ""
            lines.append(f"  {bin_name} (n={b['count']}): AUC={auc_str}, AP={ap_str}{unc_str}")

    return "\n".join(lines)
