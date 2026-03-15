"""
Mechanism Analysis: Why CL helps tail-node LP.

Measures:
1. Embedding stability (variance across seeds/augmentations) by degree bin
2. CL gain as a function of node degree (fine-grained)
3. Augmentation-only vs CL-only ablation
"""
import os
import sys
import json
import random
import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))
from src.data import load_dataset, get_node_degrees
from src.models import ENCODERS, GCNEncoder, LinkPredictor, GlobalCLModel
from src.train import train_standard, train_with_cl
from sklearn.metrics import roc_auc_score
from torch_geometric.utils import degree


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def measure_embedding_stability(model, data, n_runs=10, device="cuda"):
    """
    Measure embedding variance per node across multiple forward passes
    with dropout enabled (or small random perturbations).
    Returns per-node embedding std.
    """
    model.train()  # Enable dropout
    data = data.to(device)
    embeddings = []

    for _ in range(n_runs):
        with torch.no_grad():
            z = model.encode(data.x, data.edge_index)
            embeddings.append(z.cpu())

    embeddings = torch.stack(embeddings)  # [n_runs, num_nodes, hidden]
    per_node_std = embeddings.std(dim=0).mean(dim=1)  # [num_nodes]
    return per_node_std.numpy()


def degree_stratified_analysis(model_vanilla, model_cl, data, test_data, device="cuda"):
    """
    Fine-grained degree-stratified AUC comparison.
    Returns per-degree-bin AUC for both models.
    """
    test_data_d = test_data.to(device)
    pos_edge = test_data_d.pos_edge_label_index
    neg_edge = test_data_d.neg_edge_label_index
    edge_label_index = torch.cat([pos_edge, neg_edge], dim=1)
    labels = torch.cat([
        torch.ones(pos_edge.size(1)),
        torch.zeros(neg_edge.size(1)),
    ]).cpu().numpy()

    # Get predictions from both models
    results = {}
    for name, model in [("vanilla", model_vanilla), ("cl", model_cl)]:
        model.eval()
        model = model.to(device)
        with torch.no_grad():
            out = model(test_data_d.x, test_data_d.edge_index, edge_label_index)
            pred = torch.sigmoid(out).cpu().numpy()
        results[name] = pred

    # Compute degree on training graph
    train_deg = degree(data.edge_index[0], num_nodes=data.num_nodes)
    train_deg = train_deg + degree(data.edge_index[1], num_nodes=data.num_nodes)

    src, dst = edge_label_index.cpu()
    min_deg = torch.minimum(train_deg[src], train_deg[dst]).numpy()

    # Fine-grained bins
    bins = [(0, 2), (2, 4), (4, 6), (6, 10), (10, 20), (20, 50), (50, 1000)]
    bin_results = []

    for lo, hi in bins:
        mask = (min_deg >= lo) & (min_deg < hi)
        n = mask.sum()
        if n < 20:
            continue
        bin_labels = labels[mask]
        if len(np.unique(bin_labels)) <= 1:
            continue

        vanilla_auc = roc_auc_score(bin_labels, results["vanilla"][mask])
        cl_auc = roc_auc_score(bin_labels, results["cl"][mask])
        gain = cl_auc - vanilla_auc

        bin_results.append({
            "degree_range": f"{lo}-{hi}",
            "count": int(n),
            "vanilla_auc": vanilla_auc,
            "cl_auc": cl_auc,
            "gain": gain,
        })

    return bin_results


def augmentation_only_train(model, train_data, val_data, edge_drop_rate=0.3,
                            feat_noise_std=0.1, epochs=300, lr=0.01,
                            weight_decay=5e-4, patience=50, device="cuda"):
    """
    Train with augmented graph (edge drop + noise) but NO contrastive loss.
    Tests if augmentation alone provides the benefit.
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    train_data = train_data.to(device)
    val_data = val_data.to(device)

    best_val_auc = 0
    best_state = None
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        # Apply augmentation to graph
        keep_prob = 1.0 - edge_drop_rate
        mask = torch.bernoulli(torch.full((train_data.edge_index.size(1),), keep_prob,
                                          device=device)).bool()
        aug_ei = train_data.edge_index[:, mask]
        aug_x = train_data.x + torch.randn_like(train_data.x) * feat_noise_std

        pos_edge = train_data.pos_edge_label_index
        neg_edge = train_data.neg_edge_label_index
        edge_label_index = torch.cat([pos_edge, neg_edge], dim=1)
        labels = torch.cat([
            torch.ones(pos_edge.size(1)),
            torch.zeros(neg_edge.size(1)),
        ]).to(device)

        out = model(aug_x, aug_ei, edge_label_index)
        loss = F.binary_cross_entropy_with_logits(out, labels)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            val_auc = quick_eval(model, val_data, device)
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


def quick_eval(model, data, device):
    model.eval()
    data = data.to(device)
    with torch.no_grad():
        pos_edge = data.pos_edge_label_index
        neg_edge = data.neg_edge_label_index
        edge_label_index = torch.cat([pos_edge, neg_edge], dim=1)
        labels = torch.cat([torch.ones(pos_edge.size(1)), torch.zeros(neg_edge.size(1))])
        out = model(data.x, data.edge_index, edge_label_index)
        pred = torch.sigmoid(out).cpu().numpy()
        return roc_auc_score(labels.numpy(), pred)


def eval_by_cold_warm(model, test_data, data, device="cuda"):
    model.eval()
    test_data_d = test_data.to(device)
    pos_edge = test_data_d.pos_edge_label_index
    neg_edge = test_data_d.neg_edge_label_index
    edge_label_index = torch.cat([pos_edge, neg_edge], dim=1)
    labels = torch.cat([
        torch.ones(pos_edge.size(1)),
        torch.zeros(neg_edge.size(1)),
    ]).cpu().numpy()

    with torch.no_grad():
        out = model(test_data_d.x, test_data_d.edge_index, edge_label_index)
        pred = torch.sigmoid(out).cpu().numpy()

    train_deg = degree(data.edge_index[0], num_nodes=data.num_nodes)
    train_deg = train_deg + degree(data.edge_index[1], num_nodes=data.num_nodes)
    src, dst = edge_label_index.cpu()
    min_deg = torch.minimum(train_deg[src], train_deg[dst]).numpy()

    cold = min_deg <= 5
    warm = min_deg > 5

    overall = roc_auc_score(labels, pred)
    cold_auc = roc_auc_score(labels[cold], pred[cold]) if cold.sum() > 10 and len(np.unique(labels[cold])) > 1 else float("nan")
    warm_auc = roc_auc_score(labels[warm], pred[warm]) if warm.sum() > 10 and len(np.unique(labels[warm])) > 1 else float("nan")

    return {"overall": overall, "cold": cold_auc, "warm": warm_auc}


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    all_analysis = {}

    for ds_name in ["Cora", "CiteSeer", "PubMed", "CS"]:
        print(f"\n{'#'*60}\n# {ds_name} — Mechanism Analysis\n{'#'*60}")
        set_seed(42)
        data, train_data, val_data, test_data = load_dataset(ds_name, seed=42)
        in_ch = data.num_features
        all_analysis[ds_name] = {}

        # Train vanilla and CL models
        print("\n  Training Vanilla...")
        enc_v = GCNEncoder(in_ch, 128, dropout=0.2)
        model_v = LinkPredictor(enc_v)
        model_v, _ = train_standard(model_v, train_data, val_data, device=device)

        print("  Training GlobalCL...")
        enc_c = GCNEncoder(in_ch, 128, dropout=0.2)
        model_c = GlobalCLModel(enc_c, hidden_channels=128)
        model_c, _ = train_with_cl(model_c, train_data, val_data, device=device)

        print("  Training AugOnly...")
        enc_a = GCNEncoder(in_ch, 128, dropout=0.2)
        model_a = LinkPredictor(enc_a)
        model_a, _ = augmentation_only_train(model_a, train_data, val_data, device=device)

        # 1. Embedding stability
        print("\n  Measuring embedding stability...")
        stab_v = measure_embedding_stability(model_v, train_data, device=device)
        stab_c = measure_embedding_stability(model_c, train_data, device=device)

        train_deg = get_node_degrees(data).numpy()
        cold_nodes = train_deg <= 5
        warm_nodes = train_deg > 5

        stab_results = {
            "vanilla_cold_std": float(stab_v[cold_nodes].mean()),
            "vanilla_warm_std": float(stab_v[warm_nodes].mean()),
            "cl_cold_std": float(stab_c[cold_nodes].mean()),
            "cl_warm_std": float(stab_c[warm_nodes].mean()),
            "stability_reduction_cold": float((stab_v[cold_nodes].mean() - stab_c[cold_nodes].mean()) / stab_v[cold_nodes].mean()),
            "stability_reduction_warm": float((stab_v[warm_nodes].mean() - stab_c[warm_nodes].mean()) / stab_v[warm_nodes].mean()),
        }
        all_analysis[ds_name]["stability"] = stab_results
        print(f"    Vanilla cold std: {stab_results['vanilla_cold_std']:.6f}")
        print(f"    Vanilla warm std: {stab_results['vanilla_warm_std']:.6f}")
        print(f"    CL cold std: {stab_results['cl_cold_std']:.6f}")
        print(f"    CL warm std: {stab_results['cl_warm_std']:.6f}")
        print(f"    Stability reduction cold: {stab_results['stability_reduction_cold']:.1%}")
        print(f"    Stability reduction warm: {stab_results['stability_reduction_warm']:.1%}")

        # 2. Fine-grained degree-stratified CL gain
        print("\n  Degree-stratified analysis...")
        deg_results = degree_stratified_analysis(model_v, model_c, data, test_data, device)
        all_analysis[ds_name]["degree_stratified"] = deg_results
        for r in deg_results:
            print(f"    deg {r['degree_range']}: n={r['count']}, "
                  f"vanilla={r['vanilla_auc']:.4f}, cl={r['cl_auc']:.4f}, "
                  f"gain={r['gain']:+.4f}")

        # 3. Augmentation-only ablation
        print("\n  Ablation: AugOnly vs CL...")
        aug_res = eval_by_cold_warm(model_a, test_data, data, device)
        vanilla_res = eval_by_cold_warm(model_v, test_data, data, device)
        cl_res = eval_by_cold_warm(model_c, test_data, data, device)

        all_analysis[ds_name]["ablation"] = {
            "vanilla": vanilla_res,
            "augonly": aug_res,
            "globalcl": cl_res,
        }
        print(f"    Vanilla: overall={vanilla_res['overall']:.4f}, cold={vanilla_res['cold']:.4f}")
        print(f"    AugOnly: overall={aug_res['overall']:.4f}, cold={aug_res['cold']:.4f}")
        print(f"    GlobalCL: overall={cl_res['overall']:.4f}, cold={cl_res['cold']:.4f}")

    # Save
    output = "results/full/mechanism_analysis.json"
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w") as f:
        json.dump(all_analysis, f, indent=2, default=str)
    print(f"\nSaved to {output}")


if __name__ == "__main__":
    main()
