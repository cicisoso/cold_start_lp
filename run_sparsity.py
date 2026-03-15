"""
Controlled Sparsity Experiment.

Hypothesis: The benefit of CL over AugOnly grows as the graph becomes sparser.

Method: Take a dataset, subsample edges at different rates (100%, 75%, 50%, 25%),
and measure the CL vs AugOnly gap at each sparsity level.
"""
import os
import sys
import json
import time
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.utils import degree, to_undirected
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(__file__))
from src.data import load_dataset, get_node_degrees
from src.models import GCNEncoder, SAGEEncoder, GlobalCLModel, LinkPredictor
from src.train import train_standard, train_with_cl
from src.metrics import evaluate_lp


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sparsify_graph(train_data, keep_ratio):
    """Remove edges uniformly to simulate sparser graphs."""
    if keep_ratio >= 1.0:
        return train_data

    train_data = train_data.clone()
    edge_index = train_data.edge_index

    # For undirected: remove edges in pairs
    n_edges = edge_index.size(1)
    # Group by pairs (i, j) and (j, i)
    n_keep = int(n_edges * keep_ratio)
    # Simple approach: random mask
    mask = torch.zeros(n_edges, dtype=torch.bool)
    perm = torch.randperm(n_edges)[:n_keep]
    mask[perm] = True

    train_data.edge_index = edge_index[:, mask]
    return train_data


def augonly_train(model, train_data, val_data, edge_drop_rate=0.3,
                  feat_noise_std=0.1, epochs=300, lr=0.01,
                  weight_decay=5e-4, patience=50, device="cuda"):
    """Train with augmented graph but NO contrastive loss."""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    train_data = train_data.to(device)
    val_data = val_data.to(device)

    pos_edge = train_data.pos_edge_label_index
    neg_edge = train_data.neg_edge_label_index
    edge_label_index = torch.cat([pos_edge, neg_edge], dim=1)
    labels = torch.cat([
        torch.ones(pos_edge.size(1)),
        torch.zeros(neg_edge.size(1)),
    ]).to(device)

    best_val_auc = 0
    best_state = None
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        # Augmented graph
        keep = torch.bernoulli(torch.full((train_data.edge_index.size(1),),
                               1 - edge_drop_rate, device=device)).bool()
        aug_ei = train_data.edge_index[:, keep]
        aug_x = train_data.x + torch.randn_like(train_data.x) * feat_noise_std

        out = model(aug_x, aug_ei, edge_label_index)
        loss = F.binary_cross_entropy_with_logits(out, labels)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            val_auc = _quick_eval(model, val_data, device)
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


def _quick_eval(model, data, device):
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


def eval_cold_warm(model, test_data, data, device="cuda"):
    """Quick cold/warm evaluation."""
    results = evaluate_lp(model, test_data, data, device)
    cold = results.get("cold (2-5)", {}).get("auc", float("nan"))
    warm = results.get("warm (6-20)", {}).get("auc", float("nan"))
    overall = results.get("overall", {}).get("auc", float("nan"))
    return {"overall": overall, "cold": cold, "warm": warm}


def run_sparsity_experiment(ds_name="CS", seeds=[0, 1, 2], device="cuda"):
    """Run full sparsity sweep."""
    print(f"\n{'#'*60}")
    print(f"# Sparsity Experiment: {ds_name}")
    print(f"{'#'*60}")

    keep_ratios = [1.0, 0.75, 0.50, 0.25]
    methods = ["vanilla", "augonly", "globalcl"]
    all_results = {}

    for ratio in keep_ratios:
        print(f"\n  --- Keep ratio: {ratio:.0%} ---")
        all_results[f"keep_{ratio}"] = {}

        for method in methods:
            seed_results = []

            for seed in seeds:
                set_seed(seed)
                data, train_data, val_data, test_data = load_dataset(ds_name, seed=seed)
                in_ch = data.num_features

                # Sparsify training graph
                sparse_train = sparsify_graph(train_data, ratio)

                if seed == seeds[0] and method == methods[0]:
                    orig_edges = train_data.edge_index.size(1)
                    sparse_edges = sparse_train.edge_index.size(1)
                    print(f"    Edges: {orig_edges} → {sparse_edges} ({sparse_edges/orig_edges:.0%})")

                    # Cold node fraction at this sparsity
                    deg = degree(sparse_train.edge_index[0], num_nodes=data.num_nodes)
                    deg = deg + degree(sparse_train.edge_index[1], num_nodes=data.num_nodes)
                    cold_frac = (deg <= 5).float().mean()
                    print(f"    Cold nodes (deg≤5): {cold_frac:.1%}")

                t0 = time.time()

                if method == "vanilla":
                    encoder = GCNEncoder(in_ch, 128)
                    model = LinkPredictor(encoder)
                    model, val_auc = train_standard(model, sparse_train, val_data, device=device)
                elif method == "augonly":
                    encoder = GCNEncoder(in_ch, 128)
                    model = LinkPredictor(encoder)
                    model, val_auc = augonly_train(model, sparse_train, val_data, device=device)
                elif method == "globalcl":
                    encoder = GCNEncoder(in_ch, 128)
                    model = GlobalCLModel(encoder, hidden_channels=128)
                    model, val_auc = train_with_cl(model, sparse_train, val_data, device=device)

                res = eval_cold_warm(model, test_data, data, device)
                elapsed = time.time() - t0
                res["time"] = elapsed
                seed_results.append(res)

                print(f"    [{method}/s={seed}] overall={res['overall']:.4f} "
                      f"cold={res['cold']:.4f} ({elapsed:.1f}s)")

            # Aggregate
            overalls = [r["overall"] for r in seed_results]
            colds = [r["cold"] for r in seed_results if not np.isnan(r["cold"])]
            warms = [r["warm"] for r in seed_results if not np.isnan(r["warm"])]

            all_results[f"keep_{ratio}"][method] = {
                "overall_mean": np.mean(overalls),
                "overall_std": np.std(overalls),
                "cold_mean": np.mean(colds) if colds else float("nan"),
                "cold_std": np.std(colds) if colds else float("nan"),
                "warm_mean": np.mean(warms) if warms else float("nan"),
                "warm_std": np.std(warms) if warms else float("nan"),
            }

    return all_results


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    final = {}

    for ds in ["CS", "PubMed", "Cora"]:
        final[ds] = run_sparsity_experiment(ds, seeds=[0, 1, 2], device=device)

    # Print summary
    print(f"\n{'='*90}")
    print("SPARSITY EXPERIMENT SUMMARY")
    print(f"{'='*90}")

    for ds, ds_res in final.items():
        print(f"\n--- {ds} ---")
        print(f"{'Keep%':>8} {'Vanilla Cold':>16} {'AugOnly Cold':>16} {'CL Cold':>16} "
              f"{'CL-AugOnly':>12} {'CL-Vanilla':>12}")
        print("-" * 84)

        for ratio in [1.0, 0.75, 0.50, 0.25]:
            key = f"keep_{ratio}"
            if key not in ds_res:
                continue
            v = ds_res[key].get("vanilla", {})
            a = ds_res[key].get("augonly", {})
            c = ds_res[key].get("globalcl", {})

            vc = v.get("cold_mean", float("nan"))
            ac = a.get("cold_mean", float("nan"))
            cc = c.get("cold_mean", float("nan"))

            cl_aug_gap = cc - ac if not (np.isnan(cc) or np.isnan(ac)) else float("nan")
            cl_van_gap = cc - vc if not (np.isnan(cc) or np.isnan(vc)) else float("nan")

            print(f"{ratio:>7.0%} "
                  f"{vc:>7.4f}±{v.get('cold_std',0):>.4f} "
                  f"{ac:>7.4f}±{a.get('cold_std',0):>.4f} "
                  f"{cc:>7.4f}±{c.get('cold_std',0):>.4f} "
                  f"{cl_aug_gap:>+11.4f} "
                  f"{cl_van_gap:>+11.4f}")

    # Save
    output = "results/full/sparsity_results.json"
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w") as f:
        json.dump(final, f, indent=2, default=str)
    print(f"\nSaved to {output}")


if __name__ == "__main__":
    main()
