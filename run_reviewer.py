"""
Reviewer Response Experiments:
R1: Confounding topological factor analysis (partial correlation)
R2: MRR metric (Hits@20 already computed)
R3: Inductive cold-start evaluation
R4: Degree-preserving sparsification
"""
import os, sys, json, time, random
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.utils import degree, to_undirected
from torch_geometric.transforms import RandomLinkSplit
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy import stats
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))
from src.data import load_dataset, get_node_degrees, DATASETS
from src.models import GCNEncoder, GlobalCLModel, LinkPredictor, ENCODERS
from src.train import train_standard, train_with_cl


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

device = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================
# R1: Confounding Topological Factor Analysis
# ============================================================

def compute_node_features_topo(edge_index, num_nodes):
    """Compute topological features per node: degree, local clustering coeff,
    local homophily proxy (neighbor degree variance), neighborhood diversity."""
    from torch_geometric.utils import degree as pyg_degree

    deg = pyg_degree(edge_index[0], num_nodes=num_nodes).float()
    deg = deg + pyg_degree(edge_index[1], num_nodes=num_nodes).float()

    # Build adjacency list
    adj = defaultdict(set)
    src, dst = edge_index[0].tolist(), edge_index[1].tolist()
    for s, d in zip(src, dst):
        adj[s].add(d)
        adj[d].add(s)

    clustering = np.zeros(num_nodes)
    neighbor_deg_var = np.zeros(num_nodes)
    neighbor_diversity = np.zeros(num_nodes)  # number of unique degree values in neighborhood

    for node in range(num_nodes):
        neighbors = list(adj[node])
        k = len(neighbors)
        if k < 2:
            clustering[node] = 0.0
            neighbor_deg_var[node] = 0.0
            neighbor_diversity[node] = 0.0
            continue

        # Local clustering coefficient
        triangles = 0
        neighbor_set = set(neighbors)
        for i, n1 in enumerate(neighbors):
            for n2 in neighbors[i+1:]:
                if n2 in adj[n1]:
                    triangles += 1
        clustering[node] = 2 * triangles / (k * (k - 1))

        # Neighbor degree variance
        ndeg = [deg[n].item() for n in neighbors]
        neighbor_deg_var[node] = np.var(ndeg) if len(ndeg) > 1 else 0.0

        # Neighborhood diversity (unique degree values)
        neighbor_diversity[node] = len(set(int(d) for d in ndeg)) / max(k, 1)

    return deg.numpy(), clustering, neighbor_deg_var, neighbor_diversity


def compute_edge_ssl_gain(model_vanilla, model_cl, data, test_data, device="cuda"):
    """Compute per-edge SSL gain (CL score - Vanilla score)."""
    test_data_d = test_data.to(device)
    pos_edge = test_data_d.pos_edge_label_index
    neg_edge = test_data_d.neg_edge_label_index
    edge_label_index = torch.cat([pos_edge, neg_edge], dim=1)
    labels = torch.cat([
        torch.ones(pos_edge.size(1)),
        torch.zeros(neg_edge.size(1)),
    ]).cpu().numpy()

    preds = {}
    for name, model in [("vanilla", model_vanilla), ("cl", model_cl)]:
        model.eval()
        model = model.to(device)
        with torch.no_grad():
            out = model(test_data_d.x, test_data_d.edge_index, edge_label_index)
            preds[name] = torch.sigmoid(out).cpu().numpy()

    # Per-edge: CL correct - Vanilla correct (binary improvement)
    vanilla_correct = ((preds["vanilla"] > 0.5) == labels).astype(float)
    cl_correct = ((preds["cl"] > 0.5) == labels).astype(float)
    edge_gain = cl_correct - vanilla_correct

    return edge_label_index.cpu(), labels, preds, edge_gain


def run_r1_confounding():
    """R1: Partial correlation analysis."""
    print(f"\n{'#'*60}\n# R1: Confounding Topological Factor Analysis\n{'#'*60}")

    results = {}
    for ds_name in ["Cora", "CiteSeer", "CS"]:
        set_seed(42)
        data, train_data, val_data, test_data = load_dataset(ds_name, seed=42)
        in_ch = data.num_features

        # Train models
        enc_v = GCNEncoder(in_ch, 128)
        model_v = LinkPredictor(enc_v)
        model_v, _ = train_standard(model_v, train_data, val_data, device=device)

        enc_c = GCNEncoder(in_ch, 128)
        model_c = GlobalCLModel(enc_c, hidden_channels=128)
        model_c, _ = train_with_cl(model_c, train_data, val_data, device=device)

        # Compute topological features
        print(f"\n  Computing topo features for {ds_name}...")
        deg, clust, ndeg_var, ndiv = compute_node_features_topo(
            data.edge_index, data.num_nodes)

        # Get per-edge SSL gain
        edge_label_index, labels, preds, edge_gain = compute_edge_ssl_gain(
            model_v, model_c, data, test_data, device)

        # Per-edge topological features (min of endpoints)
        src, dst = edge_label_index[0].numpy(), edge_label_index[1].numpy()
        edge_deg = np.minimum(deg[src], deg[dst])
        edge_clust = np.minimum(clust[src], clust[dst])
        edge_ndeg_var = np.minimum(ndeg_var[src], ndeg_var[dst])
        edge_ndiv = np.minimum(ndiv[src], ndiv[dst])

        # Bin by degree and compute mean gain
        bins = [(0, 5), (5, 10), (10, 20), (20, 100), (100, 10000)]
        bin_gains = {}
        for lo, hi in bins:
            mask = (edge_deg >= lo) & (edge_deg < hi)
            if mask.sum() > 20:
                bin_gains[f"{lo}-{hi}"] = {"gain": float(edge_gain[mask].mean()),
                                           "count": int(mask.sum())}

        # Correlations
        # Remove NaN/inf
        valid = np.isfinite(edge_deg) & np.isfinite(edge_clust) & np.isfinite(edge_gain)
        ed = edge_deg[valid]
        ec = edge_clust[valid]
        ev = edge_ndeg_var[valid]
        en = edge_ndiv[valid]
        eg = edge_gain[valid]

        # Simple correlations
        r_deg, p_deg = stats.spearmanr(ed, eg)
        r_clust, p_clust = stats.spearmanr(ec, eg)
        r_var, p_var = stats.spearmanr(ev, eg)
        r_div, p_div = stats.spearmanr(en, eg)

        # Partial correlation: degree vs gain, controlling for clustering + diversity
        from numpy.linalg import lstsq
        def partial_corr(x, y, controls):
            """Partial Spearman correlation of x,y controlling for controls."""
            # Rank-transform
            rx = stats.rankdata(x)
            ry = stats.rankdata(y)
            rc = np.column_stack([stats.rankdata(c) for c in controls])
            # Residualize
            rc_aug = np.column_stack([rc, np.ones(len(rx))])
            res_x = rx - rc_aug @ lstsq(rc_aug, rx, rcond=None)[0]
            res_y = ry - rc_aug @ lstsq(rc_aug, ry, rcond=None)[0]
            r, p = stats.pearsonr(res_x, res_y)
            return r, p

        r_partial, p_partial = partial_corr(ed, eg, [ec, en])

        results[ds_name] = {
            "simple_correlations": {
                "degree": {"r": float(r_deg), "p": float(p_deg)},
                "clustering": {"r": float(r_clust), "p": float(p_clust)},
                "neighbor_deg_var": {"r": float(r_var), "p": float(p_var)},
                "neighbor_diversity": {"r": float(r_div), "p": float(p_div)},
            },
            "partial_correlation_deg_controlling_clust_div": {
                "r": float(r_partial), "p": float(p_partial)
            },
            "bin_gains": bin_gains,
        }

        print(f"  {ds_name}:")
        print(f"    Degree vs gain:     r={r_deg:.4f} (p={p_deg:.2e})")
        print(f"    Clustering vs gain: r={r_clust:.4f} (p={p_clust:.2e})")
        print(f"    NdegVar vs gain:    r={r_var:.4f} (p={p_var:.2e})")
        print(f"    NDiversity vs gain: r={r_div:.4f} (p={p_div:.2e})")
        print(f"    Partial(deg|clust,div): r={r_partial:.4f} (p={p_partial:.2e})")

    return results


# ============================================================
# R3: Inductive Cold-Start Evaluation
# ============================================================

def create_inductive_split(data, holdout_frac=0.2, seed=42):
    """Hold out a fraction of nodes entirely. Test: predict links to/from held-out nodes."""
    set_seed(seed)
    num_nodes = data.num_nodes
    n_holdout = int(num_nodes * holdout_frac)

    # Prefer holding out low-degree nodes (more realistic cold-start)
    deg = get_node_degrees(data)
    # Mix: 50% lowest-degree, 50% random
    sorted_idx = torch.argsort(deg)
    low_deg_pool = sorted_idx[:num_nodes // 2]
    perm = torch.randperm(len(low_deg_pool))[:n_holdout // 2]
    holdout_low = low_deg_pool[perm]
    remaining = torch.tensor([i for i in range(num_nodes) if i not in set(holdout_low.tolist())])
    perm2 = torch.randperm(len(remaining))[:n_holdout - len(holdout_low)]
    holdout_rand = remaining[perm2]
    holdout_nodes = torch.cat([holdout_low, holdout_rand])

    holdout_set = set(holdout_nodes.tolist())
    train_mask = torch.tensor([i not in holdout_set for i in range(num_nodes)])

    # Split edges
    src, dst = data.edge_index
    both_train = train_mask[src] & train_mask[dst]
    has_holdout = ~both_train

    train_edge_index = data.edge_index[:, both_train]

    # Test edges: edges involving at least one holdout node
    test_edges = data.edge_index[:, has_holdout]

    # Subsample test edges
    n_test = min(test_edges.size(1), 5000)
    perm = torch.randperm(test_edges.size(1))[:n_test]
    test_pos = test_edges[:, perm]

    # Negative samples
    neg_dst = torch.randint(0, num_nodes, (n_test,))
    test_neg = torch.stack([test_pos[0], neg_dst])

    return {
        "train_edge_index": train_edge_index,
        "train_x": data.x,  # ALL node features available (inductive = features known)
        "test_pos": test_pos,
        "test_neg": test_neg,
        "holdout_nodes": holdout_nodes,
        "num_nodes": num_nodes,
        "num_holdout": n_holdout,
    }


def train_and_eval_inductive(data, split, method="vanilla", device="cuda"):
    """Train on train subgraph, evaluate on inductive edges."""
    in_ch = data.num_features
    x = split["train_x"].to(device)
    train_ei = split["train_edge_index"].to(device)

    # Create training labels (subsample train edges)
    n_train = min(train_ei.size(1) // 2, 50000)
    perm = torch.randperm(train_ei.size(1))[:n_train]
    train_pos = train_ei[:, perm].t()  # [n, 2]
    train_neg_dst = torch.randint(0, split["num_nodes"], (n_train,), device=device)
    train_neg = torch.stack([train_pos[:, 0], train_neg_dst]).t()

    if method == "vanilla":
        encoder = GCNEncoder(in_ch, 128)
        model = LinkPredictor(encoder).to(device)
    else:
        encoder = GCNEncoder(in_ch, 128)
        model = GlobalCLModel(encoder, hidden_channels=128).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    for epoch in range(200):
        model.train()
        optimizer.zero_grad()

        # LP loss
        pos_out = model(x, train_ei, train_pos.t())
        neg_out = model(x, train_ei, train_neg.t())
        out = torch.cat([pos_out, neg_out])
        labels = torch.cat([torch.ones(n_train), torch.zeros(n_train)]).to(device)
        lp_loss = F.binary_cross_entropy_with_logits(out, labels)

        if method == "cl":
            cl_loss = model.compute_cl_loss(x, train_ei)
            loss = lp_loss + 0.5 * cl_loss
        else:
            loss = lp_loss

        loss.backward()
        optimizer.step()

    # Evaluate on inductive edges
    model.eval()
    test_pos = split["test_pos"].to(device)
    test_neg = split["test_neg"].to(device)

    with torch.no_grad():
        pos_out = torch.sigmoid(model(x, train_ei, test_pos)).cpu().numpy()
        neg_out = torch.sigmoid(model(x, train_ei, test_neg)).cpu().numpy()

    labels = np.concatenate([np.ones(len(pos_out)), np.zeros(len(neg_out))])
    preds = np.concatenate([pos_out, neg_out])

    if len(np.unique(labels)) > 1:
        auc = roc_auc_score(labels, preds)
        ap = average_precision_score(labels, preds)
    else:
        auc = ap = 0.5

    return {"auc": auc, "ap": ap}


def run_r3_inductive():
    """R3: Inductive evaluation."""
    print(f"\n{'#'*60}\n# R3: Inductive Cold-Start Evaluation\n{'#'*60}")

    results = {}
    for ds_name in ["Cora", "CiteSeer", "PubMed", "CS"]:
        print(f"\n  {ds_name}:")
        ds_results = {}

        for method in ["vanilla", "cl"]:
            seed_res = []
            for seed in [0, 1, 2]:
                set_seed(seed)
                dataset = DATASETS[ds_name](os.path.join(os.path.dirname(__file__), "data"))
                data = dataset[0]
                data.edge_index = to_undirected(data.edge_index)

                split = create_inductive_split(data, holdout_frac=0.2, seed=seed)
                res = train_and_eval_inductive(data, split, method=method, device=device)
                seed_res.append(res)
                print(f"    [{method}/s={seed}] AUC={res['auc']:.4f} AP={res['ap']:.4f}")

            aucs = [r["auc"] for r in seed_res]
            aps = [r["ap"] for r in seed_res]
            ds_results[method] = {
                "auc_mean": float(np.mean(aucs)), "auc_std": float(np.std(aucs)),
                "ap_mean": float(np.mean(aps)), "ap_std": float(np.std(aps)),
            }
            print(f"    >> {method}: AUC={np.mean(aucs):.4f}±{np.std(aucs):.4f} "
                  f"AP={np.mean(aps):.4f}±{np.std(aps):.4f}")

        gain_auc = ds_results["cl"]["auc_mean"] - ds_results["vanilla"]["auc_mean"]
        gain_ap = ds_results["cl"]["ap_mean"] - ds_results["vanilla"]["ap_mean"]
        print(f"    CL gain: AUC +{gain_auc:.4f}, AP +{gain_ap:.4f}")
        ds_results["gain_auc"] = gain_auc
        ds_results["gain_ap"] = gain_ap
        results[ds_name] = ds_results

    return results


# ============================================================
# R4: Degree-Preserving Sparsification
# ============================================================

def degree_preserving_sparsify(train_data, keep_ratio, seed=42):
    """Remove edges while approximately preserving the degree distribution.
    Strategy: for each node, keep at least ceil(deg * keep_ratio) edges."""
    set_seed(seed)
    edge_index = train_data.edge_index.clone()
    num_nodes = train_data.num_nodes
    n_edges = edge_index.size(1)

    deg = degree(edge_index[0], num_nodes=num_nodes)
    deg = deg + degree(edge_index[1], num_nodes=num_nodes)

    # For each edge, compute how "needed" it is (inverse of endpoint degrees)
    src, dst = edge_index
    importance = 1.0 / (deg[src] + 1) + 1.0 / (deg[dst] + 1)
    # Add noise to break ties
    importance = importance + torch.rand_like(importance) * 0.01

    # Keep top edges by importance (preserves low-degree connections)
    n_keep = int(n_edges * keep_ratio)
    _, top_idx = torch.topk(importance, n_keep)
    keep_mask = torch.zeros(n_edges, dtype=torch.bool)
    keep_mask[top_idx] = True

    result = train_data.clone()
    result.edge_index = edge_index[:, keep_mask]
    return result


def run_r4_sparsification():
    """R4: Compare uniform vs degree-preserving sparsification."""
    print(f"\n{'#'*60}\n# R4: Degree-Preserving Sparsification\n{'#'*60}")

    from run_sparsity import sparsify_graph, augonly_train, eval_cold_warm
    results = {}

    for ds_name in ["CS", "Cora"]:
        print(f"\n  {ds_name}:")
        results[ds_name] = {}

        for ratio in [1.0, 0.50, 0.25]:
            results[ds_name][f"keep_{ratio}"] = {}

            for sparsify_method in ["uniform", "degree_preserving"]:
                seed_res = []
                for seed in [0, 1, 2]:
                    set_seed(seed)
                    data, train_data, val_data, test_data = load_dataset(ds_name, seed=seed)
                    in_ch = data.num_features

                    if ratio < 1.0:
                        if sparsify_method == "uniform":
                            sparse_train = sparsify_graph(train_data, ratio)
                        else:
                            sparse_train = degree_preserving_sparsify(train_data, ratio, seed)
                    else:
                        sparse_train = train_data

                    # Train GlobalCL
                    enc = GCNEncoder(in_ch, 128)
                    model = GlobalCLModel(enc, hidden_channels=128)
                    model, _ = train_with_cl(model, sparse_train, val_data, device=device)
                    res = eval_cold_warm(model, test_data, data, device)
                    seed_res.append(res)

                colds = [r["cold"] for r in seed_res if not np.isnan(r["cold"])]
                overalls = [r["overall"] for r in seed_res]
                results[ds_name][f"keep_{ratio}"][sparsify_method] = {
                    "cold_mean": float(np.mean(colds)) if colds else float("nan"),
                    "cold_std": float(np.std(colds)) if colds else float("nan"),
                    "overall_mean": float(np.mean(overalls)),
                }
                print(f"    keep={ratio:.0%}, {sparsify_method}: "
                      f"cold={np.mean(colds):.4f}±{np.std(colds):.4f}, "
                      f"overall={np.mean(overalls):.4f}")

    return results


# ============================================================
# Main
# ============================================================

def main():
    all_results = {}

    all_results["R1_confounding"] = run_r1_confounding()
    all_results["R3_inductive"] = run_r3_inductive()
    all_results["R4_sparsification"] = run_r4_sparsification()

    output = "results/full/reviewer_response.json"
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nAll reviewer experiments saved to {output}")


if __name__ == "__main__":
    main()
