"""
ColdCL v2: Uncertainty-Guided Selective Contrastive Regularization
for Tail-Node Link Prediction.

Key improvements over v1:
1. Training-graph degree only (no leakage)
2. Uncertainty-guided adaptive node selection (replaces fixed degree threshold)
3. Additional baselines: focal loss, uncertainty-weighted loss, random selective CL
4. Ablation: degree-gated vs uncertainty-gated vs random-gated
5. Mechanism analysis: embedding stability metrics
"""
import os
import sys
import json
import time
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))
from src.data import load_dataset, get_node_degrees
from src.models import ENCODERS, LinkPredictor, GCNEncoder, SAGEEncoder
from src.metrics import evaluate_lp, expected_calibration_error
from sklearn.metrics import roc_auc_score


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# Improved ColdCL with uncertainty-guided gating
# ============================================================

class UncertaintyCLModel(nn.Module):
    """
    Uncertainty-Guided Selective CL for Link Prediction.

    Instead of fixed degree threshold, selects nodes for CL based on
    the model's own uncertainty (predictive variance via MC dropout).
    """

    def __init__(self, in_channels, hidden_channels=128, encoder_type="GCN",
                 proj_dim=64, top_p=0.3, edge_drop_rate=0.3,
                 feat_noise_std=0.1, temperature=0.5, mc_dropout=0.2):
        super().__init__()
        EncoderClass = ENCODERS[encoder_type]
        self.encoder = EncoderClass(in_channels, hidden_channels, dropout=mc_dropout)
        self.projector = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, proj_dim),
        )
        self.mc_drop = nn.Dropout(p=mc_dropout)
        self.top_p = top_p
        self.edge_drop_rate = edge_drop_rate
        self.feat_noise_std = feat_noise_std
        self.temperature = temperature
        self._uncertain_mask = None  # Updated periodically

    def encode(self, x, edge_index):
        return self.encoder(x, edge_index)

    def decode(self, z, edge_label_index):
        src, dst = edge_label_index
        return (z[src] * z[dst]).sum(dim=-1)

    def forward(self, x, edge_index, edge_label_index):
        z = self.encode(x, edge_index)
        return self.decode(z, edge_label_index)

    @torch.no_grad()
    def estimate_node_uncertainty(self, x, edge_index, n_samples=5):
        """Estimate per-node embedding variance via MC dropout."""
        self.train()
        embeddings = []
        for _ in range(n_samples):
            z = self.encode(x, edge_index)
            z = self.mc_drop(z)
            embeddings.append(z)
        embeddings = torch.stack(embeddings)  # [n_samples, num_nodes, hidden]
        var = embeddings.var(dim=0).mean(dim=1)  # [num_nodes]
        return var

    def update_uncertain_mask(self, x, edge_index, n_samples=5):
        """Update which nodes are selected for CL (top-p most uncertain)."""
        var = self.estimate_node_uncertainty(x, edge_index, n_samples)
        k = max(1, int(x.size(0) * self.top_p))
        _, top_idx = torch.topk(var, k)
        mask = torch.zeros(x.size(0), dtype=torch.bool, device=x.device)
        mask[top_idx] = True
        self._uncertain_mask = mask
        return mask

    def selective_augment(self, x, edge_index, target_mask):
        """Augment ONLY target nodes' ego-nets."""
        src, dst = edge_index
        incident = target_mask[src] | target_mask[dst]

        # Edge dropout on incident edges only
        keep_prob = torch.ones(edge_index.size(1), device=edge_index.device)
        keep_prob[incident] = 1.0 - self.edge_drop_rate
        keep_mask = torch.bernoulli(keep_prob).bool()
        aug_ei = edge_index[:, keep_mask]

        # Feature noise on target nodes only
        aug_x = x.clone()
        target_idx = torch.where(target_mask)[0]
        if len(target_idx) > 0:
            noise = torch.randn(len(target_idx), x.size(1), device=x.device) * self.feat_noise_std
            aug_x[target_idx] = aug_x[target_idx] + noise

        return aug_x, aug_ei

    def contrastive_loss(self, z_orig, z_aug, target_mask, max_nodes=512):
        """InfoNCE on selected nodes."""
        h1 = self.projector(z_orig)
        h2 = self.projector(z_aug)
        idx = torch.where(target_mask)[0]
        if len(idx) == 0:
            return torch.tensor(0.0, device=z_orig.device)
        h1 = h1[idx]
        h2 = h2[idx]
        if h1.size(0) > max_nodes:
            perm = torch.randperm(h1.size(0), device=h1.device)[:max_nodes]
            h1 = h1[perm]
            h2 = h2[perm]
        h1 = F.normalize(h1, dim=1)
        h2 = F.normalize(h2, dim=1)
        sim = torch.mm(h1, h2.t()) / self.temperature
        labels = torch.arange(h1.size(0), device=sim.device)
        return F.cross_entropy(sim, labels)

    def compute_cl_loss(self, x, edge_index):
        """Full CL: augment selected nodes, compute InfoNCE."""
        target_mask = self._uncertain_mask
        if target_mask is None:
            target_mask = self.update_uncertain_mask(x, edge_index)

        z_orig = self.encode(x, edge_index)
        aug_x, aug_ei = self.selective_augment(x, edge_index, target_mask)
        z_aug = self.encode(aug_x, aug_ei)
        return self.contrastive_loss(z_orig, z_aug, target_mask)


# ============================================================
# Additional baselines
# ============================================================

class FocalLossPredictor(nn.Module):
    """LP with focal loss (upweights hard examples)."""
    def __init__(self, encoder, gamma=2.0):
        super().__init__()
        self.encoder = encoder
        self.gamma = gamma

    def encode(self, x, edge_index):
        return self.encoder(x, edge_index)

    def decode(self, z, edge_label_index):
        src, dst = edge_label_index
        return (z[src] * z[dst]).sum(dim=-1)

    def forward(self, x, edge_index, edge_label_index):
        z = self.encode(x, edge_index)
        return self.decode(z, edge_label_index)

    def focal_loss(self, logits, labels):
        bce = F.binary_cross_entropy_with_logits(logits, labels, reduction='none')
        p = torch.sigmoid(logits)
        pt = p * labels + (1 - p) * (1 - labels)
        focal_weight = (1 - pt) ** self.gamma
        return (focal_weight * bce).mean()


def degree_gated_mask(edge_index, num_nodes, threshold=5):
    """Fixed degree threshold mask (v1 ColdCL)."""
    from torch_geometric.utils import degree
    deg = degree(edge_index[0], num_nodes=num_nodes)
    deg = deg + degree(edge_index[1], num_nodes=num_nodes)
    return deg <= threshold


def random_mask(num_nodes, frac=0.3, device="cpu"):
    """Random node selection (ablation baseline)."""
    k = max(1, int(num_nodes * frac))
    idx = torch.randperm(num_nodes, device=device)[:k]
    mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    mask[idx] = True
    return mask


# ============================================================
# Training functions
# ============================================================

def get_lp_labels(train_data, device):
    pos_edge = train_data.pos_edge_label_index
    neg_edge = train_data.neg_edge_label_index
    edge_label_index = torch.cat([pos_edge, neg_edge], dim=1)
    labels = torch.cat([
        torch.ones(pos_edge.size(1)),
        torch.zeros(neg_edge.size(1)),
    ]).to(device)
    return edge_label_index, labels


def train_uncertainty_cl(model, train_data, val_data, cl_weight=0.5,
                         update_interval=20, epochs=300, lr=0.01,
                         weight_decay=5e-4, patience=50, device="cuda"):
    """Train UncertaintyCL model."""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    train_data = train_data.to(device)
    val_data = val_data.to(device)

    edge_label_index, labels = get_lp_labels(train_data, device)
    best_val_auc = 0
    best_state = None
    no_improve = 0

    for epoch in range(epochs):
        # Update uncertainty mask periodically
        if epoch % update_interval == 0:
            model.update_uncertain_mask(train_data.x, train_data.edge_index)

        model.train()
        optimizer.zero_grad()

        out = model(train_data.x, train_data.edge_index, edge_label_index)
        lp_loss = F.binary_cross_entropy_with_logits(out, labels)
        cl_loss = model.compute_cl_loss(train_data.x, train_data.edge_index)
        loss = lp_loss + cl_weight * cl_loss
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


def train_focal(model, train_data, val_data, epochs=300, lr=0.01,
                weight_decay=5e-4, patience=50, device="cuda"):
    """Train with focal loss."""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    train_data = train_data.to(device)
    val_data = val_data.to(device)

    edge_label_index, labels = get_lp_labels(train_data, device)
    best_val_auc = 0
    best_state = None
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(train_data.x, train_data.edge_index, edge_label_index)
        loss = model.focal_loss(out, labels)
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


def train_degree_cl(model, train_data, val_data, deg_threshold=5,
                    cl_weight=0.5, epochs=300, lr=0.01, weight_decay=5e-4,
                    patience=50, device="cuda"):
    """Train with fixed degree-gated CL (v1 ColdCL for ablation)."""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    train_data = train_data.to(device)
    val_data = val_data.to(device)

    deg_mask = degree_gated_mask(train_data.edge_index, train_data.num_nodes, deg_threshold).to(device)
    model._uncertain_mask = deg_mask  # Override with degree mask

    edge_label_index, labels = get_lp_labels(train_data, device)
    best_val_auc = 0
    best_state = None
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(train_data.x, train_data.edge_index, edge_label_index)
        lp_loss = F.binary_cross_entropy_with_logits(out, labels)
        cl_loss = model.compute_cl_loss(train_data.x, train_data.edge_index)
        loss = lp_loss + cl_weight * cl_loss
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


def train_random_cl(model, train_data, val_data, frac=0.3,
                    cl_weight=0.5, epochs=300, lr=0.01, weight_decay=5e-4,
                    patience=50, device="cuda"):
    """Train with random selective CL (ablation baseline)."""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    train_data = train_data.to(device)
    val_data = val_data.to(device)

    edge_label_index, labels = get_lp_labels(train_data, device)
    best_val_auc = 0
    best_state = None
    no_improve = 0

    for epoch in range(epochs):
        # Random mask changes every epoch
        model._uncertain_mask = random_mask(train_data.num_nodes, frac, device)

        model.train()
        optimizer.zero_grad()
        out = model(train_data.x, train_data.edge_index, edge_label_index)
        lp_loss = F.binary_cross_entropy_with_logits(out, labels)
        cl_loss = model.compute_cl_loss(train_data.x, train_data.edge_index)
        loss = lp_loss + cl_weight * cl_loss
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
    from sklearn.metrics import roc_auc_score
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


# ============================================================
# Main experiment
# ============================================================

def run_method(method, data, train_data, val_data, test_data,
               encoder_type, in_channels, hidden=128, device="cuda"):
    EncoderClass = ENCODERS[encoder_type]

    if method == "Vanilla":
        encoder = EncoderClass(in_channels, hidden)
        model = LinkPredictor(encoder)
        from src.train import train_standard
        model, val_auc = train_standard(model, train_data, val_data, device=device)

    elif method == "GlobalCL":
        from src.models import GlobalCLModel
        encoder = EncoderClass(in_channels, hidden)
        model = GlobalCLModel(encoder, hidden_channels=hidden)
        from src.train import train_with_cl
        model, val_auc = train_with_cl(model, train_data, val_data, device=device)

    elif method == "FocalLoss":
        encoder = EncoderClass(in_channels, hidden)
        model = FocalLossPredictor(encoder, gamma=2.0)
        model, val_auc = train_focal(model, train_data, val_data, device=device)

    elif method == "DegreeGatedCL":
        model = UncertaintyCLModel(in_channels, hidden, encoder_type)
        model, val_auc = train_degree_cl(model, train_data, val_data, device=device)

    elif method == "RandomCL":
        model = UncertaintyCLModel(in_channels, hidden, encoder_type)
        model, val_auc = train_random_cl(model, train_data, val_data, device=device)

    elif method == "UncertaintyCL":
        model = UncertaintyCLModel(in_channels, hidden, encoder_type, top_p=0.3)
        model, val_auc = train_uncertainty_cl(model, train_data, val_data, device=device)

    elif method == "Reweight":
        encoder = EncoderClass(in_channels, hidden)
        model = LinkPredictor(encoder)
        from src.train import train_reweight
        model, val_auc = train_reweight(model, train_data, val_data, device=device)

    else:
        raise ValueError(f"Unknown method: {method}")

    results = evaluate_lp(model, test_data, data, device=device)
    return results, val_auc


def aggregate_seeds(seed_results):
    valid = [r for r in seed_results if "error" not in r]
    if not valid:
        return {"error": "all failed"}
    agg = {}
    for key in valid[0]:
        if isinstance(valid[0][key], dict):
            agg[key] = {}
            for metric in valid[0][key]:
                vals = [r[key][metric] for r in valid if key in r and metric in r[key]
                        and isinstance(r[key][metric], (int, float)) and not np.isnan(r[key][metric])]
                if vals:
                    agg[key][f"{metric}_mean"] = np.mean(vals)
                    agg[key][f"{metric}_std"] = np.std(vals)
    return agg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["Cora", "CiteSeer", "PubMed", "CS"])
    parser.add_argument("--encoders", nargs="+", default=["GCN"])
    parser.add_argument("--methods", nargs="+",
                        default=["Vanilla", "GlobalCL", "Reweight", "FocalLoss",
                                 "DegreeGatedCL", "RandomCL", "UncertaintyCL"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output", type=str, default="results/full/v2_results.json")
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    all_results = {}
    start = time.time()

    for ds in args.datasets:
        print(f"\n{'#'*70}\n# {ds}\n{'#'*70}")
        all_results[ds] = {}

        for enc in args.encoders:
            all_results[ds][enc] = {}

            for method in args.methods:
                seed_results = []
                for seed in args.seeds:
                    set_seed(seed)
                    data, train_data, val_data, test_data = load_dataset(ds, seed=seed)
                    print(f"  [{ds}/{enc}/{method}/s={seed}]", end=" ")
                    t0 = time.time()
                    try:
                        res, val_auc = run_method(
                            method, data, train_data, val_data, test_data,
                            enc, data.num_features, args.hidden, device
                        )
                        res["time_sec"] = time.time() - t0
                        seed_results.append(res)
                        cold = res.get("cold (2-5)", {}).get("auc", float("nan"))
                        print(f"AUC={res['overall']['auc']:.4f} cold={cold:.4f} ({res['time_sec']:.1f}s)")
                    except Exception as e:
                        print(f"FAILED: {e}")
                        seed_results.append({"error": str(e)})

                agg = aggregate_seeds(seed_results)
                all_results[ds][enc][method] = {"per_seed": seed_results, "aggregated": agg}

                if "overall" in agg:
                    o = agg["overall"]
                    c = agg.get("cold (2-5)", {})
                    print(f"  >> {method}: AUC={o.get('auc_mean',0):.4f}±{o.get('auc_std',0):.4f} "
                          f"cold={c.get('auc_mean',0):.4f}±{c.get('auc_std',0):.4f}")

    print(f"\nTotal: {(time.time()-start)/60:.1f} min")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"Saved to {args.output}")

    # Print summary
    print(f"\n{'='*100}")
    print("V2 SUMMARY: Overall AUC / Cold (2-5) AUC / Warm (6-20) AUC")
    print(f"{'='*100}")
    for ds, ds_res in all_results.items():
        print(f"\n--- {ds} ---")
        for enc, enc_res in ds_res.items():
            header = f"{'Method':<16} {'Overall':>16} {'Cold(2-5)':>16} {'Warm(6-20)':>16}"
            print(header)
            print("-" * len(header))
            for method, m_res in enc_res.items():
                a = m_res.get("aggregated", {})
                o = a.get("overall", {})
                c = a.get("cold (2-5)", {})
                w = a.get("warm (6-20)", {})
                print(f"{method:<16} "
                      f"{o.get('auc_mean',0):.4f}±{o.get('auc_std',0):.4f}  "
                      f"{c.get('auc_mean',0):.4f}±{c.get('auc_std',0):.4f}  "
                      f"{w.get('auc_mean',0):.4f}±{w.get('auc_std',0):.4f}")


if __name__ == "__main__":
    main()
