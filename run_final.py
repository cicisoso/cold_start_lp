"""
Final experiment run: all metrics (AUC, AP, Hits@20, Hits@50, MRR, ECE) +
GAT backbone + full method suite + 5 seeds.

This produces the definitive result tables for the 8-page paper.
"""
import os, sys, json, time, random, argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from src.data import load_dataset, dataset_stats
from src.models import ENCODERS, LinkPredictor, GlobalCLModel, NodeDupPredictor
from src.train import train_standard, train_with_cl, train_nodedup, train_reweight
from src.metrics import evaluate_lp
from run_v2 import (
    UncertaintyCLModel, FocalLossPredictor,
    train_uncertainty_cl, train_focal, train_degree_cl, train_random_cl,
    set_seed, degree_gated_mask
)


def run_method(method, data, train_data, val_data, test_data,
               encoder_type, in_channels, hidden=128, device="cuda"):
    EncoderClass = ENCODERS[encoder_type]

    if method == "Vanilla":
        encoder = EncoderClass(in_channels, hidden)
        model = LinkPredictor(encoder)
        model, val_auc = train_standard(model, train_data, val_data, device=device)
    elif method == "GlobalCL":
        encoder = EncoderClass(in_channels, hidden)
        model = GlobalCLModel(encoder, hidden_channels=hidden)
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
        model, val_auc = train_reweight(model, train_data, val_data, device=device)
    elif method == "NodeDup":
        encoder = EncoderClass(in_channels, hidden)
        model = NodeDupPredictor(encoder)
        model, val_auc = train_nodedup(model, train_data, val_data, device=device)
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
                    agg[key][f"{metric}_mean"] = float(np.mean(vals))
                    agg[key][f"{metric}_std"] = float(np.std(vals))
    return agg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["Cora", "CiteSeer", "PubMed", "CS"])
    parser.add_argument("--encoders", nargs="+", default=["GCN", "SAGE", "GAT"])
    parser.add_argument("--methods", nargs="+",
                        default=["Vanilla", "Reweight", "FocalLoss", "NodeDup",
                                 "GlobalCL", "DegreeGatedCL", "RandomCL", "UncertaintyCL"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default="results/full/final_results.json")
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    all_results = {}
    total_start = time.time()

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
                    t0 = time.time()
                    try:
                        res, val_auc = run_method(
                            method, data, train_data, val_data, test_data,
                            enc, data.num_features, args.hidden, device)
                        res["time_sec"] = time.time() - t0
                        seed_results.append(res)
                        o = res["overall"]
                        c = res.get("cold (2-5)", {})
                        print(f"  [{ds}/{enc}/{method}/s={seed}] "
                              f"AUC={o['auc']:.4f} AP={o['ap']:.4f} H@20={o.get('hits20',0):.4f} "
                              f"cold_AUC={c.get('auc',float('nan')):.4f} ({res['time_sec']:.1f}s)")
                    except Exception as e:
                        print(f"  [{ds}/{enc}/{method}/s={seed}] FAILED: {e}")
                        seed_results.append({"error": str(e)})

                agg = aggregate_seeds(seed_results)
                all_results[ds][enc][method] = {"per_seed": seed_results, "aggregated": agg}
                if "overall" in agg:
                    o = agg["overall"]
                    c = agg.get("cold (2-5)", {})
                    print(f"  >> {method}/{enc}: "
                          f"AUC={o.get('auc_mean',0):.4f}±{o.get('auc_std',0):.4f} "
                          f"AP={o.get('ap_mean',0):.4f}±{o.get('ap_std',0):.4f} "
                          f"H@20={o.get('hits20_mean',0):.4f} "
                          f"cold={c.get('auc_mean',0):.4f}±{c.get('auc_std',0):.4f}")

    total_time = time.time() - total_start
    print(f"\nTotal time: {total_time/60:.1f} min")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"Saved to {args.output}")

    # Print compact summary
    print(f"\n{'='*120}")
    print("FINAL SUMMARY: AUC / AP / Hits@20 — Overall and Cold(2-5)")
    print(f"{'='*120}")
    for ds in args.datasets:
        print(f"\n--- {ds} ---")
        for enc in args.encoders:
            print(f"  Backbone: {enc}")
            hdr = f"    {'Method':<16} {'AUC':>8} {'AP':>8} {'H@20':>8} | {'Cold AUC':>10} {'Cold AP':>10} {'Cold H@20':>10}"
            print(hdr)
            print("    " + "-" * (len(hdr) - 4))
            for method in args.methods:
                a = all_results[ds][enc][method].get("aggregated", {})
                o = a.get("overall", {})
                c = a.get("cold (2-5)", {})
                print(f"    {method:<16} "
                      f"{o.get('auc_mean',0):>7.4f}  {o.get('ap_mean',0):>7.4f}  {o.get('hits20_mean',0):>7.4f} | "
                      f"{c.get('auc_mean',0):>9.4f}  {c.get('ap_mean',0):>9.4f}  {c.get('hits20_mean',0):>9.4f}")


if __name__ == "__main__":
    main()
