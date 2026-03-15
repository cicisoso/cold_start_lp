"""
ColdCL: Full experiment runner.

Usage:
    python run_full.py --datasets Cora CiteSeer PubMed Photo CS \
                       --encoders GCN SAGE \
                       --seeds 0 1 2 3 4 \
                       --device cuda
"""
import argparse
import json
import os
import sys
import time
import random
import numpy as np
import torch
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))

from src.data import load_dataset, dataset_stats
from src.models import (
    ENCODERS, LinkPredictor, ColdCLModel, GlobalCLModel,
    NodeDupPredictor, MCDropoutPredictor,
)
from src.train import (
    train_standard, train_with_cl, train_nodedup, train_reweight,
)
from src.metrics import evaluate_lp, format_results


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_method(method_name, data, train_data, val_data, test_data,
               encoder_type, in_channels, hidden=128, device="cuda", **kwargs):
    """Run a single method and return test results."""
    EncoderClass = ENCODERS[encoder_type]

    if method_name == "Vanilla":
        encoder = EncoderClass(in_channels, hidden)
        model = LinkPredictor(encoder)
        model, val_auc = train_standard(model, train_data, val_data, device=device)

    elif method_name == "ColdCL":
        encoder = EncoderClass(in_channels, hidden)
        model = ColdCLModel(
            encoder, hidden_channels=hidden,
            deg_threshold=kwargs.get("deg_threshold", 5),
            edge_drop_rate=kwargs.get("edge_drop_rate", 0.3),
            feat_noise_std=kwargs.get("feat_noise_std", 0.1),
        )
        model, val_auc = train_with_cl(
            model, train_data, val_data,
            cl_weight=kwargs.get("cl_weight", 0.5),
            device=device,
        )

    elif method_name == "GlobalCL":
        encoder = EncoderClass(in_channels, hidden)
        model = GlobalCLModel(encoder, hidden_channels=hidden)
        model, val_auc = train_with_cl(
            model, train_data, val_data,
            cl_weight=kwargs.get("cl_weight", 0.5),
            device=device,
        )

    elif method_name == "NodeDup":
        encoder = EncoderClass(in_channels, hidden)
        model = NodeDupPredictor(encoder)
        model, val_auc = train_nodedup(model, train_data, val_data, device=device)

    elif method_name == "Reweight":
        encoder = EncoderClass(in_channels, hidden)
        model = LinkPredictor(encoder)
        model, val_auc = train_reweight(
            model, train_data, val_data,
            cold_weight=kwargs.get("cold_weight", 3.0),
            device=device,
        )

    elif method_name == "MC-ColdCL":
        # ColdCL + MC dropout for uncertainty
        model = MCDropoutPredictor(in_channels, hidden, dropout=0.3, encoder_type=encoder_type)
        # First train with CL-like objective but using MC architecture
        model, val_auc = train_standard(model, train_data, val_data, device=device)
        # Evaluate with MC inference
        results = evaluate_lp(model, test_data, data, device=device, mc_samples=30)
        return results, val_auc

    else:
        raise ValueError(f"Unknown method: {method_name}")

    results = evaluate_lp(model, test_data, data, device=device)
    return results, val_auc


def main():
    parser = argparse.ArgumentParser(description="ColdCL Full Experiments")
    parser.add_argument("--datasets", nargs="+", default=["Cora", "CiteSeer", "PubMed"])
    parser.add_argument("--encoders", nargs="+", default=["GCN", "SAGE"])
    parser.add_argument("--methods", nargs="+",
                        default=["Vanilla", "GlobalCL", "NodeDup", "Reweight", "ColdCL", "MC-ColdCL"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output", type=str, default="results/full/results.json")
    parser.add_argument("--cl_weight", type=float, default=0.5)
    parser.add_argument("--deg_threshold", type=int, default=5)
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    all_results = {}
    start_time = time.time()

    for ds_name in args.datasets:
        print(f"\n{'#'*70}")
        print(f"# Dataset: {ds_name}")
        print(f"{'#'*70}")

        all_results[ds_name] = {}

        for enc_name in args.encoders:
            all_results[ds_name][enc_name] = {}

            for method in args.methods:
                seed_results = []

                for seed in args.seeds:
                    set_seed(seed)
                    print(f"\n  [{ds_name}/{enc_name}/{method}/seed={seed}]", end=" ")

                    data, train_data, val_data, test_data = load_dataset(ds_name, seed=seed)

                    if seed == args.seeds[0] and method == args.methods[0]:
                        dataset_stats(data, ds_name)

                    t0 = time.time()
                    try:
                        results, val_auc = run_method(
                            method, data, train_data, val_data, test_data,
                            enc_name, data.num_features, args.hidden, device,
                            cl_weight=args.cl_weight,
                            deg_threshold=args.deg_threshold,
                        )
                        elapsed = time.time() - t0
                        results["val_auc"] = val_auc
                        results["time_sec"] = elapsed
                        seed_results.append(results)
                        print(f"AUC={results['overall']['auc']:.4f} "
                              f"cold={results.get('cold (2-5)', {}).get('auc', float('nan')):.4f} "
                              f"({elapsed:.1f}s)")
                    except Exception as e:
                        print(f"FAILED: {e}")
                        seed_results.append({"error": str(e)})

                # Aggregate across seeds
                agg = _aggregate_seeds(seed_results)
                all_results[ds_name][enc_name][method] = {
                    "per_seed": seed_results,
                    "aggregated": agg,
                }

                if "overall" in agg:
                    print(f"  >> {method}: AUC={agg['overall']['auc_mean']:.4f}±{agg['overall']['auc_std']:.4f}, "
                          f"cold={agg.get('cold (2-5)', {}).get('auc_mean', float('nan')):.4f}±"
                          f"{agg.get('cold (2-5)', {}).get('auc_std', float('nan')):.4f}")

    elapsed_total = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"Total time: {elapsed_total/60:.1f} min")

    # Save results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"Results saved to {args.output}")

    # Print summary table
    print_summary_table(all_results)


def _aggregate_seeds(seed_results):
    """Aggregate results across seeds."""
    valid = [r for r in seed_results if "error" not in r]
    if not valid:
        return {"error": "all seeds failed"}

    agg = {}
    # Collect all keys from first valid result
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


def print_summary_table(all_results):
    """Print a compact summary table."""
    print(f"\n{'='*90}")
    print("SUMMARY TABLE: Overall AUC / Cold (2-5) AUC")
    print(f"{'='*90}")

    for ds_name, ds_res in all_results.items():
        print(f"\n--- {ds_name} ---")
        header = f"{'Method':<15} {'Encoder':<8} {'Overall AUC':>18} {'Cold AUC':>18} {'Warm AUC':>18}"
        print(header)
        print("-" * len(header))

        for enc_name, enc_res in ds_res.items():
            for method, m_res in enc_res.items():
                agg = m_res.get("aggregated", {})
                overall = agg.get("overall", {})
                cold = agg.get("cold (2-5)", {})
                warm = agg.get("warm (6-20)", {})

                o_str = f"{overall.get('auc_mean', 0):.4f}±{overall.get('auc_std', 0):.4f}"
                c_str = f"{cold.get('auc_mean', 0):.4f}±{cold.get('auc_std', 0):.4f}"
                w_str = f"{warm.get('auc_mean', 0):.4f}±{warm.get('auc_std', 0):.4f}"
                print(f"{method:<15} {enc_name:<8} {o_str:>18} {c_str:>18} {w_str:>18}")


if __name__ == "__main__":
    main()
