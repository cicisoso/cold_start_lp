"""Generate all paper figures from experiment results."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import os

sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.2)
COLORS = sns.color_palette("Set2", 8)
OUT = os.path.join(os.path.dirname(__file__), "paper", "figures")
os.makedirs(OUT, exist_ok=True)


def load_json(path):
    with open(path) as f:
        return json.load(f)


# ============================================================
# Figure 1: Teaser — CL gain vs degree bin (CS + PubMed)
# ============================================================
def fig1_teaser():
    # Mechanism analysis has fine-grained degree data
    data = load_json("results/full/mechanism_analysis.json")

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2), sharey=True)

    for idx, (ds, ax) in enumerate(zip(["CS", "PubMed"], axes)):
        deg_data = data[ds]["degree_stratified"]
        labels = [d["degree_range"] for d in deg_data]
        gains = [d["gain"] * 100 for d in deg_data]
        counts = [d["count"] for d in deg_data]

        bars = ax.bar(range(len(labels)), gains, color=COLORS[0], edgecolor='white', width=0.7)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=9)
        ax.set_title(ds, fontweight='bold', fontsize=12)
        ax.set_xlabel("Min-endpoint degree range")
        if idx == 0:
            ax.set_ylabel("AUC gain from CL (%)")
        ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)

        # Annotate counts
        for bar, c in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    f'n={c}', ha='center', va='bottom', fontsize=7, color='gray')

    fig.suptitle("Graph CL Disproportionately Benefits Low-Degree Edges", fontsize=13, y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(OUT, "fig1_teaser.pdf"), bbox_inches='tight', dpi=300)
    plt.close()
    print("  fig1_teaser.pdf")


# ============================================================
# Figure 2: Heatmap — CL gain across datasets × degree bins
# ============================================================
def fig2_heatmap():
    data = load_json("results/full/mechanism_analysis.json")

    datasets = ["Cora", "CiteSeer", "PubMed", "CS"]
    all_bins = set()
    for ds in datasets:
        for d in data[ds]["degree_stratified"]:
            all_bins.add(d["degree_range"])

    # Ordered bins
    bin_order = ["2-4", "4-6", "6-10", "10-20", "20-50", "50-1000"]
    bin_order = [b for b in bin_order if b in all_bins]

    matrix = []
    for ds in datasets:
        row = []
        deg_dict = {d["degree_range"]: d["gain"]*100 for d in data[ds]["degree_stratified"]}
        for b in bin_order:
            row.append(deg_dict.get(b, np.nan))
        matrix.append(row)

    matrix = np.array(matrix)

    fig, ax = plt.subplots(figsize=(6, 3))
    im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=14)

    ax.set_xticks(range(len(bin_order)))
    ax.set_xticklabels(bin_order, fontsize=10)
    ax.set_yticks(range(len(datasets)))
    ax.set_yticklabels(datasets, fontsize=10)
    ax.set_xlabel("Min-endpoint degree range")

    for i in range(len(datasets)):
        for j in range(len(bin_order)):
            val = matrix[i, j]
            if not np.isnan(val):
                color = 'white' if val > 8 else 'black'
                ax.text(j, i, f'{val:.1f}', ha='center', va='center', fontsize=9, color=color)

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('CL gain (%)')
    ax.set_title("CL Improvement by Degree Bin (%)", fontsize=12)
    plt.tight_layout()
    fig.savefig(os.path.join(OUT, "fig2_heatmap.pdf"), bbox_inches='tight', dpi=300)
    plt.close()
    print("  fig2_heatmap.pdf")


# ============================================================
# Figure 3: Decomposition — AugOnly vs CL across datasets
# ============================================================
def fig3_decomposition():
    data = load_json("results/full/mechanism_analysis.json")

    datasets = ["Cora", "CiteSeer", "PubMed", "CS"]
    aug_gains = []
    cl_gains = []

    for ds in datasets:
        abl = data[ds]["ablation"]
        vanilla_cold = abl["vanilla"]["cold"]
        aug_cold = abl["augonly"]["cold"]
        cl_cold = abl["globalcl"]["cold"]
        aug_gains.append((aug_cold - vanilla_cold) * 100)
        cl_gains.append((cl_cold - vanilla_cold) * 100)

    # Add ogbl-collab from OGB results
    datasets.append("ogbl-collab")
    aug_gains.append(8.8)  # from our results
    cl_gains.append(8.2)

    x = np.arange(len(datasets))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 3.5))
    bars1 = ax.bar(x - width/2, aug_gains, width, label='AugOnly gain', color=COLORS[1], edgecolor='white')
    bars2 = ax.bar(x + width/2, cl_gains, width, label='GlobalCL gain', color=COLORS[0], edgecolor='white')

    ax.set_xticks(x)
    ax.set_xticklabels(datasets, fontsize=10)
    ax.set_ylabel("Cold-edge AUC gain (%)")
    ax.set_title("Augmentation vs Contrastive Objective: Cold-Edge Gains", fontsize=12)
    ax.legend(loc='upper left', fontsize=9)
    ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)

    # Annotate CL-AugOnly gap
    for i in range(len(datasets)):
        gap = cl_gains[i] - aug_gains[i]
        y = max(aug_gains[i], cl_gains[i]) + 0.5
        ax.text(x[i], y, f'Δ={gap:+.1f}', ha='center', fontsize=8, color='darkred')

    plt.tight_layout()
    fig.savefig(os.path.join(OUT, "fig3_decomposition.pdf"), bbox_inches='tight', dpi=300)
    plt.close()
    print("  fig3_decomposition.pdf")


# ============================================================
# Figure 4: Selective CL comparison
# ============================================================
def fig4_selective():
    v2 = load_json("results/full/v2_results.json")

    datasets = ["Cora", "CiteSeer", "PubMed", "CS"]
    methods = ["Vanilla", "Reweight", "FocalLoss", "GlobalCL", "DegreeGatedCL", "RandomCL", "UncertaintyCL"]
    method_colors = {
        "Vanilla": 'gray', "Reweight": COLORS[3], "FocalLoss": COLORS[4],
        "GlobalCL": COLORS[0], "DegreeGatedCL": COLORS[1], "RandomCL": COLORS[2],
        "UncertaintyCL": COLORS[5]
    }

    fig, axes = plt.subplots(1, 4, figsize=(14, 3.5), sharey=True)

    for idx, (ds, ax) in enumerate(zip(datasets, axes)):
        cold_means = []
        cold_stds = []
        labels = []
        colors = []

        for m in methods:
            agg = v2[ds]["GCN"][m]["aggregated"]
            cold = agg.get("cold (2-5)", {})
            cm = cold.get("auc_mean", float("nan"))
            cs = cold.get("auc_std", 0)
            cold_means.append(cm)
            cold_stds.append(cs)
            labels.append(m)
            colors.append(method_colors[m])

        y_pos = range(len(labels))
        ax.barh(y_pos, cold_means, xerr=cold_stds, color=colors, edgecolor='white', height=0.7)
        ax.set_yticks(y_pos)
        if idx == 0:
            ax.set_yticklabels(labels, fontsize=9)
        else:
            ax.set_yticklabels([])
        ax.set_title(ds, fontweight='bold', fontsize=11)
        ax.set_xlabel("Cold AUC")
        ax.set_xlim(0.65, 0.96)

    fig.suptitle("Cold-Edge AUC: CL Variants vs Non-CL Baselines", fontsize=13, y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(OUT, "fig4_selective.pdf"), bbox_inches='tight', dpi=300)
    plt.close()
    print("  fig4_selective.pdf")


# ============================================================
# Figure 5: Sparsity experiment
# ============================================================
def fig5_sparsity():
    data = load_json("results/full/sparsity_results.json")

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2))

    for idx, (ds, ax) in enumerate(zip(["Cora", "CS"], axes)):
        ratios = [1.0, 0.75, 0.50, 0.25]
        vanilla = []
        augonly = []
        cl = []

        for r in ratios:
            key = f"keep_{r}"
            vanilla.append(data[ds][key]["vanilla"]["cold_mean"])
            augonly.append(data[ds][key]["augonly"]["cold_mean"])
            cl.append(data[ds][key]["globalcl"]["cold_mean"])

        x_labels = ["100%", "75%", "50%", "25%"]
        x = range(len(x_labels))
        ax.plot(x, vanilla, 'o-', color='gray', label='Vanilla', linewidth=2)
        ax.plot(x, augonly, 's-', color=COLORS[1], label='AugOnly', linewidth=2)
        ax.plot(x, cl, '^-', color=COLORS[0], label='GlobalCL', linewidth=2)

        ax.set_xticks(x)
        ax.set_xticklabels(x_labels)
        ax.set_xlabel("Edges kept (%)")
        if idx == 0:
            ax.set_ylabel("Cold-edge AUC")
        ax.set_title(ds, fontweight='bold', fontsize=12)
        ax.legend(fontsize=8)

    fig.suptitle("Effect of Graph Sparsity on Self-Supervision Benefits", fontsize=13, y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(OUT, "fig5_sparsity.pdf"), bbox_inches='tight', dpi=300)
    plt.close()
    print("  fig5_sparsity.pdf")


if __name__ == "__main__":
    print("Generating figures...")
    fig1_teaser()
    fig2_heatmap()
    fig3_decomposition()
    fig4_selective()
    fig5_sparsity()
    print("Done!")
