# Idea Discovery Report

**Direction**: 图神经网络在链路预测中的冷启动与稀疏性 (GNN Cold Start & Sparsity in Link Prediction)
**Generated**: 2026-03-15
**Ideas evaluated**: 10 generated → 6 survived filtering → 3 piloted → 2 recommended

## Executive Summary

Cold-Only Contrastive Learning (Idea 5) shows the strongest signal: degree-gated augmentation improves cold-edge AUC by +6% on Cora and +13% on CiteSeer without degrading warm regions. Calibrated Cold-Start LP (Idea 8) confirms that cold regions are systematically more uncertain and poorly calibrated, and selective prediction based on uncertainty significantly boosts AUC. Both ideas can be combined into a single flagship paper on uncertainty-aware, cold-targeted LP.

## Literature Landscape

### Sub-direction 1: Cold-Start Node Handling
- **NodeDup** (ICLR'25 rejected): Simple node duplication augmentation. +38% isolated, +13% low-degree.
- **FS-GNN** (AAAI'25): Feature + Structure completion for cold-start recommendation.
- **SpotTarget** (WSDM'24): Target-link inclusion disproportionately affects low-degree nodes. Up to 15x gain.
- **Graph Neural Patching** (Springer'25): Avoids degrading warm users during cold-start adaptation.

### Sub-direction 2: Long-Tailed / Structural Sparsity
- **LTLP** (KDD'24): Common-neighbor count > node degree for LP accuracy. Tail AUC 0.76→0.86.
- **JT-IMPN** (ML Journal'25): Multi-hop + residual for low-degree graphs.
- **NCNC** (ICLR'24): Common neighbor completion via learned model. Addresses graph incompleteness.

### Sub-direction 3: Contrastive Learning for Sparsity
- **DAHGN** (KBS'24): Degree-aware CL for node classification (not LP).
- **ECLiP** (IPM'24): Edge-level contrastive learning for LP (not degree-aware).
- **CoGCL** (arXiv'24): Discrete-code contrastive views for sparse recommendation.

### Sub-direction 4: Foundation Models / Zero-Shot LP
- **ULTRA** (ICLR'24): Zero-shot LP across 50+ KGs.
- **KG-ICL** (NeurIPS'24): In-context learning for KG reasoning.
- **MOTIF/KGFM** (ICML'25): Richer motifs for expressive KG foundation models.

### Key Gap
No existing work combines **degree-gated contrastive augmentation** with **uncertainty-aware prediction** specifically for cold-start link prediction on general graphs (beyond RecSys).

## Ranked Ideas

### 🏆 Idea 5: Cold-Only Contrastive Learning — RECOMMENDED
- **Hypothesis**: Applying contrastive augmentation only inside low-degree ego-nets (degree-gated edge-drop + feature perturbation + consistency loss) improves cold LP without degrading warm regions.
- **Minimum experiment**: Compare no-CL, global-CL, and cold-only-CL on Cora/CiteSeer/PubMed with GCN/GraphSAGE backbones.
- **Expected outcome**: Cold-edge AUC improves 5-15%; warm-edge AUC maintained within 1%.
- **Novelty**: 9/10 — DAHGN does degree-aware CL for node classification, ECLiP does edge CL for LP, but nobody combines degree-gated CL for cold LP.
- **Feasibility**: 1 GPU, ~2 days for full experiments.
- **Risk**: LOW
- **Contribution type**: New method
- **Pilot results**:
  - Cora: cold AUC 0.808→0.869 (+7.5%), warm 0.899→0.915 (+1.7%) ✅
  - CiteSeer: cold AUC 0.764→0.890 (+16.5%), warm 0.868→0.933 (+7.5%) ✅
  - Signal: **POSITIVE** on both datasets
- **Reviewer's likely objection**: "Why not just use loss weighting by degree?" — Need ablation showing CL provides qualitatively different improvement from reweighting.
- **Why we should do this**: Clean gap in the literature, strong empirical signal, simple and reproducible method.

### Idea 8: Calibrated Cold-Start LP — STRONG BACKUP / COMPLEMENTARY
- **Hypothesis**: Cold-start LP failure is partly an overconfidence problem. MC dropout + selective prediction can improve cold-region performance under risk-coverage evaluation.
- **Minimum experiment**: Add MC dropout to LP backbone, measure ECE/Brier/selective-AUC by degree bin.
- **Expected outcome**: Cold regions are 2-3x more uncertain; selective prediction at 70% coverage beats full-coverage AUC.
- **Novelty**: 8/10 — No paper studies calibration of LP models by degree bin.
- **Feasibility**: 1 GPU, ~1 day.
- **Risk**: LOW
- **Contribution type**: Empirical finding
- **Pilot results**:
  - Cora: cold uncertainty 0.085 vs warm 0.045 (1.9x), selective AUC@70% = 0.935 vs full 0.889 ✅
  - CiteSeer: cold uncertainty 0.086 vs warm 0.038 (2.3x), cold ECE improved by MC ✅
  - Signal: **POSITIVE** on both datasets
- **Reviewer's likely objection**: "This is evaluation work, not ML research." — Strengthen by adding abstention mechanism or uncertainty-guided augmentation.
- **Why we should do this**: Orthogonal to Idea 5 and can be combined. Cheap and robust.

### Idea 1: Relative Perturbation Law — INCONCLUSIVE
- **Hypothesis**: If high-degree nodes lose the same fraction of evidence as low-degree nodes, the LP gap should shrink.
- **Pilot results**:
  - Cora: cold improved slightly (+0.009), hot improved dramatically (+0.27). Signal: POSITIVE
  - CiteSeer: cold slightly worse (-0.004), gap widened. Signal: NEGATIVE
  - Signal: **MIXED** — Needs more careful experimental design (the equalization didn't actually change edges because target_frac was computed as median=0).
- **Next step**: Fix the equalization logic to actually mask edges. Could be a strong diagnostic if done properly.

## Eliminated Ideas

| Idea | Reason Eliminated |
|------|-------------------|
| Idea 4: Edge-Level Virtual Motif Imputation | Too close to NCNC (ICLR'24). Differentiation risk. |
| Idea 9: Sparse-Subgraph Adapter for Zero-Shot LP | HIGH risk, months of effort, competitive space. |
| Idea 10: Phase Transition for Message-Passing LP | HIGH risk, requires theory. Better as follow-up. |
| Idea 2: Variance, Not Semantics? | Weak as standalone paper. Better as ablation within Idea 5. |
| Idea 6: Arrival-Episode Pretraining | MEDIUM risk, weeks of effort, hard to differentiate from standard SSL. |
| Idea 7: Homophily-Conditioned Augmentation | Good diagnostic but narrow scope for top venue. |

## Pilot Experiment Results

| Idea | Dataset | Cold AUC (Baseline) | Cold AUC (Ours) | Δ | Warm AUC (Ours) | Signal |
|------|---------|--------------------|-----------------|----|-----------------|--------|
| 5 (Cold-Only CL) | Cora | 0.808 | 0.869 | +7.5% | 0.915 | POSITIVE |
| 5 (Cold-Only CL) | CiteSeer | 0.764 | 0.890 | +16.5% | 0.933 | POSITIVE |
| 8 (Calibration) | Cora | ECE 0.23 | ECE 0.20 | -13% | — | POSITIVE |
| 8 (Calibration) | CiteSeer | ECE 0.23 | ECE 0.22 | -4% | — | POSITIVE |
| 1 (Perturbation) | Cora | 0.808 | 0.817 | +1.1% | 0.897 | MIXED |
| 1 (Perturbation) | CiteSeer | 0.764 | 0.760 | -0.5% | 0.887 | NEGATIVE |

## Suggested Execution Plan

### Option A: Single Flagship Paper (Recommended)
Combine Ideas 5 + 8 into one paper: **"ColdCL: Uncertainty-Aware Contrastive Learning for Cold-Start Link Prediction"**
1. Cold-Only CL as the main method contribution
2. Calibration analysis as motivation + evaluation framework
3. Uncertainty-guided augmentation as the bridge (augment more where uncertainty is higher)
4. Target: KDD / WWW / ICLR

### Option B: Two-Paper Portfolio
- Paper 1: Idea 5 (method paper, KDD/WWW)
- Paper 2: Idea 8 + Idea 1 (diagnostic/evaluation paper, WSDM/WebConf)

## Next Steps
- [ ] Implement Idea 5 at full scale (multi-seed, PubMed, OGB-Collab, proper baselines)
- [ ] Add Idea 8 calibration metrics as standard evaluation
- [ ] /run-experiment to deploy full-scale experiments
- [ ] /auto-review-loop to iterate until submission-ready
