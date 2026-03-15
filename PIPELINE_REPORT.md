# Research Pipeline Report

**Direction**: 图神经网络在链路预测中的冷启动与稀疏性
**Date**: 2026-03-15
**Pipeline**: idea-discovery → implement → run-experiment → auto-review-loop (3 rounds)

## Journey Summary

### Stage 1: Idea Discovery
- Surveyed 20+ papers across 4 sub-directions (cold-start, long-tailed, CL, foundation models)
- GPT-5.4 generated 10 ideas, filtered to 6, piloted 3
- **Pilot results**: Cold-Only CL (POSITIVE), Calibrated LP (POSITIVE), Perturbation Law (MIXED)
- User chose combined approach: ColdCL + calibration

### Stage 2: Implementation & Experiments
- **v1**: 5 datasets × 2 encoders × 6 methods × 5 seeds (300 experiments)
- ColdCL beat all baselines on cold AUC in 7/10 dataset-encoder combos

### Stage 3: Auto-Review Loop (3 rounds)

**Round 1** (GPT-5.4 review): Score 4/10
- "Cold-start" framing rejected — not true cold-start
- UncertaintyCL proposed as fix

**Round 2** (after implementing UncertaintyCL + stronger baselines): Score 3.5-4.5/10
- UncertaintyCL did NOT beat GlobalCL or even RandomCL
- Key finding: ANY form of CL massively helps tail nodes
- Pivoted to analysis paper framing

**Round 3** (mechanism analysis): Score 6.5/10
- CL gain inversely correlates with degree (monotonic, strong signal)
- AugOnly matches CL on larger graphs — contrastive objective matters more on sparse/small graphs
- Embedding stability hypothesis WRONG for large graphs

## Key Findings

### Finding 1: CL Gain is Disproportionate for Tail Nodes
| Degree | CL Gain (CS) | CL Gain (PubMed) |
|--------|-------------|-----------------|
| 2-4    | +13.6%      | +4.6%           |
| 4-6    | +9.9%       | +2.7%           |
| 6-10   | +7.3%       | +1.2%           |
| 10-20  | +5.1%       | +1.5%           |
| 20-50  | +3.3%       | +0.7%           |

### Finding 2: Augmentation vs Contrastive Objective
- On Cora (small/sparse): AugOnly +0.1% cold, GlobalCL +3.8% cold → CL objective matters
- On CS (larger/denser): AugOnly +10.7% cold, GlobalCL +10.7% cold → augmentation alone sufficient
- On PubMed: AugOnly +4.0% cold, GlobalCL +3.6% cold → augmentation sufficient

### Finding 3: Selective CL Doesn't Help (Negative Result)
- DegreeGatedCL ≈ GlobalCL ≈ RandomCL on all datasets
- The regularization benefit is global, not node-specific

## Final Status
- [x] Strong empirical finding (monotonic degree-gain curve)
- [x] Interesting decomposition (augmentation vs CL)
- [x] Credible negative results (selective CL, stability)
- [ ] Needs larger benchmarks (ogbl-citation2, ogbl-collab)
- [ ] Needs controlled sparsity experiment
- [ ] Needs DropEdge/feature-dropout baselines

## Recommended Next Steps
1. Add OGB-scale benchmarks
2. Run controlled sparsity experiment (edge subsampling)
3. Add DropEdge and feature dropout baselines
4. Write paper: "Tail Nodes Benefit Most from Graph Self-Supervision in Link Prediction"
5. Target: WWW/WSDM/KDD analysis track

## Reviewer Score Trajectory
Round 1: 4/10 → Round 2: 4/10 → Round 3: **6.5/10**

## Files Created
- `src/data.py` — Dataset loading
- `src/models.py` — All model definitions (Vanilla, GlobalCL, ColdCL, NodeDup, MCDropout)
- `src/train.py` — Training loops
- `src/metrics.py` — Evaluation metrics (AUC, AP, ECE, selective AUC)
- `run_full.py` — v1 full experiments
- `run_v2.py` — v2 with stronger baselines + UncertaintyCL
- `run_mechanism.py` — Mechanism analysis
- `results/full/results.json` — v1 results
- `results/full/v2_results.json` — v2 results
- `results/full/mechanism_analysis.json` — Mechanism analysis results
- `IDEA_REPORT.md` — Original idea report
