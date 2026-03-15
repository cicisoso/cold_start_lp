# Narrative Report: Tail Nodes Benefit Most from Graph Self-Supervision in Link Prediction

## Core Claims

### Claim 1: Graph self-supervision (contrastive learning and augmentation) disproportionately improves link prediction for tail (low-degree) nodes
- CL gain is monotonically decreasing with node degree
- On CS: deg 2-4 gains +13.6%, deg 20-50 gains +3.3%
- On PubMed: deg 2-4 gains +4.6%, deg 50+ gains -1.5%
- Consistent across 7 datasets (Cora, CiteSeer, PubMed, CS, Photo, ogbl-collab), 2 backbones (GCN, SAGE), 5 seeds

### Claim 2: The benefit decomposes into augmentation regularization + contrastive objective, with their relative importance depending on graph sparsity
- On small/sparse graphs (Cora): AugOnly gives +1.5% cold AUC, GlobalCL gives +3.9% → contrastive objective adds +2.4%
- On large/dense graphs (ogbl-collab): AugOnly gives +8.8% cold AUC, GlobalCL gives +8.2% → contrastive objective adds nothing
- Crossover: as graph density increases, augmentation alone becomes sufficient

### Claim 3: Selective/targeted CL does not outperform global CL (negative result)
- DegreeGatedCL ≈ GlobalCL ≈ RandomCL across all datasets
- UncertaintyCL (MC dropout-guided) is slightly worse
- The regularization benefit is global, not node-specific

### Claim 4: Non-CL approaches (loss reweighting, focal loss) provide much smaller tail-node gains than self-supervision
- Reweight: cold AUC gains of 0-3% across datasets
- FocalLoss: cold AUC gains of 0-1%
- GlobalCL: cold AUC gains of 4-16%

## Experimental Setup

### Datasets
| Dataset | Nodes | Edges | Features | Cold% (deg≤5) | Type |
|---------|-------|-------|----------|---------------|------|
| Cora | 2,708 | 5,278 | 1,433 | 39% | Citation |
| CiteSeer | 3,327 | 4,552 | 3,703 | 65% | Citation |
| PubMed | 19,717 | 44,324 | 500 | 63% | Citation |
| CS | 18,333 | 81,894 | 6,805 | 15% | Coauthor |
| Photo | 7,650 | 119,081 | 745 | 6% | Co-purchase |
| ogbl-collab | 235,868 | 280,729 | 128 | ~30% | Collaboration |

### Methods compared
1. **Vanilla**: Standard GCN/SAGE + dot-product LP
2. **Reweight**: Degree-weighted loss (3× weight on cold edges)
3. **FocalLoss**: γ=2.0 focal loss for hard examples
4. **GlobalCL**: Full-graph augmentation + InfoNCE
5. **DegreeGatedCL**: CL only on nodes with deg≤5
6. **RandomCL**: CL on random 30% of nodes
7. **UncertaintyCL**: CL on top-30% uncertain nodes (MC dropout)
8. **AugOnly**: Train on augmented graph, no contrastive loss
9. **NodeDup**: Node duplication baseline

### Evaluation
- AUC stratified by min-endpoint training-graph degree
- 5 seeds, mean±std reported
- Degree bins: isolated (0-1), cold (2-5), warm (6-20), hot (>20)
- Fine-grained: 2-4, 4-6, 6-10, 10-20, 20-50, 50+

## Key Results

### Table 1: GCN backbone, Overall/Cold/Warm AUC (5 seeds)

#### Cora
| Method | Overall | Cold(2-5) | Warm(6-20) |
|--------|---------|-----------|------------|
| Vanilla | .900±.012 | .843±.030 | .907±.016 |
| Reweight | .888±.007 | .820±.043 | .904±.012 |
| FocalLoss | .895±.007 | .843±.024 | .898±.011 |
| GlobalCL | .922±.002 | .885±.017 | .926±.008 |
| DegreeGatedCL | .920±.008 | .884±.023 | .928±.017 |
| RandomCL | .924±.003 | .889±.010 | .927±.006 |
| UncertaintyCL | .916±.006 | .880±.010 | .921±.012 |

#### CiteSeer
| Method | Overall | Cold(2-5) | Warm(6-20) |
|--------|---------|-----------|------------|
| Vanilla | .864±.007 | .782±.012 | .907±.030 |
| Reweight | .888±.008 | .811±.011 | .940±.007 |
| FocalLoss | .866±.023 | .794±.035 | .919±.011 |
| GlobalCL | .926±.008 | .883±.010 | .952±.011 |
| DegreeGatedCL | .922±.002 | .882±.005 | .951±.002 |
| RandomCL | .923±.009 | .882±.014 | .951±.011 |
| UncertaintyCL | .916±.010 | .871±.019 | .942±.012 |

#### PubMed
| Method | Overall | Cold(2-5) | Warm(6-20) |
|--------|---------|-----------|------------|
| Vanilla | .926±.002 | .898±.005 | .864±.001 |
| Reweight | .945±.002 | .926±.004 | .899±.003 |
| FocalLoss | .923±.005 | .902±.006 | .847±.003 |
| GlobalCL | .949±.012 | .945±.005 | .875±.054 |
| DegreeGatedCL | .954±.004 | .938±.006 | .916±.005 |
| RandomCL | .957±.002 | .945±.004 | .912±.004 |
| UncertaintyCL | .938±.017 | .937±.013 | .838±.059 |

#### CS
| Method | Overall | Cold(2-5) | Warm(6-20) |
|--------|---------|-----------|------------|
| Vanilla | .895±.023 | .750±.049 | .864±.030 |
| Reweight | .948±.004 | .877±.014 | .935±.006 |
| FocalLoss | .950±.006 | .893±.013 | .936±.006 |
| GlobalCL | .955±.001 | .915±.005 | .944±.003 |
| DegreeGatedCL | .952±.002 | .916±.010 | .944±.002 |
| RandomCL | .955±.003 | .908±.011 | .944±.004 |
| UncertaintyCL | .952±.004 | .897±.008 | .940±.005 |

### Table 2: OGB benchmark (ogbl-collab, 3 seeds)
| Method | Overall AUC | Cold(0-5) AUC |
|--------|-----------|--------------|
| Vanilla | .799±.001 | .701±.004 |
| AugOnly | .848±.002 | .789±.003 |
| GlobalCL | .833±.006 | .783±.010 |

### Figure 1: CL gain vs node degree (monotonic decrease)
CS dataset, seed 0:
- deg 2-4: +13.6%
- deg 4-6: +9.9%
- deg 6-10: +7.3%
- deg 10-20: +5.1%
- deg 20-50: +3.3%
- deg 50+: +4.7%

PubMed dataset:
- deg 2-4: +4.6%
- deg 4-6: +2.7%
- deg 6-10: +1.2%
- deg 10-20: +1.5%
- deg 20-50: +0.7%
- deg 50+: -1.5%

### Figure 2: Augmentation vs CL decomposition
| Dataset | AugOnly Cold Gain | CL Cold Gain | CL-AugOnly |
|---------|------------------|-------------|------------|
| Cora | +1.5% | +3.9% | +2.4% |
| CiteSeer | +4.2% | +5.2% | +1.0% |
| PubMed | +4.0% | +3.6% | -0.4% |
| CS | +10.7% | +10.7% | 0.0% |
| ogbl-collab | +8.8% | +8.2% | -0.6% |

### Figure 3: Sparsity experiment
CS, keep ratio vs CL-AugOnly gap on cold AUC:
- 100% edges: gap = +0.7%
- 75% edges: gap = +2.8%
- 50% edges: gap = +1.5%
- 25% edges: gap = +0.9%

Cora:
- 100%: gap = +2.4%
- 75%: gap = +3.1%
- 50%: gap = +1.3%
- 25%: gap = +2.9%

## Related Work
- NodeDup (ICLR'25 rejected): node duplication augmentation
- SpotTarget (WSDM'24): target-link inclusion hurts low-degree nodes
- LTLP (KDD'24): common-neighbor count matters more than degree
- NCNC (ICLR'24): common neighbor completion
- DAHGN (KBS'24): degree-aware CL for node classification
- ECLiP (IPM'24): edge-level CL for LP
- ULTRA (ICLR'24): zero-shot LP foundation model
- CoGCL (arXiv'24): discrete-code CL for sparse recommendation
