# Paper Writing Pipeline Report

**Input**: NARRATIVE_REPORT.md (from research pipeline)
**Venue**: KDD/WWW (ACM sigconf format)
**Date**: 2026-03-15

## Pipeline Summary

| Phase | Status | Output |
|-------|--------|--------|
| 1. Paper Plan | Done | GPT-5.4 structured outline |
| 2. Figures | Done | 5 auto-generated PDFs in figures/ |
| 3. LaTeX Writing | Done | 7 section files + references.bib |
| 4. Compilation | Done | main.pdf (5 pages) |
| 5. Improvement | Done | 2 review rounds |

## Improvement Scores
| Round | Score | Key Changes |
|-------|-------|-------------|
| Round 0 | 5/10 | Baseline draft |
| Round 1 | 6/10 | Tightened claims, added statistical significance, expanded limitations, fixed causal language |
| Round 2 | 6/10 | Stable — needs additional metrics/backbone for further improvement |

## Deliverables
- `paper/main.pdf` — Final paper (5 pages)
- `paper/main_round0_original.pdf` — Before improvement
- `paper/main_round1.pdf` — After round 1
- `paper/main_round2.pdf` — After round 2
- `paper/sections/` — 7 LaTeX section files
- `paper/figures/` — 5 auto-generated figures
- `paper/references.bib` — 12 references

## Remaining Issues
- Add Hits@K or AP metric for robustness (top reviewer recommendation)
- Consider adding one more backbone (e.g., GAT) for broader coverage
- Could expand to 8 pages for full KDD submission with additional ablations

## File Structure
```
paper/
├── main.tex
├── main.pdf
├── main_round0_original.pdf
├── main_round1.pdf
├── main_round2.pdf
├── references.bib
├── sections/
│   ├── introduction.tex
│   ├── related.tex
│   ├── setup.tex
│   ├── results.tex
│   ├── mechanism.tex
│   ├── alternatives.tex
│   └── discussion.tex
└── figures/
    ├── fig1_teaser.pdf
    ├── fig2_heatmap.pdf
    ├── fig3_decomposition.pdf
    ├── fig4_selective.pdf
    └── fig5_sparsity.pdf
```
