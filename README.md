# CGS 410: Empirical Structural Analysis of DAGs in Natural Language Dependency Trees

**Project LE2** | Rahul Buraniya (240829)

## Research Question

Do natural language dependency trees exhibit systematically constrained structural properties — in **arity**, **depth**, and **density** — compared to randomly generated DAGs of the same size?

## Overview

This project analyses 3,000 dependency trees (500 sentences × 6 typologically diverse languages) from Universal Dependencies v2.13 and compares their graph-theoretic properties against a matched random DAG null model. All three hypotheses are strongly supported: natural language trees are significantly lower in arity, shallower, and sparser than random baselines.

## Languages

| Language | Family | Word Order | Morphology | UD Treebank |
|----------|--------|------------|------------|-------------|
| English | Indo-European (Germanic) | SVO | Analytic | EWT |
| German | Indo-European (Germanic) | SOV/V2 | Fusional | GSD |
| Spanish | Indo-European (Romance) | SVO | Fusional | GSD |
| French | Indo-European (Romance) | SVO | Fusional | GSD |
| Hindi | Indo-Aryan | SOV | Fusional | HDTB |
| Mandarin | Sino-Tibetan | SVO | Isolating | GSD |

## Setup

### Prerequisites

```bash
pip install conllu networkx scipy matplotlib numpy
```

### Data

Place your `.conllu` files in a `ud-treebanks/` directory:

```
ud-treebanks/
├── en_ewt-ud-train.conllu
├── de_gsd-ud-test.conllu
├── es_gsd-ud-test.conllu
├── fr_gsd-ud-test.conllu
├── hi_hdtb-ud-test.conllu
└── zh_gsd-ud-test.conllu
```

### Run

```bash
python main.py
```

All figures will be saved to `figures/`.

## Output Figures

| Figure | Description |
|--------|-------------|
| `fig1_distributions.png` | Three-panel density histograms (arity, depth, density) |
| `fig2a_arity_kde.png` | Arity KDE + boxplot |
| `fig2b_depth_kde.png` | Depth KDE + boxplot |
| `fig2c_density_kde.png` | Density KDE + boxplot |
| `fig3_per_language_arity.png` | Per-language mean arity bar chart with 95% CI |
| `fig4_violin.png` | Inter-language violin plots |
| `fig5_complexity_space.png` | Depth vs Arity scatter (complexity space) |
| `fig6_depth_vs_density.png` | Depth vs Density scatter |

## References

- de Marneffe et al. (2021). Universal Dependencies. *Computational Linguistics*, 47(2), 255–308.
- Futrell, R., Mahowald, K., & Gibson, E. (2015). Large-scale evidence of dependency length minimization in 37 languages. *PNAS*, 112(33), 10336–10341.
- Gibson, E. (1998). Linguistic complexity: Locality of syntactic dependencies. *Cognition*, 68(1), 1–76.
- Hauser, M. D., Chomsky, N., & Fitch, W. T. (2002). The faculty of language. *Science*, 298, 1569–1579.
