#!/usr/bin/env python3
"""
CGS 410: Empirical Structural Analysis of DAGs in Natural Language Dependency Trees
Project LE2 | Rahul Buraniya (240829)

Complete analysis pipeline: data loading, metric computation, null model generation,
statistical testing, and visualisation. Running end-to-end reproduces all results
and figures in the report.

Repository: github.com/rahulburaniya/cgs410-dag-analysis
"""

# ══════════════════════════════════════════════════════════════════════════════
# A1: Imports and Global Configuration
# ══════════════════════════════════════════════════════════════════════════════

import os
import random
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for saving figures
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import gaussian_kde, mannwhitneyu, shapiro, sem as scipy_sem
from scipy import stats
import conllu  # pip install conllu

# Reproducible results
np.random.seed(42)
random.seed(42)

# ── Paths: point to your downloaded UD .conllu files ─────────────────────────
TREEBANK_PATHS = {
    'English':  'ud-treebanks/en_ewt-ud-train.conllu',
    'German':   'ud-treebanks/de_gsd-ud-test.conllu',
    'Spanish':  'ud-treebanks/es_gsd-ud-test.conllu',
    'French':   'ud-treebanks/fr_gsd-ud-test.conllu',
    'Hindi':    'ud-treebanks/hi_hdtb-ud-test.conllu',
    'Mandarin': 'ud-treebanks/zh_gsd-ud-test.conllu',
}

N_SENTENCES  = 500    # sentences to sample per language
RAND_EDGE_P  = 0.15   # edge probability for random DAG null model
OUTPUT_DIR   = 'figures'
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# A2: Loading UD Treebanks and Building Dependency Graphs
# ══════════════════════════════════════════════════════════════════════════════

def load_ud_trees(filepath, n_sentences=500, min_tokens=3):
    """
    Parse a CoNLL-U treebank file and return up to n_sentences valid
    dependency trees as networkx.DiGraph objects.

    Parameters
    ----------
    filepath    : str  Path to the .conllu file.
    n_sentences : int  Maximum number of sentences to load.
    min_tokens  : int  Skip sentences shorter than this (avoids trivial graphs).

    Returns
    -------
    list of nx.DiGraph
    """
    trees = []
    with open(filepath, encoding='utf-8') as f:
        for sentence in conllu.parse_incr(f):
            if len(trees) >= n_sentences:
                break

            G = nx.DiGraph()
            for token in sentence:
                # Skip multi-word tokens (id is a tuple like (1, 2))
                # and empty nodes (id is a float like 1.1)
                if not isinstance(token['id'], int):
                    continue
                # Skip punctuation tokens
                if token['upos'] == 'PUNCT':
                    continue

                G.add_node(token['id'], form=token['form'], upos=token['upos'])
                head = token['head']
                if head is not None and head != 0:  # 0 = root
                    G.add_edge(head, token['id'], deprel=token['deprel'])

            # Validity checks
            if G.number_of_nodes() < min_tokens:
                continue
            if not nx.is_weakly_connected(G):
                continue
            roots = [n for n, d in G.in_degree() if d == 0]
            if len(roots) != 1:
                continue

            trees.append(G)

    print(f"  Loaded {len(trees)} trees from {os.path.basename(filepath)}")
    return trees


# ── Load all six languages ───────────────────────────────────────────────────
print("Loading treebanks...")
lang_trees = {}
for lang, path in TREEBANK_PATHS.items():
    lang_trees[lang] = load_ud_trees(path, n_sentences=N_SENTENCES)
print(f"Total NL trees: {sum(len(v) for v in lang_trees.values())}")


# ══════════════════════════════════════════════════════════════════════════════
# A3: Computing Structural Metrics
# ══════════════════════════════════════════════════════════════════════════════

def compute_arity(G):
    """Mean out-degree (arity): alpha = (1/|V|) * sum outdeg(v)"""
    n = G.number_of_nodes()
    if n == 0:
        return 0.0
    return sum(d for _, d in G.out_degree()) / n


def compute_depth(G):
    """Tree depth: longest directed path from root to any leaf."""
    roots = [n for n, d in G.in_degree() if d == 0]
    if not roots:
        return 0
    root = roots[0]
    path_lengths = nx.single_source_shortest_path_length(G, root)
    return max(path_lengths.values())


def compute_density(G):
    """Graph density: rho = |E| / (|V|(|V|-1)/2)"""
    n = G.number_of_nodes()
    if n < 2:
        return 0.0
    max_edges = n * (n - 1) / 2
    return G.number_of_edges() / max_edges


def compute_all_metrics(trees):
    """Apply all three metrics to a list of graphs. Returns dict of arrays."""
    return dict(
        arity   = np.array([compute_arity(g)  for g in trees]),
        depth   = np.array([compute_depth(g)   for g in trees]),
        density = np.array([compute_density(g) for g in trees]),
    )


# ── Apply to all NL trees ────────────────────────────────────────────────────
print("Computing NL metrics...")
lang_metrics = {}
for lang, trees in lang_trees.items():
    lang_metrics[lang] = compute_all_metrics(trees)
    a = lang_metrics[lang]['arity']
    d = lang_metrics[lang]['depth']
    print(f"  {lang:10s}: n={len(trees)}, "
          f"arity={np.mean(a):.3f}, "
          f"depth={np.mean(d):.3f}, "
          f"density={np.mean(lang_metrics[lang]['density']):.4f}")

# Pool all languages for overall comparison
nl_arity   = np.concatenate([lang_metrics[l]['arity']   for l in lang_metrics])
nl_depth   = np.concatenate([lang_metrics[l]['depth']    for l in lang_metrics])
nl_density = np.concatenate([lang_metrics[l]['density']  for l in lang_metrics])


# ══════════════════════════════════════════════════════════════════════════════
# A4: Random DAG Null Model Generation
# ══════════════════════════════════════════════════════════════════════════════

def random_dag(n, p=RAND_EDGE_P):
    """
    Generate a random DAG on n nodes using topological edge sampling.

    Procedure:
      1. Assign nodes a topological order {0, 1, ..., n-1}.
      2. For each pair (i, j) with i < j, add directed edge i->j
         with probability p.
    This guarantees acyclicity by construction. Expected density = p.

    Parameters
    ----------
    n : int    Number of nodes (= sentence length of matched NL tree).
    p : float  Edge probability. Default 0.15.

    Returns
    -------
    nx.DiGraph
    """
    G = nx.DiGraph()
    G.add_nodes_from(range(n))
    for i in range(n):
        for j in range(i + 1, n):
            if random.random() < p:
                G.add_edge(i, j)
    return G


# ── Generate one random DAG per NL sentence ──────────────────────────────────
print("Generating random DAG null models...")
rand_graphs = []
for lang, trees in lang_trees.items():
    for tree in trees:
        n = tree.number_of_nodes()
        rand_graphs.append(random_dag(n, p=RAND_EDGE_P))
print(f"Generated {len(rand_graphs)} random DAGs")

# Compute metrics for all random DAGs
rand_metrics = compute_all_metrics(rand_graphs)
rand_arity   = rand_metrics['arity']
rand_depth   = rand_metrics['depth']
rand_density = rand_metrics['density']
print(f"Random DAG means — arity: {np.mean(rand_arity):.3f}, "
      f"depth: {np.mean(rand_depth):.3f}, "
      f"density: {np.mean(rand_density):.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# A5: Statistical Hypothesis Testing
# ══════════════════════════════════════════════════════════════════════════════

def cohens_d(a, b):
    """Standardised mean difference (Cohen's d) between two samples."""
    pooled_sd = np.sqrt((np.std(a, ddof=1)**2 + np.std(b, ddof=1)**2) / 2)
    return (np.mean(a) - np.mean(b)) / pooled_sd if pooled_sd > 0 else 0.0


ALPHA_BONFERRONI = 0.05 / 3  # 3 simultaneous tests

# ── Normality check ──────────────────────────────────────────────────────────
print("\n══ Normality check (Shapiro-Wilk on 200-sample subset) ══")
for label, arr in [("NL Arity", nl_arity),   ("NL Depth", nl_depth),
                   ("NL Density", nl_density), ("Rand Arity", rand_arity),
                   ("Rand Depth", rand_depth), ("Rand Density", rand_density)]:
    sample = arr[np.random.choice(len(arr), min(200, len(arr)), replace=False)]
    stat, p = shapiro(sample)
    print(f"  {label:15s}: W={stat:.4f}, p={p:.4e} "
          f"{'Normal' if p > 0.05 else 'NON-normal'}")

# ── Mann-Whitney U tests ────────────────────────────────────────────────────
print(f"\n══ Mann-Whitney U tests (two-sided, Bonferroni threshold = "
      f"{ALPHA_BONFERRONI:.4f}) ══")
results = {}
for label, nl_arr, rd_arr in [
    ("Arity",   nl_arity,   rand_arity),
    ("Depth",   nl_depth,   rand_depth),
    ("Density", nl_density, rand_density),
]:
    u_stat, p_val = mannwhitneyu(nl_arr, rd_arr, alternative='two-sided')
    d = cohens_d(nl_arr, rd_arr)
    significant = p_val < ALPHA_BONFERRONI
    results[label] = dict(U=u_stat, p=p_val, d=d, sig=significant)
    print(f"  {label:8s}: NL_mean={np.mean(nl_arr):.3f} +/- {np.std(nl_arr):.3f}  "
          f"RD_mean={np.mean(rd_arr):.3f} +/- {np.std(rd_arr):.3f}  "
          f"U={u_stat:.0f}  p={p_val:.3e}  "
          f"d={d:.2f}  {'SIGNIFICANT' if significant else 'not significant'}")

print("\nAll three hypotheses:", "SUPPORTED"
      if all(r['sig'] for r in results.values()) else "NOT all supported")


# ══════════════════════════════════════════════════════════════════════════════
# A6: Full Visualisation Pipeline
# ══════════════════════════════════════════════════════════════════════════════

NL_COLOR   = '#2E86AB'
RAND_COLOR = '#E07B39'

LANG_COLORS = {
    'English':  '#4C72B0',
    'German':   '#C44E52',
    'Spanish':  '#55A868',
    'French':   '#8172B2',
    'Hindi':    '#DD8452',
    'Mandarin': '#8C8C8C',
}
LANGUAGES = list(LANG_COLORS.keys())


# ── Figure 1: Distribution histograms (arity, depth, density) ────────────────
def plot_distributions():
    """Three-panel density histogram comparing NL trees and random DAGs."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    fig.subplots_adjust(wspace=0.35, left=0.07, right=0.97,
                        top=0.88, bottom=0.14)

    panels = [
        (nl_arity,   rand_arity,   "Mean Arity (out-degree)", "Arity",   (0, 12)),
        (nl_depth,   rand_depth,   "Tree Depth (root-to-leaf)", "Depth", (0, 22)),
        (nl_density, rand_density, "Graph Density",   "Density",         (0, 0.6)),
    ]

    for ax, (nl, rd, title, xlabel, xlim) in zip(axes, panels):
        ax.hist(nl, bins=35, range=xlim, color=NL_COLOR,
                alpha=0.65, density=True, label="Natural Language")
        ax.hist(rd, bins=35, range=xlim, color=RAND_COLOR,
                alpha=0.65, density=True, label="Random DAG")
        ax.axvline(np.mean(nl), color=NL_COLOR,   lw=2, ls='--')
        ax.axvline(np.mean(rd), color=RAND_COLOR,  lw=2, ls='--')
        ax.set_title(title, fontsize=11, fontweight='bold', pad=6)
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel("Density", fontsize=10)
        ax.set_xlim(xlim)
        ax.spines[['top', 'right']].set_visible(False)

    nl_patch   = mpatches.Patch(color=NL_COLOR,   alpha=0.65, label="Natural Language")
    rand_patch = mpatches.Patch(color=RAND_COLOR, alpha=0.65, label="Random DAG")
    fig.legend(handles=[nl_patch, rand_patch], loc='upper center', ncol=2,
               fontsize=10, bbox_to_anchor=(0.5, 0.98), frameon=False)

    path = os.path.join(OUTPUT_DIR, 'fig1_distributions.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {path}')


# ── Figure 2: KDE + Boxplot pairs ───────────────────────────────────────────
def plot_kde_boxplot(nl_arr, rd_arr, xlabel, xlim, filename):
    """Side-by-side KDE density plot and horizontal boxplot."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    fig.subplots_adjust(wspace=0.32, left=0.08, right=0.97,
                        top=0.88, bottom=0.14)

    # KDE panel
    xs = np.linspace(xlim[0], xlim[1], 500)
    for arr, colour, label in [
        (nl_arr, NL_COLOR,   'Natural Language'),
        (rd_arr, RAND_COLOR, 'Random DAG'),
    ]:
        kde = gaussian_kde(arr, bw_method=0.3)
        ax1.fill_between(xs, kde(xs), alpha=0.45, color=colour)
        ax1.plot(xs, kde(xs), color=colour, lw=1.8, label=label)
        ax1.axvline(np.mean(arr), color=colour, lw=1.5, ls='--', alpha=0.8)
    ax1.set_xlabel(xlabel, fontsize=10)
    ax1.set_ylabel('Density', fontsize=10)
    ax1.set_xlim(xlim)
    ax1.set_title(f'{xlabel} (KDE)', fontsize=11, fontweight='bold')
    ax1.legend(fontsize=9, frameon=False)
    ax1.spines[['top', 'right']].set_visible(False)

    # Boxplot panel
    bp = ax2.boxplot(
        [nl_arr, rd_arr], vert=False, patch_artist=True, widths=0.5,
        flierprops=dict(marker='o', markersize=2.5, alpha=0.35),
        medianprops=dict(color='black', lw=2),
    )
    for box, colour in zip(bp['boxes'], [NL_COLOR, RAND_COLOR]):
        box.set_facecolor(colour)
        box.set_alpha(0.72)
    for w in bp['whiskers']:
        w.set_color('#444')
    for c in bp['caps']:
        c.set_color('#444')
    ax2.set_yticks([1, 2])
    ax2.set_yticklabels(['Natural Language', 'Random DAG'], fontsize=9)
    ax2.set_xlabel(xlabel, fontsize=10)
    ax2.set_title(f'{xlabel} (Boxplot)', fontsize=11, fontweight='bold')
    ax2.spines[['top', 'right']].set_visible(False)

    out = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {out}')


# ── Figure 3: Per-language mean arity bar chart ──────────────────────────────
def plot_per_language_arity():
    """Bar chart of mean arity per language with 95% CI error bars."""
    means = [np.mean(lang_metrics[l]['arity']) for l in LANGUAGES]
    sems  = [scipy_sem(lang_metrics[l]['arity']) * 1.96 for l in LANGUAGES]

    fig, ax = plt.subplots(figsize=(9, 4.2))
    x = np.arange(len(LANGUAGES))
    ax.bar(x, means, yerr=sems, capsize=4,
           color=[LANG_COLORS[l] for l in LANGUAGES], alpha=0.85,
           error_kw=dict(ecolor='#333', lw=1.5))

    rand_mean = float(np.mean(rand_arity))
    ax.axhline(rand_mean, color=RAND_COLOR, lw=2, ls='--',
               label=f'Random DAG mean ({rand_mean:.2f})')

    ax.set_xticks(x)
    ax.set_xticklabels(LANGUAGES, rotation=30, ha='right', fontsize=10)
    ax.set_ylabel("Mean Arity", fontsize=11)
    ax.set_title("Mean Arity per Language vs. Random DAG Baseline",
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=10, frameon=False)
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_ylim(0, rand_mean * 1.6)

    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'fig3_per_language_arity.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {path}')


# ── Figure 4: Inter-language violin plots ────────────────────────────────────
def plot_violin_interlanguage():
    """Three violin plots showing per-language distributions of all metrics."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.subplots_adjust(wspace=0.38, left=0.06, right=0.98,
                        top=0.88, bottom=0.18)

    metrics_list = [
        ('arity',   'Arity',   (0, 5)),
        ('depth',   'Depth',   (0, 20)),
        ('density', 'Density', (0, 0.12)),
    ]

    for ax, (key, label, ylim) in zip(axes, metrics_list):
        data_arrays = [lang_metrics[l][key] for l in LANGUAGES]
        colour_list = [LANG_COLORS[l] for l in LANGUAGES]

        parts = ax.violinplot(
            data_arrays, positions=range(len(LANGUAGES)),
            showmedians=True, showextrema=False,
        )
        for body, colour in zip(parts['bodies'], colour_list):
            body.set_facecolor(colour)
            body.set_alpha(0.72)
        parts['cmedians'].set_color('black')
        parts['cmedians'].set_linewidth(1.8)

        ax.set_xticks(range(len(LANGUAGES)))
        ax.set_xticklabels(LANGUAGES, rotation=38, ha='right', fontsize=8.5)
        ax.set_ylabel(label, fontsize=10)
        ax.set_ylim(ylim)
        ax.set_title(f'Inter-language {label}', fontsize=11, fontweight='bold')
        ax.spines[['top', 'right']].set_visible(False)

    out = os.path.join(OUTPUT_DIR, 'fig4_violin.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {out}')


# ── Figure 5: Complexity space scatter plot (Depth vs Arity) ─────────────────
def plot_complexity_space():
    """Scatter: Depth vs Arity for each NL language + random DAGs (grey)."""
    fig, ax = plt.subplots(figsize=(8, 5.5))

    for lang in LANGUAGES:
        a_arr = lang_metrics[lang]['arity']
        d_arr = lang_metrics[lang]['depth']
        idx = np.random.choice(len(a_arr), min(80, len(a_arr)), replace=False)
        ax.scatter(a_arr[idx], d_arr[idx],
                   color=LANG_COLORS[lang], alpha=0.55, s=22, label=lang)

    ri = np.random.choice(len(rand_arity), 250, replace=False)
    ax.scatter(rand_arity[ri], rand_depth[ri],
               color='#AAAAAA', alpha=0.3, s=18, marker='x', label='Random DAG')

    ax.set_xlabel('Arity (mean out-degree)', fontsize=11)
    ax.set_ylabel('Depth (longest root-to-leaf path)', fontsize=11)
    ax.set_title('Complexity Space: Depth vs Arity', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8.2, ncol=2, frameon=True, framealpha=0.85,
              loc='upper right', markerscale=1.4)
    ax.spines[['top', 'right']].set_visible(False)

    fig.tight_layout()
    out = os.path.join(OUTPUT_DIR, 'fig5_complexity_space.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {out}')


# ── Figure 6: Depth vs Density scatter plot ──────────────────────────────────
def plot_depth_density_scatter(n_sample=600):
    """Scatter: tree depth vs. graph density for NL and random DAGs."""
    fig, ax = plt.subplots(figsize=(7, 4.5))

    n_nl   = min(n_sample, len(nl_depth))
    n_rand = min(n_sample, len(rand_depth))
    idx_nl   = np.random.choice(len(nl_depth),   n_nl,   replace=False)
    idx_rand = np.random.choice(len(rand_depth), n_rand, replace=False)

    ax.scatter(nl_depth[idx_nl],     nl_density[idx_nl],
               c=NL_COLOR,   alpha=0.45, s=18, label="Natural Language")
    ax.scatter(rand_depth[idx_rand], rand_density[idx_rand],
               c=RAND_COLOR, alpha=0.35, s=18, label="Random DAG")

    ax.set_xlabel("Tree Depth", fontsize=11)
    ax.set_ylabel("Graph Density", fontsize=11)
    ax.set_title("Tree Depth vs. Graph Density", fontsize=12, fontweight='bold')
    ax.legend(fontsize=10, frameon=False)
    ax.spines[['top', 'right']].set_visible(False)

    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'fig6_depth_vs_density.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {path}')


# ══════════════════════════════════════════════════════════════════════════════
# Run all visualisations
# ══════════════════════════════════════════════════════════════════════════════
print("\nGenerating figures...")

# Fig 1: Distribution histograms
plot_distributions()

# Fig 2a-c: KDE + Boxplot pairs
plot_kde_boxplot(nl_arity,   rand_arity,   'Arity',   (0, 12),   'fig2a_arity_kde.png')
plot_kde_boxplot(nl_depth,   rand_depth,   'Depth',   (0, 32),   'fig2b_depth_kde.png')
plot_kde_boxplot(nl_density, rand_density, 'Density', (0, 0.85), 'fig2c_density_kde.png')

# Fig 3: Per-language arity
plot_per_language_arity()

# Fig 4: Inter-language violin plots
plot_violin_interlanguage()

# Fig 5: Complexity space
plot_complexity_space()

# Fig 6: Depth vs Density
plot_depth_density_scatter()

print(f"\nAll figures saved to '{OUTPUT_DIR}/'. Analysis complete.")
