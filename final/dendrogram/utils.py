from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial import ConvexHull
from scipy.stats import chi2_contingency, mannwhitneyu
from sklearn.metrics import (adjusted_rand_score, normalized_mutual_info_score,
                              silhouette_score)


# K selection via silhouette 

def fit_hierarchical(X, k_range=range(2, 6), method='ward'):
    """Fit agglomerative clustering and select k via silhouette score.

    Computes Ward linkage once, then evaluates every k in k_range using the
    average silhouette score (higher = better separated clusters).  This is
    the hierarchical-clustering analog of BIC-based k selection used in GMM.

    Args:
        X:        2-D array of shape (n_samples, n_features).
        k_range:  Iterable of candidate k values (default 2–5).
        method:   Linkage method passed to scipy (default 'ward').

    Returns:
        Z        : linkage matrix (n-1 × 4), needed for dendrogram plots.
        labels   : 0-indexed cluster assignment array of length n_samples.
        best_k   : chosen number of clusters.
        sil_scores: dict mapping k → silhouette score (useful for diagnostics).
    """
    if isinstance(X, pd.DataFrame):
        X = X.values

    Z = linkage(X, method=method)

    sil_scores = {}
    for k in k_range:
        lbl = fcluster(Z, t=k, criterion='maxclust')
        sil_scores[k] = silhouette_score(X, lbl)

    best_k = max(sil_scores, key=sil_scores.get)
    labels = fcluster(Z, t=best_k, criterion='maxclust') - 1   # → 0-indexed
    return Z, labels, best_k, sil_scores


# Dendrogram 

def plot_dendrogram(Z, best_k, title, output_path, truncate_p=30):
    """Plot the Ward dendrogram with a horizontal cut line at the chosen k.

    The cut line is drawn at the midpoint between the merge distance that
    produces best_k clusters and the one that produces best_k-1 clusters,
    making the chosen partition visually obvious.

    Args:
        Z           : linkage matrix from fit_hierarchical.
        best_k      : number of clusters chosen by silhouette.
        title       : figure title string.
        output_path : where to save the PNG.
        truncate_p  : show only the last truncate_p merged nodes (default 30).
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Compute the cut threshold: midpoint between the two relevant merge heights
    sorted_heights = np.sort(Z[:, 2])
    cut_height = (sorted_heights[-(best_k - 1)] + sorted_heights[-best_k]) / 2

    fig, ax = plt.subplots(figsize=(12, 5))
    dendrogram(
        Z, ax=ax,
        truncate_mode='lastp', p=truncate_p,
        leaf_rotation=90, leaf_font_size=8,
        color_threshold=cut_height,
        above_threshold_color='grey',
    )
    ax.axhline(y=cut_height, color='tomato', linestyle='--', linewidth=1.5,
               label=f'Cut → k={best_k}')
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xlabel('Observations (leaf = merged node size)', fontsize=10)
    ax.set_ylabel('Ward distance', fontsize=10)
    ax.legend(fontsize=9)
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


# Silhouette score plot  

def plot_silhouette_scores(sil_scores, best_k, title, output_path):
    """Bar chart of silhouette scores for each candidate k.

    Args:
        sil_scores  : dict {k: score} from fit_hierarchical.
        best_k      : chosen k, highlighted in a different colour.
        title       : figure title.
        output_path : where to save the PNG.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ks     = sorted(sil_scores.keys())
    scores = [sil_scores[k] for k in ks]
    colors = ['tomato' if k == best_k else '#4C72B0' for k in ks]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(ks, scores, color=colors, edgecolor='white', width=0.6)
    ax.set_xlabel('Number of clusters (k)', fontsize=11)
    ax.set_ylabel('Average silhouette score', fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xticks(ks)
    ax.spines[['top', 'right']].set_visible(False)

    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color='tomato', label=f'Best k={best_k}')], fontsize=9)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


# ── 2-D cluster scatter with convex hulls 

def plot_clusters_with_hulls(canvas_2d, labels, k,
                              title, xlabel, ylabel, output_path):
    """Scatter plot of the top-2 dimensions coloured by cluster, with convex hulls.

    Convex hulls replace Gaussian ellipses because hierarchical clustering
    does not produce parametric covariance estimates.

    Args:
        canvas_2d   : (n, 2) array — the two dimensions to plot.
        labels      : 0-indexed cluster assignment.
        k           : number of clusters.
        title / xlabel / ylabel / output_path : self-explanatory.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    colors = plt.cm.tab10.colors

    fig, ax = plt.subplots(figsize=(8, 6))
    for c in range(k):
        mask  = labels == c
        pts   = canvas_2d[mask]
        color = colors[c % 10]
        ax.scatter(pts[:, 0], pts[:, 1],
                   alpha=0.6, s=30, color=color, label=f'Cluster {c}')
        if pts.shape[0] >= 3:
            try:
                hull = ConvexHull(pts)
                hull_pts = np.append(hull.vertices, hull.vertices[0])
                ax.plot(pts[hull_pts, 0], pts[hull_pts, 1],
                        color=color, linewidth=1.5, alpha=0.7)
                ax.fill(pts[hull.vertices, 0], pts[hull.vertices, 1],
                        color=color, alpha=0.08)
            except Exception:
                pass   # degenerate hull — skip silently

    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


# Phase overlay 

def plot_phases(canvas_2d, phases, title, xlabel, ylabel, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    colors = plt.cm.tab10.colors
    unique_phases = sorted(set(phases))

    fig, ax = plt.subplots(figsize=(8, 6))
    for i, phase in enumerate(unique_phases):
        mask = np.array(phases) == phase
        ax.scatter(canvas_2d[mask, 0], canvas_2d[mask, 1],
                   alpha=0.6, s=30, color=colors[i % 10], label=str(phase))
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    ax.legend(title='Phase')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


# Contingency heatmap + stats 

def contingency_heatmap(labels, group_col, col_name, title, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    group_arr = np.array(group_col)
    ct_raw  = pd.crosstab(labels, group_arr)
    ct_norm = pd.crosstab(labels, group_arr, normalize='index')

    ari      = adjusted_rand_score(labels, group_arr)
    nmi      = normalized_mutual_info_score(labels, group_arr)
    chi2_stat, p_val, _, _ = chi2_contingency(ct_raw)
    n        = len(labels)
    min_dim  = min(ct_raw.shape) - 1
    cramers_v = np.sqrt(chi2_stat / (n * max(min_dim, 1)))

    stats_text = (f'ARI={ari:.3f}   NMI={nmi:.3f}   '
                  f'χ²p={p_val:.3f}   V={cramers_v:.3f}')

    fig_w = max(6, len(ct_norm.columns) * 0.9 + 2)
    fig_h = max(3, len(ct_norm) * 0.7 + 2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    sns.heatmap(ct_norm, annot=True, fmt='.2f', cmap='Blues',
                vmin=0, vmax=1, ax=ax)
    ax.set_xlabel(col_name)
    ax.set_ylabel('Cluster')
    ax.set_title(f'{title}\n{stats_text}', fontsize=10)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


# Discriminating features

def discriminating_features(labels, X_normalized_df, biosignal_cols,
                              output_dir, prefix):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    unique_clusters = sorted(set(labels))
    records = []

    for feat in biosignal_cols:
        vals = X_normalized_df[feat].values
        cluster_means = {f'mean_cluster_{c}': vals[labels == c].mean()
                         for c in unique_clusters}
        max_d = 0.0
        min_p = 1.0
        for i, c1 in enumerate(unique_clusters):
            for c2 in unique_clusters[i + 1:]:
                g1 = vals[labels == c1]
                g2 = vals[labels == c2]
                pooled_std = np.sqrt((g1.std(ddof=1) ** 2 + g2.std(ddof=1) ** 2) / 2 + 1e-10)
                d = abs(g1.mean() - g2.mean()) / pooled_std
                _, p = mannwhitneyu(g1, g2, alternative='two-sided')
                if d > max_d:
                    max_d = d
                if p < min_p:
                    min_p = p
        row = {'feature': feat, 'cohens_d': round(max_d, 4),
               'mannwhitney_p': round(min_p, 6)}
        row.update(cluster_means)
        records.append(row)

    df_ranked = (pd.DataFrame(records)
                 .sort_values('cohens_d', ascending=False)
                 .reset_index(drop=True))
    df_ranked.to_csv(output_dir / f'{prefix}_discriminating.csv', index=False)

    top10 = df_ranked.head(10)
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(top10)))
    ax.barh(top10['feature'].values[::-1], top10['cohens_d'].values[::-1],
            color=colors[::-1])
    ax.set_xlabel("Cohen's d (max across cluster pairs)")
    ax.set_title(f"Top {len(top10)} discriminating biosignal features\n({prefix.upper()})")
    ax.grid(True, axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / f'{prefix}_top10_cohens_d.png',
                dpi=300, bbox_inches='tight')
    plt.close()
    return df_ranked