from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Ellipse
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.decomposition import PCA
from scipy.stats import chi2_contingency, mannwhitneyu


def fit_gmm_bic(X, k_range=range(2, 6), cov_types=None):
    if cov_types is None:
        cov_types = ['full', 'tied', 'diag', 'spherical']
    if isinstance(X, pd.DataFrame):
        X = X.values
    best_bic = np.inf
    best_model = None
    for cov in cov_types:
        for k in k_range:
            gmm = GaussianMixture(n_components=k, covariance_type=cov,
                                  random_state=42, n_init=5)
            gmm.fit(X)
            bic = gmm.bic(X)
            if bic < best_bic:
                best_bic = bic
                best_model = gmm
    labels = best_model.predict(X)
    return best_model, labels


def _get_component_cov(gmm_2d, component_idx):
    cov_type = gmm_2d.covariance_type
    if cov_type == 'full':
        return gmm_2d.covariances_[component_idx]
    elif cov_type == 'tied':
        return gmm_2d.covariances_
    elif cov_type == 'diag':
        return np.diag(gmm_2d.covariances_[component_idx])
    else:  # spherical
        return np.eye(2) * gmm_2d.covariances_[component_idx]


def draw_ellipse(ax, mean, cov, color, n_std=2.0):
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    w, h = 2 * n_std * np.sqrt(np.maximum(vals, 0))
    ellipse = Ellipse(xy=mean, width=w, height=h, angle=theta,
                      edgecolor=color, facecolor='none', linewidth=2, alpha=0.7)
    ax.add_patch(ellipse)


def plot_clusters_with_ellipses(canvas_2d, labels, k, cov_type,
                                 title, xlabel, ylabel, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    colors = plt.cm.tab10.colors
    gmm_2d = GaussianMixture(n_components=k, covariance_type=cov_type,
                              random_state=42, n_init=5)
    gmm_2d.fit(canvas_2d)
    fig, ax = plt.subplots(figsize=(8, 6))
    for c in range(k):
        mask = labels == c
        ax.scatter(canvas_2d[mask, 0], canvas_2d[mask, 1],
                   alpha=0.6, s=30, color=colors[c % 10], label=f'Cluster {c}')
        cov = _get_component_cov(gmm_2d, c)
        draw_ellipse(ax, gmm_2d.means_[c], cov, color=colors[c % 10])
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


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


def contingency_heatmap(labels, group_col, col_name, title, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    group_arr = np.array(group_col)
    ct_raw = pd.crosstab(labels, group_arr)
    ct_norm = pd.crosstab(labels, group_arr, normalize='index')

    ari = adjusted_rand_score(labels, group_arr)
    nmi = normalized_mutual_info_score(labels, group_arr)
    chi2_stat, p_val, _, _ = chi2_contingency(ct_raw)
    n = len(labels)
    min_dim = min(ct_raw.shape) - 1
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
                s1_var = g1.std(ddof=1) ** 2
                s2_var = g2.std(ddof=1) ** 2
                pooled_var = (s1_var + s2_var) / 2
                pooled_std = np.sqrt(pooled_var + 1e-10)  # Add small epsilon to prevent numerical instability
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
