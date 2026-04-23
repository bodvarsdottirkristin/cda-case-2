import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import umap
from sklearn.decomposition import PCA, SparsePCA
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from dim_reduction.utils.high_corr import highly_corr

FIGURES_DIR = Path(__file__).resolve().parent / 'figures'
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR = PROJECT_ROOT / 'data' / 'processed'
RESULTS_DIR = PROJECT_ROOT / 'results' / 'tables'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

META_COLS = ['Round', 'Phase', 'Individual', 'Puzzler', 'Cohort']
QUESTIONNAIRE_COLS = [
    'Frustrated', 'upset', 'hostile', 'alert', 'ashamed', 'inspired',
    'nervous', 'attentive', 'afraid', 'active', 'determined'
]


def load_data():
    return pd.read_csv(PROCESSED_DIR / 'HR_data_2.csv')


def preprocess(df):
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    ID_COLS = ['Unnamed: 0', 'original_ID']
    biosignal_cols = [c for c in numeric_cols if c not in META_COLS + QUESTIONNAIRE_COLS + ID_COLS]

    # 1. Phase-wise NaN imputation
    df = df.copy()
    df[biosignal_cols] = df.groupby('Phase')[biosignal_cols].transform(
        lambda x: x.fillna(x.mean())
    )

    # 2. Drop highly correlated features (>95% pairwise correlation)
    redundant = highly_corr(df[biosignal_cols], perf=0.95)
    remaining = [c for c in biosignal_cols if c not in redundant]

    # 3. Individual-wise standardization
    def safe_standardize(series):
        std = series.std(ddof=0)
        if pd.isna(std) or std == 0:
            return pd.Series(np.zeros(len(series)), index=series.index)
        return (series - series.mean()) / std

    df_scaled = df.copy()
    for col in remaining:
        df_scaled[col] = df_scaled.groupby('Individual')[col].transform(safe_standardize)

    X = df_scaled[remaining]
    meta = df_scaled[[c for c in META_COLS if c in df_scaled.columns]]

    df_scaled.to_csv(PROCESSED_DIR / 'HR_data_gmm_preprocessed.csv', index=False)
    print(f"Preprocessed: {X.shape[0]} rows, {len(remaining)} biosignal features")
    return X, meta, remaining


def reduce(X, remaining_biosignals):
    X_arr = X.values if isinstance(X, pd.DataFrame) else X

    # 1. Standard PCA — select n_components by 80% variance threshold
    pca_full = PCA(random_state=42)
    pca_full.fit(X_arr)
    cum_var = np.cumsum(pca_full.explained_variance_ratio_)
    mask = cum_var >= 0.80
    n_components = int(np.argmax(mask) + 1) if mask.any() else len(cum_var)
    print(f"PCA: {n_components} components explain ≥80% variance")

    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X_arr)

    # Scree + cumulative variance plots
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].bar(range(1, n_components + 1), pca.explained_variance_ratio_)
    axes[0].set(xlabel='PC', ylabel='Explained variance ratio', title='PCA scree plot')
    axes[1].plot(range(1, len(cum_var) + 1), cum_var, marker='o')
    axes[1].axhline(0.80, linestyle='--', label='80% threshold')
    axes[1].axvline(n_components, linestyle='--', label=f'{n_components} components')
    axes[1].set(xlabel='Components', ylabel='Cumulative variance', title='PCA cumulative variance')
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'gmm_pca_variance.png', dpi=300)
    plt.close()

    # 2. SparsePCA — same n_components, alpha=1
    print("Fitting SparsePCA (this may take a minute)...")
    spca = SparsePCA(n_components=n_components, alpha=1, random_state=42, max_iter=1000)
    X_spca = spca.fit_transform(X_arr)

    # SparsePCA component loadings heatmap
    components_df = pd.DataFrame(
        spca.components_,
        columns=remaining_biosignals,
        index=[f'PC{i+1}' for i in range(n_components)]
    )
    plt.figure(figsize=(max(10, len(remaining_biosignals) * 0.4), n_components * 0.8 + 2))
    sns.heatmap(components_df, center=0, cmap='RdBu_r', linewidths=0.3)
    plt.title('SparsePCA component loadings')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'gmm_spca_loadings.png', dpi=300)
    plt.close()

    # 3. UMAP 10D — for clustering
    print("Fitting UMAP 10D...")
    reducer_10d = umap.UMAP(
        n_components=10,
        n_neighbors=15,
        min_dist=0.1,
        metric='euclidean',
        random_state=42
    )
    X_umap = reducer_10d.fit_transform(X_arr)

    # 4. PCA 2D — visualization canvas for PCA and SparsePCA
    pca_2d = PCA(n_components=2, random_state=42)
    X_2d = pca_2d.fit_transform(X_arr)

    # 5. UMAP 2D — visualization canvas for UMAP
    print("Fitting UMAP 2D...")
    reducer_2d = umap.UMAP(
        n_components=2,
        n_neighbors=15,
        min_dist=0.1,
        metric='euclidean',
        random_state=42
    )
    X_umap_2d = reducer_2d.fit_transform(X_arr)

    return X_pca, X_spca, X_umap, X_2d, X_umap_2d, n_components


def fit_gmm(X_pca, X_spca, X_umap=None, k_range=range(2, 9)):
    cov_types = ['full', 'tied', 'diag', 'spherical']
    results = {}

    datasets = [('pca', X_pca), ('spca', X_spca)]
    if X_umap is not None:
        datasets.append(('umap', X_umap))

    for name, X in datasets:
        print(f"\nFitting GMM grid on {name.upper()}...")
        best_bic = np.inf
        best_model = None
        bic_grid = {cov: [] for cov in cov_types}
        aic_grid = {cov: [] for cov in cov_types}

        for cov in cov_types:
            for k in k_range:
                gmm = GaussianMixture(
                    n_components=k,
                    covariance_type=cov,
                    random_state=42,
                    n_init=5
                )
                gmm.fit(X)
                bic = gmm.bic(X)
                aic = gmm.aic(X)
                bic_grid[cov].append(bic)
                aic_grid[cov].append(aic)
                if bic < best_bic:
                    best_bic = bic
                    best_model = gmm

        if best_model is None:
            raise ValueError(f"k_range was empty — no GMM fitted for {name.upper()}")
        labels = best_model.predict(X)
        probs = best_model.predict_proba(X)
        k_list = list(k_range)

        # BIC + AIC plot
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        colors = plt.cm.tab10.colors
        for i, cov in enumerate(cov_types):
            axes[0].plot(k_list, bic_grid[cov], marker='o', label=cov, color=colors[i])
            axes[1].plot(k_list, aic_grid[cov], marker='o', label=cov, color=colors[i])
        for ax, metric in zip(axes, ['BIC', 'AIC']):
            ax.set(xlabel='k (number of components)', ylabel=metric,
                   title=f'{metric} — GMM on {name.upper()}')
            ax.legend()
            ax.grid(True)
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / f'gmm_{name}_bic_aic.png', dpi=300)
        plt.close()

        print(f"  Best: k={best_model.n_components}, "
              f"cov={best_model.covariance_type}, BIC={best_bic:.1f}")

        results[name] = {
            'model': best_model,
            'labels': labels,
            'probs': probs,
            'bic_grid': bic_grid,
            'aic_grid': aic_grid,
            'best_k': best_model.n_components,
            'best_cov': best_model.covariance_type,
        }

    return results


def evaluate(results, X_pca, X_spca, meta, X_umap=None, X_features=None):
    summary_rows = []

    reduction_map = {'pca': X_pca, 'spca': X_spca}
    if X_umap is not None:
        reduction_map['umap'] = X_umap

    for name, X in reduction_map.items():
        r = results[name]
        labels = r['labels']
        probs = r['probs']
        model = r['model']

        # 1. Silhouette score
        sil = silhouette_score(X, labels)
        print(f"{name.upper()} | k={r['best_k']} cov={r['best_cov']} | "
              f"silhouette={sil:.3f} BIC={model.bic(X):.1f} AIC={model.aic(X):.1f}")

        summary_rows.append({
            'method': name,
            'k': r['best_k'],
            'covariance_type': r['best_cov'],
            'bic': model.bic(X),
            'aic': model.aic(X),
            'silhouette': sil,
        })

        # 2. Phase cross-tab (row-normalised: % of each cluster in each phase)
        ct = pd.crosstab(labels, meta['Phase'].values, normalize='index')
        plt.figure(figsize=(6, max(3, r['best_k'] * 0.5 + 1)))
        sns.heatmap(ct, annot=True, fmt='.2f', cmap='Blues', vmin=0, vmax=1)
        plt.xlabel('Phase')
        plt.ylabel('Cluster')
        plt.title(f'Phase distribution per cluster ({name.upper()})')
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / f'gmm_{name}_phase_crosstab.png', dpi=300)
        plt.close()

        # 3. Phase × Role cross-tab (single table, row-normalised across all phase+role combos)
        if 'Puzzler' in meta.columns:
            role_labels = meta['Puzzler'].map({1: 'Active', 0: 'Instructor'}).values
            ct_combined = pd.crosstab(
                labels,
                [meta['Phase'].values, role_labels],
                normalize='index'
            )
            ct_combined.columns = [f'{phase}\n{role}' for phase, role in ct_combined.columns]
            fig_w = max(8, len(ct_combined.columns) * 0.9)
            fig_h = max(3, r['best_k'] * 0.5 + 1)
            plt.figure(figsize=(fig_w, fig_h))
            sns.heatmap(ct_combined, annot=True, fmt='.2f', cmap='Blues', vmin=0, vmax=1)
            plt.xlabel('Phase / Role')
            plt.ylabel('Cluster')
            plt.title(f'Phase & Role distribution per cluster ({name.upper()})')
            plt.tight_layout()
            plt.savefig(FIGURES_DIR / f'gmm_{name}_phase_role_crosstab.png',
                        dpi=300, bbox_inches='tight')
            plt.close()

        # 4. Soft probability heatmap (rows sorted by dominant cluster)
        sort_idx = np.argsort(np.argmax(probs, axis=1))
        probs_sorted = probs[sort_idx]
        plt.figure(figsize=(max(4, r['best_k']), max(6, len(labels) * 0.04 + 2)))
        sns.heatmap(probs_sorted, cmap='viridis', vmin=0, vmax=1,
                    xticklabels=[f'C{i}' for i in range(r['best_k'])],
                    yticklabels=False)
        plt.xlabel('Cluster')
        plt.ylabel('Sample (sorted by dominant cluster)')
        plt.title(f'Soft assignment probabilities ({name.upper()})')
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / f'gmm_{name}_soft_probs.png', dpi=300)
        plt.close()

        # 5. Physiological cluster profiles (mean of original features per cluster)
        if X_features is not None:
            feat_df = X_features.copy() if isinstance(X_features, pd.DataFrame) else pd.DataFrame(X_features)
            feat_df = feat_df.reset_index(drop=True)
            feat_df['cluster'] = labels
            cluster_means = feat_df.groupby('cluster').mean()

            fig_w = max(14, len(cluster_means.columns) * 0.38)
            fig_h = max(3, r['best_k'] * 0.55 + 1.5)
            plt.figure(figsize=(fig_w, fig_h))
            sns.heatmap(cluster_means, cmap='RdBu_r', center=0,
                        linewidths=0.3, annot=False)
            plt.title(f'Mean physiological profile per cluster ({name.upper()})\n'
                      f'(individual-wise z-scored; red = above personal baseline)')
            plt.xlabel('Feature')
            plt.ylabel('Cluster')
            plt.xticks(rotation=45, ha='right', fontsize=7)
            plt.tight_layout()
            plt.savefig(FIGURES_DIR / f'gmm_{name}_cluster_profiles.png', dpi=300, bbox_inches='tight')
            plt.close()

            cluster_means.to_csv(RESULTS_DIR / f'gmm_{name}_cluster_profiles.csv')

        # 6. Save labelled output
        df_out = meta.copy().reset_index(drop=True)
        df_out['cluster'] = labels
        for i in range(probs.shape[1]):
            df_out[f'prob_cluster_{i}'] = probs[:, i]
        df_out.to_csv(PROCESSED_DIR / f'HR_data_gmm_{name}.csv', index=False)

    pd.DataFrame(summary_rows).to_csv(RESULTS_DIR / 'gmm_summary.csv', index=False)
    return summary_rows


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


def _draw_ellipse(ax, mean, cov, color, n_std=2.0):
    from matplotlib.patches import Ellipse
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    w, h = 2 * n_std * np.sqrt(np.maximum(vals, 0))
    ellipse = Ellipse(
        xy=mean, width=w, height=h, angle=theta,
        edgecolor=color, facecolor='none', linewidth=2, alpha=0.7
    )
    ax.add_patch(ellipse)


def plot(results, canvases, meta):
    """canvases: dict mapping method name to its 2D array for visualization."""
    colors = plt.cm.tab10.colors

    for name in results:
        r = results[name]
        labels = r['labels']
        k = r['best_k']
        canvas = canvases[name]
        x_label, y_label = ('UMAP1', 'UMAP2') if name == 'umap' else ('PC1', 'PC2')

        # 1. Scatter colored by hard cluster label
        fig, ax = plt.subplots(figsize=(8, 6))
        for c in range(k):
            mask = labels == c
            ax.scatter(canvas[mask, 0], canvas[mask, 1],
                       label=f'Cluster {c}', alpha=0.7, s=30, color=colors[c % 10])
        ax.set(xlabel=x_label, ylabel=y_label,
               title=f'GMM clusters — {name.upper()} (k={k}, cov={r["best_cov"]})')
        ax.legend()
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / f'gmm_{name}_clusters.png', dpi=300)
        plt.close()

        # 2. Scatter colored by Phase (sanity check)
        phases = meta['Phase'].values
        unique_phases = sorted(set(phases))
        fig, ax = plt.subplots(figsize=(8, 6))
        for i, phase in enumerate(unique_phases):
            mask = phases == phase
            ax.scatter(canvas[mask, 0], canvas[mask, 1],
                       label=phase, alpha=0.7, s=30, color=colors[i % 10])
        ax.set(xlabel=x_label, ylabel=y_label,
               title=f'Phase overlay — {name.upper()}')
        ax.legend(title='Phase')
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / f'gmm_{name}_phase.png', dpi=300)
        plt.close()

        # 3. Phase × Role scatter (color = phase, marker = puzzler role)
        if 'Puzzler' in meta.columns:
            roles = meta['Puzzler'].values
            role_markers = {1: ('o', 'Active'), 0: ('^', 'Instructor')}
            phase_colors = {p: colors[i % 10] for i, p in enumerate(sorted(set(phases)))}
            fig, ax = plt.subplots(figsize=(8, 6))
            for role_val, (marker, role_label) in role_markers.items():
                for phase in sorted(set(phases)):
                    mask = (phases == phase) & (roles == role_val)
                    if mask.sum() == 0:
                        continue
                    ax.scatter(canvas[mask, 0], canvas[mask, 1],
                               label=f'{phase} / {role_label}',
                               alpha=0.7, s=35, color=phase_colors[phase], marker=marker)
            ax.set(xlabel=x_label, ylabel=y_label,
                   title=f'Phase & Role overlay — {name.upper()}')
            ax.legend(title='Phase / Role', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
            plt.tight_layout()
            plt.savefig(FIGURES_DIR / f'gmm_{name}_phase_role.png', dpi=300, bbox_inches='tight')
            plt.close()

        # 4. Scatter colored by Individual
        ind_colors = plt.cm.tab20.colors
        individuals = meta['Individual'].values
        unique_individuals = sorted(set(individuals))
        fig, ax = plt.subplots(figsize=(8, 6))
        for i, ind in enumerate(unique_individuals):
            mask = individuals == ind
            ax.scatter(canvas[mask, 0], canvas[mask, 1],
                       label=ind, alpha=0.7, s=30, color=ind_colors[i % 20])
        ax.set(xlabel=x_label, ylabel=y_label,
               title=f'Individual overlay — {name.upper()}')
        ax.legend(title='Individual', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=7)
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / f'gmm_{name}_individual.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 5. GMM confidence ellipses (2D GMM fitted on canvas for visualization only)
        gmm_2d = GaussianMixture(
            n_components=k,
            covariance_type=r['best_cov'],
            random_state=42,
            n_init=5
        )
        gmm_2d.fit(canvas)
        labels_2d = gmm_2d.predict(canvas)

        fig, ax = plt.subplots(figsize=(8, 6))
        for c in range(k):
            mask = labels_2d == c
            ax.scatter(canvas[mask, 0], canvas[mask, 1],
                       alpha=0.5, s=25, color=colors[c % 10])
            cov = _get_component_cov(gmm_2d, c)
            _draw_ellipse(ax, gmm_2d.means_[c], cov,
                          color=colors[c % 10], n_std=2.0)
        ax.set(xlabel=x_label, ylabel=y_label,
               title=f'GMM confidence ellipses — {name.upper()} (k={k})')
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / f'gmm_{name}_ellipses.png', dpi=300)
        plt.close()

    print(f"Figures saved to {FIGURES_DIR}")


def main():
    df = load_data()
    X, meta, remaining = preprocess(df)
    X_pca, X_spca, X_umap, X_2d, X_umap_2d, n_components = reduce(X, remaining)
    results = fit_gmm(X_pca, X_spca, X_umap=X_umap)
    evaluate(results, X_pca, X_spca, meta, X_umap=X_umap, X_features=X)
    canvases = {'pca': X_2d, 'spca': X_2d, 'umap': X_umap_2d}
    plot(results, canvases, meta)
    print("GMM clustering complete.")


if __name__ == '__main__':
    main()
