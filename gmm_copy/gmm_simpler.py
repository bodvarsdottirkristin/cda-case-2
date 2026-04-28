import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

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
    X = pd.read_csv(PROCESSED_DIR / 'HR_data_2.csv') #raw
    X_pca2 = pd.read_csv(PROCESSED_DIR / 'HR_data_PCA2.csv') # processed
    return X, X_pca2

# covarianced reduced to 2, only full or diagonal
def fit_gmm(X, k_range=range(2, 9)):
    cov_types = ['full', 'diag']
    
    #initialize values
    results = {}
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
        raise ValueError(f"k_range was empty — no GMM fitted")
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
                title=f'{metric} — GMM')
        ax.legend()
        ax.grid(True)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f'gmm_PCA2_bic_aic.png', dpi=300)
    plt.close()

    print(f"  Best: k={best_model.n_components}, "
            f"cov={best_model.covariance_type}, BIC={best_bic:.1f}")

    results= {
        'model': best_model,
        'labels': labels,
        'probs': probs,
        'bic_grid': bic_grid,
        'aic_grid': aic_grid,
        'best_k': best_model.n_components,
        'best_cov': best_model.covariance_type,
    }

    return results


def evaluate(results, X, meta, X_features=None):
    summary_rows = []
    labels = results['labels']
    probs = results['probs']
    model = results['model']
    best_k = results['best_k']  # Define this once to avoid typos

    # 1. Silhouette score
    sil = silhouette_score(X, labels)
    print(f"k={best_k} cov={results['best_cov']} | "
            f"silhouette={sil:.3f} BIC={model.bic(X):.1f} AIC={model.aic(X):.1f}")

    summary_rows.append({
        'k': best_k,
        'covariance_type': results['best_cov'],
        'bic': model.bic(X),
        'aic': model.aic(X),
        'silhouette': sil,
    })

    # 2. Phase cross-tab
    ct = pd.crosstab(labels, meta['Phase'].values, normalize='index')
    plt.figure(figsize=(6, max(3, best_k * 0.5 + 1)))
    sns.heatmap(ct, annot=True, fmt='.2f', cmap='Blues', vmin=0, vmax=1)
    plt.xlabel('Phase')
    plt.ylabel('Cluster')
    plt.title('Phase distribution per cluster')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'gmm_PCA2_phase_crosstab.png', dpi=300)
    plt.close()

    # 3. Phase × Role cross-tab
    if 'Puzzler' in meta.columns:
        role_labels = meta['Puzzler'].map({1: 'Active', 0: 'Instructor'}).values
        ct_combined = pd.crosstab(
            labels,
            [meta['Phase'].values, role_labels],
            normalize='index'
        )
        ct_combined.columns = [f'{phase}\n{role}' for phase, role in ct_combined.columns]
        fig_w = max(8, len(ct_combined.columns) * 0.9)
        fig_h = max(3, best_k * 0.5 + 1) # FIXED: Changed 'r' to 'results' or 'best_k'
        plt.figure(figsize=(fig_w, fig_h))
        sns.heatmap(ct_combined, annot=True, fmt='.2f', cmap='Blues', vmin=0, vmax=1)
        plt.xlabel('Phase / Role')
        plt.ylabel('Cluster')
        plt.title('Phase & Role distribution per cluster')
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / 'gmm_PCA2_phase_role_crosstab.png',
                    dpi=300, bbox_inches='tight')
        plt.close()

    # 5. Physiological cluster profiles
    if X_features is not None:
        feat_df = X_features.copy() if isinstance(X_features, pd.DataFrame) else pd.DataFrame(X_features)
        feat_df = feat_df.reset_index(drop=True)
        feat_df['cluster'] = labels
        
        # FIXED: Added numeric_only=True to prevent string errors
        cluster_means = feat_df.groupby('cluster').mean(numeric_only=True)
        
        # Optional: Drop non-biosignal columns that might still be numeric
        to_drop = [c for c in ['Puzzler', 'Cohort', 'Unnamed: 0'] if c in cluster_means.columns]
        cluster_means = cluster_means.drop(columns=to_drop)

        fig_w = max(14, len(cluster_means.columns) * 0.38)
        fig_h = max(3, best_k * 0.55 + 1.5)
        plt.figure(figsize=(fig_w, fig_h))
        sns.heatmap(cluster_means, cmap='RdBu_r', center=0, linewidths=0.3)
        plt.title('Mean physiological profile per cluster')
        plt.xticks(rotation=45, ha='right', fontsize=7)
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / 'gmm_cluster_profiles.png', dpi=300, bbox_inches='tight')
        plt.close()

        cluster_means.to_csv(RESULTS_DIR / 'gmm_cluster_profiles.csv')

    # 6. Save labelled output
    df_out = meta.copy().reset_index(drop=True)
    df_out['cluster'] = labels
    for i in range(probs.shape[1]):
        df_out[f'prob_cluster_{i}'] = probs[:, i]
    df_out.to_csv(PROCESSED_DIR / 'HR_data_GMM.csv', index=False)

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
    """
    results: The dict returned by fit_gmm
    canvases: dict mapping 'pca' to the 2D array (first 2 PCs)
    meta: The metadata dataframe
    """
    colors = plt.cm.tab10.colors
    
    # Since fit_gmm now returns a single result dict, 
    # we treat it as the 'pca' result for this function.
    labels = results['labels']
    k = results['best_k']
    cov_type = results['best_cov']
    
    # We focus on the PCA canvas
    if 'pca' not in canvases:
        print("Warning: 'pca' key not found in canvases. Skipping plot.")
        return
        
    canvas = canvases['pca']
    x_label, y_label = 'PC1', 'PC2'

    # --- 1. GMM CLUSTERS (Hard Labels) ---
    fig, ax = plt.subplots(figsize=(8, 6))
    for c in range(k):
        mask = labels == c
        ax.scatter(canvas[mask, 0], canvas[mask, 1],
                   label=f'Cluster {c}', alpha=0.7, s=30, color=colors[c % 10])
    ax.set(xlabel=x_label, ylabel=y_label,
           title=f'GMM Biological Clusters (k={k}, cov={cov_type})')
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'gmm_pca_clusters.png', dpi=300)
    plt.close()

    # --- 2. PHASE OVERLAY (The "Experimental Reality") ---
    phases = meta['Phase'].values
    unique_phases = sorted(set(phases))
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, phase in enumerate(unique_phases):
        mask = phases == phase
        ax.scatter(canvas[mask, 0], canvas[mask, 1],
                   label=phase, alpha=0.6, s=30, color=colors[i % 10])
    ax.set(xlabel=x_label, ylabel=y_label,
           title='Experimental Phase Overlay')
    ax.legend(title='Phase')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'gmm_pca_phase_overlay.png', dpi=300)
    plt.close()

    # --- 3. CONFIDENCE ELLIPSES (The "Proof") ---
    # We fit a quick 2D GMM on the canvas just for the visualization ellipses
    gmm_2d = GaussianMixture(
        n_components=k,
        covariance_type=cov_type,
        random_state=42,
        n_init=2
    )
    gmm_2d.fit(canvas)

    fig, ax = plt.subplots(figsize=(8, 6))
    for c in range(k):
        mask = labels == c
        ax.scatter(canvas[mask, 0], canvas[mask, 1],
                   alpha=0.4, s=20, color=colors[c % 10])
        # _get_component_cov and _draw_ellipse are helper functions defined elsewhere
        cov = _get_component_cov(gmm_2d, c)
        _draw_ellipse(ax, gmm_2d.means_[c], cov, color=colors[c % 10], n_std=2.0)
        
    ax.set(xlabel=x_label, ylabel=y_label,
           title=f'GMM Confidence Ellipses (95% Boundary)')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'gmm_pca_ellipses.png', dpi=300)
    plt.close()

    print(f" Essential comparison figures saved to {FIGURES_DIR}")


def main():
    X_raw, X_pca2 = load_data()
    meta = X_raw[[c for c in META_COLS if c in X_raw.columns]].copy()

    # --- THE FIX STARTS HERE ---
    # Identify only the columns that look like 'PC1', 'PC2', etc.
    # This ignores 'Round', 'Individual', 'Phase', etc.
    pc_features = [c for c in X_pca2.columns if c.startswith('PC')]
    X_clustering = X_pca2[pc_features]
    # ---------------------------

    # Pass the stripped numeric-only data to fit_gmm
    results = fit_gmm(X_clustering, k_range=range(2, 6))
    
    # Use the same numeric-only data for evaluate
    evaluate(results, X_clustering, meta, X_features=X_raw)
    
    # For plotting, we just need the first two columns of our numeric set
    pca_2d = X_clustering.iloc[:, :2].values
    canvases = {'pca': pca_2d}

    plot(results, canvases, meta)
    
    print("GMM complete without metadata leaks.")

if __name__ == '__main__':
    main()