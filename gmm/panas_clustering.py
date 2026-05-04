import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Ellipse
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

FIGURES_DIR = Path(__file__).resolve().parent / 'figures'
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR = PROJECT_ROOT / 'data' / 'processed'
RESULTS_DIR = PROJECT_ROOT / 'results' / 'tables'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

PA_ITEMS = ['inspired', 'alert', 'attentive', 'active', 'determined']
# 'Frustrated' retains its capital-F as it appears in HR_data_2.csv
NA_ITEMS = ['Frustrated', 'upset', 'hostile', 'ashamed', 'nervous', 'afraid']
META_COLS = ['Round', 'Phase', 'Individual', 'Puzzler', 'Cohort']

ALL_ITEMS = PA_ITEMS + NA_ITEMS  # all 11 questionnaire items

EMOTIONAL_LABELS = {
    (True,  False): 'Engaged/Calm',
    (False, True):  'Tense/Stressed',
    (False, False): 'Drained/Disengaged',
    (True,  True):  'Alert/Anxious',
}


def load_data():
    return pd.read_csv(PROCESSED_DIR / 'HR_data_2.csv')


def compute_panas_scores(df):
    df = df.copy()
    df['pa_score'] = df[PA_ITEMS].mean(axis=1)
    df['na_score'] = df[NA_ITEMS].mean(axis=1)
    return df


def fit_gmm_panas(X, k_range=range(2, 7)):
    import warnings
    cov_types = ['full', 'tied', 'diag', 'spherical']
    if not list(k_range):
        raise ValueError("k_range must contain at least one value")
    best_bic = np.inf
    best_model = None
    records = []

    for cov in cov_types:
        for k in k_range:
            gmm = GaussianMixture(n_components=k, covariance_type=cov,
                                  random_state=42, n_init=10)
            gmm.fit(X)
            if not gmm.converged_:
                warnings.warn(f"GMM k={k}, cov={cov} did not converge; BIC may be unreliable")
            bic = gmm.bic(X)
            records.append({'k': k, 'covariance_type': cov, 'bic': bic})
            if bic < best_bic:
                best_bic = bic
                best_model = gmm

    return best_model, pd.DataFrame(records)


def fit_gmm_questionnaire(df, k_range=range(2, 7)):
    import warnings
    X_raw = df[ALL_ITEMS].dropna()
    valid_idx = X_raw.index
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)

    cov_types = ['full', 'tied', 'diag', 'spherical']
    best_bic = np.inf
    best_model = None
    records = []

    for cov in cov_types:
        for k in k_range:
            gmm = GaussianMixture(n_components=k, covariance_type=cov,
                                  random_state=42, n_init=10)
            gmm.fit(X)
            if not gmm.converged_:
                warnings.warn(f"GMM k={k}, cov={cov} did not converge")
            bic = gmm.bic(X)
            records.append({'k': k, 'covariance_type': cov, 'bic': bic})
            if bic < best_bic:
                best_bic = bic
                best_model = gmm

    return best_model, pd.DataFrame(records), X, valid_idx


def plot_questionnaire_clusters(df, model, bic_df, X_scaled, valid_idx):
    colors = plt.cm.tab10.colors
    k = model.n_components
    labels = model.predict(X_scaled)

    # BIC curve
    fig, ax = plt.subplots(figsize=(8, 5))
    for cov, grp in bic_df.groupby('covariance_type'):
        ax.plot(grp['k'], grp['bic'], marker='o', label=cov)
    ax.set(xlabel='k', ylabel='BIC (lower = better)',
           title='GMM on all 11 questionnaire items — BIC')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'questionnaire_bic.png', dpi=300)
    plt.close()

    # Cluster profile heatmap on the 11 items
    df_q = df.loc[valid_idx].copy().reset_index(drop=True)
    df_q['q_cluster'] = labels
    cluster_means = df_q.groupby('q_cluster')[ALL_ITEMS].mean()

    plt.figure(figsize=(max(8, len(ALL_ITEMS) * 0.7), max(3, k * 0.6 + 1.5)))
    sns.heatmap(cluster_means, cmap='RdBu_r',
                center=cluster_means.values.mean(),
                annot=True, fmt='.2f', linewidths=0.5)
    plt.title(f'GMM k={k}, cov={model.covariance_type} — '
              f'Mean questionnaire score per cluster')
    plt.xlabel('Questionnaire Item')
    plt.ylabel('Cluster')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'questionnaire_cluster_profiles.png',
                dpi=300, bbox_inches='tight')
    plt.close()

    # PCA 2D scatter for visualization
    pca_2d = PCA(n_components=2, random_state=42)
    X_2d = pca_2d.fit_transform(X_scaled)

    fig, ax = plt.subplots(figsize=(8, 6))
    for c in range(k):
        mask = labels == c
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                   label=f'Cluster {c}', alpha=0.7, s=35, color=colors[c % 10])
    ax.set(xlabel='PC1 (of questionnaire items)', ylabel='PC2',
           title=f'GMM clusters — all questionnaire items (k={k})')
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'questionnaire_clusters_pca2d.png', dpi=300)
    plt.close()

    print(f"Best GMM (11-item): k={k}, cov={model.covariance_type}")
    print(f"Saved: questionnaire_bic.png, questionnaire_cluster_profiles.png, "
          f"questionnaire_clusters_pca2d.png")
    return df_q


def label_clusters(means):
    # means: array (k, 2), col 0 = pa_score, col 1 = na_score
    # Uses >= for median split; ties fall into the high side.
    pa_med = np.median(means[:, 0])
    na_med = np.median(means[:, 1])
    base_labels = [EMOTIONAL_LABELS[(pa >= pa_med, na >= na_med)] for pa, na in means]
    # Disambiguate duplicates when k > 4 (multiple clusters in same quadrant)
    counts: dict = {}
    result = []
    for label in base_labels:
        counts[label] = counts.get(label, 0) + 1
        result.append(label if counts[label] == 1 else f"{label} ({counts[label]})")
    return result


def _draw_ellipse(ax, mean, cov, color, n_std=2.0):
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    w, h = 2 * n_std * np.sqrt(np.maximum(vals, 0))
    ellipse = Ellipse(xy=mean, width=w, height=h, angle=theta,
                      edgecolor=color, facecolor='none', linewidth=2, alpha=0.7)
    ax.add_patch(ellipse)


def _get_cov(model, component_idx):
    if model.covariance_type == 'full':
        return model.covariances_[component_idx]
    elif model.covariance_type == 'tied':
        return model.covariances_
    elif model.covariance_type == 'diag':
        return np.diag(model.covariances_[component_idx])
    else:  # spherical
        return np.eye(2) * model.covariances_[component_idx]


def plot_panas_space(df_labeled, model, cluster_names, bic_df):
    colors = plt.cm.tab10.colors
    k = model.n_components
    X = df_labeled[['pa_score', 'na_score']].values

    # 1. PA/NA scatter colored by phase
    phase_colors = {'phase1': colors[0], 'phase2': colors[1], 'phase3': colors[2]}
    fig, ax = plt.subplots(figsize=(8, 6))
    for phase, grp in df_labeled.groupby('Phase'):
        ax.scatter(grp['pa_score'], grp['na_score'],
                   label=phase, alpha=0.6, s=30,
                   color=phase_colors.get(phase, 'gray'))
    ax.set(xlabel='Positive Affect (PA)', ylabel='Negative Affect (NA)',
           title='PANAS space — colored by Phase')
    ax.legend(title='Phase')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'panas_pa_na_phase.png', dpi=300)
    plt.close()

    # 2. BIC curve
    fig, ax = plt.subplots(figsize=(8, 5))
    for cov, grp in bic_df.groupby('covariance_type'):
        ax.plot(grp['k'], grp['bic'], marker='o', label=cov)
    ax.set(xlabel='k', ylabel='BIC (lower is better)',
           title='GMM model selection — PANAS space')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'panas_bic.png', dpi=300)
    plt.close()

    # 3. GMM ellipses colored by emotional cluster
    labels = df_labeled['emotional_cluster_id'].values
    fig, ax = plt.subplots(figsize=(8, 6))
    for c in range(k):
        mask = labels == c
        name = cluster_names[c]
        ax.scatter(X[mask, 0], X[mask, 1],
                   label=name, alpha=0.7, s=35, color=colors[c % 10])
        _draw_ellipse(ax, model.means_[c], _get_cov(model, c),
                      color=colors[c % 10])
    ax.set(xlabel='Positive Affect (PA)', ylabel='Negative Affect (NA)',
           title=f'Emotional clusters — GMM (k={k}, cov={model.covariance_type})')
    ax.legend(title='Emotional State')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'panas_gmm_ellipses.png', dpi=300)
    plt.close()
    print(f"Saved: panas_pa_na_phase.png, panas_bic.png, panas_gmm_ellipses.png")


def plot_cross_tabs(df_labeled):
    # Phase cross-tab
    ct_phase = pd.crosstab(df_labeled['emotional_cluster'],
                           df_labeled['Phase'], normalize='index')
    fig, ax = plt.subplots(figsize=(7, max(3, len(ct_phase) * 0.6 + 1)))
    sns.heatmap(ct_phase, annot=True, fmt='.2f', cmap='Blues',
                vmin=0, vmax=1, ax=ax)
    ax.set(xlabel='Phase', ylabel='Emotional State',
           title='Phase distribution per emotional cluster')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'panas_phase_crosstab.png', dpi=300)
    plt.close()

    # Role cross-tab
    if 'Puzzler' in df_labeled.columns:
        role_labels = df_labeled['Puzzler'].map({1: 'Active', 0: 'Instructor'})
        ct_role = pd.crosstab(df_labeled['emotional_cluster'],
                              role_labels, normalize='index')
        fig, ax = plt.subplots(figsize=(6, max(3, len(ct_role) * 0.6 + 1)))
        sns.heatmap(ct_role, annot=True, fmt='.2f', cmap='Blues',
                    vmin=0, vmax=1, ax=ax)
        ax.set(xlabel='Role', ylabel='Emotional State',
               title='Role distribution per emotional cluster')
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / 'panas_role_crosstab.png', dpi=300)
        plt.close()
        print("Saved: panas_phase_crosstab.png, panas_role_crosstab.png")
    else:
        print("Saved: panas_phase_crosstab.png")


def plot_biosignal_profiles(df_labeled):
    preprocessed = pd.read_csv(PROCESSED_DIR / 'HR_data_gmm_preprocessed.csv')

    merge_cols = ['Round', 'Phase', 'Individual']
    merged = preprocessed.merge(
        df_labeled[merge_cols + ['emotional_cluster']],
        on=merge_cols, how='inner'
    )

    if len(merged) < len(df_labeled):
        import warnings
        warnings.warn(f"Inner join dropped {len(df_labeled) - len(merged)} rows — "
                      "preprocessed file may have fewer rows than labeled data")
    non_biosig = set(META_COLS + ['Unnamed: 0', 'original_ID', 'raw_data_path',
                                   'Team_ID', 'emotional_cluster'] + PA_ITEMS + NA_ITEMS)
    biosig_cols = [c for c in merged.columns if c not in non_biosig]

    cluster_means = merged.groupby('emotional_cluster')[biosig_cols].mean()
    cluster_means.to_csv(RESULTS_DIR / 'panas_cluster_profiles.csv')

    fig_w = max(14, len(biosig_cols) * 0.38)
    fig_h = max(3, len(cluster_means) * 0.6 + 1.5)
    plt.figure(figsize=(fig_w, fig_h))
    sns.heatmap(cluster_means, cmap='RdBu_r', center=0,
                linewidths=0.3, annot=False)
    plt.title('Mean biosignal profile per emotional cluster\n'
              '(individual-wise z-scored; red = above personal baseline)')
    plt.xlabel('Feature')
    plt.ylabel('Emotional State')
    plt.xticks(rotation=45, ha='right', fontsize=7)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'panas_cluster_profiles.png', dpi=300,
                bbox_inches='tight')
    plt.close()
    print("Saved: panas_cluster_profiles.png and panas_cluster_profiles.csv")


def main():
    df = load_data()
    df = compute_panas_scores(df)

    X = df[['pa_score', 'na_score']].values
    model, bic_df = fit_gmm_panas(X)
    cluster_names = label_clusters(model.means_)

    labels = model.predict(X)
    probs = model.predict_proba(X)
    k = model.n_components

    df['emotional_cluster_id'] = labels
    df['emotional_cluster'] = [cluster_names[idx] for idx in labels]
    for i in range(k):
        df[f'prob_emotional_{i}'] = probs[:, i]

    print(f"Best GMM: k={k}, covariance_type={model.covariance_type}")
    print("Cluster names:", cluster_names)
    print("\nPhase × Emotional Cluster:")
    print(pd.crosstab(df['Phase'], df['emotional_cluster']))

    plot_panas_space(df, model, cluster_names, bic_df)
    plot_cross_tabs(df)
    plot_biosignal_profiles(df)

    out_cols = META_COLS + ['pa_score', 'na_score',
                             'emotional_cluster_id', 'emotional_cluster'] + \
               [f'prob_emotional_{i}' for i in range(k)]
    df[out_cols].to_csv(PROCESSED_DIR / 'HR_data_panas.csv', index=False)
    # --- All-item questionnaire clustering ---
    q_model, q_bic_df, X_q_scaled, valid_idx = fit_gmm_questionnaire(df)
    df_q = plot_questionnaire_clusters(df, q_model, q_bic_df, X_q_scaled, valid_idx)

    # Save questionnaire cluster assignments for bridge analysis
    out_q_cols = META_COLS + ['q_cluster']
    df_q[out_q_cols].to_csv(
        PROCESSED_DIR / 'HR_data_questionnaire_clusters.csv', index=False
    )
    print("Questionnaire clusters saved to HR_data_questionnaire_clusters.csv")

    print(f"\nAll outputs saved. Figures: {FIGURES_DIR}  Tables: {RESULTS_DIR}")


if __name__ == '__main__':
    main()
