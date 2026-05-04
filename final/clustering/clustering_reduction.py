import os
import sys
import warnings
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Ellipse
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from scipy.stats import chi2_contingency, mannwhitneyu

warnings.filterwarnings('ignore')
sns.set_theme(style="whitegrid")

# ==========================================
# 1. STATS & PLOTTING FUNCTIONS
# ==========================================

def cohens_d(x, y):
    nx, ny = len(x), len(y)
    dof = nx + ny - 2
    if dof <= 0: return 0
    pooled_std = np.sqrt(((nx-1)*np.var(x, ddof=1) + (ny-1)*np.var(y, ddof=1)) / dof)
    return (np.mean(x) - np.mean(y)) / pooled_std if pooled_std > 0 else 0

def draw_empirical_ellipses(ax, X, labels, palette, alpha=0.2):
    unique_labels = np.unique(labels)
    colors = sns.color_palette(palette, n_colors=len(unique_labels))
    for idx, c in enumerate(unique_labels):
        X_c = X[labels == c]
        if len(X_c) < 2: continue
        mean = np.mean(X_c, axis=0)
        cov = np.cov(X_c, rowvar=False)
        v, w = np.linalg.eigh(cov)
        u = w[0] / np.linalg.norm(w[0])
        angle = np.arctan2(u[1], u[0]) * 180 / np.pi
        v = 2. * np.sqrt(2.) * np.sqrt(v)
        ell = Ellipse(xy=mean, width=v[0], height=v[1], angle=180+angle, color=colors[idx])
        ell.set_clip_box(ax.bbox)
        ell.set_alpha(alpha)
        ax.add_artist(ell)

def save_combined_cluster_plots(df, coords, models_dict, phase_col, save_path, xlabel, ylabel):
    n_models = len(models_dict)
    fig, axes = plt.subplots(n_models, 2, figsize=(14, 5 * n_models))
    dim1, dim2 = coords[:, 0], coords[:, 1]
    phases = df[phase_col].values

    for i, (model_name, cluster_col) in enumerate(models_dict.items()):
        labels = df[cluster_col].values
        ax_cluster = axes[i, 0] if n_models > 1 else axes[0]
        sns.scatterplot(x=dim1, y=dim2, hue=labels, palette='viridis', ax=ax_cluster, s=50, alpha=0.7)
        draw_empirical_ellipses(ax_cluster, coords, labels, palette='viridis')
        ax_cluster.set_title(f"{model_name}: Predicted Clusters")
        ax_cluster.set_xlabel(xlabel)
        ax_cluster.set_ylabel(ylabel)

        ax_gt = axes[i, 1] if n_models > 1 else axes[1]
        sns.scatterplot(x=dim1, y=dim2, hue=phases, palette='Set1', ax=ax_gt, s=50, alpha=0.8)
        ax_gt.set_title("Ground Truth Phases")
        ax_gt.set_xlabel(xlabel)
        ax_gt.set_ylabel(ylabel)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def evaluate_optimal_k_and_save(X, title, save_path, max_k=5):
    k_range = list(range(2, max_k + 1))
    bics, sil_kmeans, sil_kmedoids = [], [], []

    for k in k_range:
        bics.append(GaussianMixture(n_components=k, random_state=42).fit(X).bic(X))
        sil_kmeans.append(silhouette_score(X, KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(X)))
        sil_kmedoids.append(silhouette_score(X, KMedoids(n_clusters=k, random_state=42, method='pam').fit_predict(X)))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(k_range, bics, marker='o', color='purple')
    axes[0].set_title(f'GMM: BIC Score ({title})')
    axes[0].set_xlabel('K')
    axes[0].set_xticks(k_range)

    axes[1].plot(k_range, sil_kmeans, marker='o', label='K-Means')
    axes[1].plot(k_range, sil_kmedoids, marker='s', label='K-Medoids')
    axes[1].set_title(f'Silhouette Score ({title})')
    axes[1].set_xlabel('K')
    axes[1].set_xticks(k_range)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    return {
        'K-Means': k_range[np.argmax(sil_kmeans)], 
        'K-Medoids': k_range[np.argmax(sil_kmedoids)],
        'GMM': k_range[np.argmin(bics)]
    }

# ==========================================
# 2. MAIN PIPELINE
# ==========================================

if __name__ == "__main__":
    print("Starting pipeline...")
    
    # Adatta il percorso al tuo ambiente. 
    # Se lo script è in una cartella es. `scripts/`, usa .parents[1]. 
    # Se lo lanci dalla root del progetto, usa .resolve() direttamente.
    PROJECT_ROOT = Path(os.getcwd()).resolve().parents[0] 
    sys.path.append(str(PROJECT_ROOT)) 
    
    BASE_OUTPUT_DIR = PROJECT_ROOT / 'cda-case-2' / 'final' / 'clustering' / 'results' 
    BASE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Load Original Data for Biological Feature Extraction
    orig_data_path = PROJECT_ROOT / 'cda-case-2' / 'data' / 'processed' / 'HR_data_2.csv'
    
    if not orig_data_path.exists():
        print(f"ERROR: Original dataset not found at {orig_data_path}")
        sys.exit(1)
        
    df_orig = pd.read_csv(orig_data_path)
    if 'phase' in df_orig.columns: df_orig.rename(columns={'phase': 'Phase'}, inplace=True)

    meta_cols = ['original ID', 'raw_data Path', 'Team ID', 'Individual', 'Phase', 'Cohort', 'Round', 'Role', 'Puzzler']
    quest_cols = ['Frustrated', 'upset', 'hostile', 'alert', 'ashamed', 'inspired', 'nervous', 'attentive', 'afraid', 'active', 'determined']
    
    # Isolate original biosignals (for Mann-Whitney interpretation)
    orig_bio_features = [c for c in df_orig.select_dtypes(include='number').columns if c not in meta_cols + quest_cols]

    # 2. Iterate over Reduced Datasets
    data_paths = [
        PROJECT_ROOT / 'cda-case-2' / 'data' / 'processed' / 'final' / 'HR_data_pca.csv', 
        PROJECT_ROOT / 'cda-case-2' / 'data' / 'processed' / 'final' / 'HR_data_spca.csv', 
        PROJECT_ROOT / 'cda-case-2' / 'data' / 'processed' / 'final' / 'HR_data_umap.csv'
    ]

    for data_file in data_paths:
        if not data_file.exists(): 
            print(f"Skipping {data_file.name} (File not found)")
            continue
            
        print(f"\nProcessing: {data_file.name}...")
        
        # Create output directory for this specific method
        method_out_dir = BASE_OUTPUT_DIR / data_file.stem
        method_out_dir.mkdir(parents=True, exist_ok=True)
        
        df_reduced = pd.read_csv(data_file)
        if 'phase' in df_reduced.columns: df_reduced.rename(columns={'phase': 'Phase'}, inplace=True)
        
        # Ensure row alignment between reduced and original datasets
        assert len(df_reduced) == len(df_orig), f"Row mismatch in {data_file.name} vs original data."

        # Extract reduced latent dimensions (NO STANDARD SCALER)
        reduced_feature_cols = [c for c in df_reduced.select_dtypes(include=[np.number]).columns if c not in meta_cols + quest_cols]
        X_reduced = df_reduced[reduced_feature_cols].fillna(df_reduced[reduced_feature_cols].median()).values
        
        # Determine Optimal K and save plot
        optimal_k_plot_path = method_out_dir / "1_optimal_k.png"
        best_ks = evaluate_optimal_k_and_save(X_reduced, title=data_file.stem, save_path=optimal_k_plot_path, max_k=5)

        models = {
            'K-Means': KMeans(n_clusters=best_ks['K-Means'], random_state=42, n_init=10),
            'K-Medoids': KMedoids(n_clusters=best_ks['K-Medoids'], random_state=42, method='pam'),
            'GMM': GaussianMixture(n_components=best_ks['GMM'], random_state=42)
        }

        models_cols = {}
        
        # Open Markdown file to save text results
        md_file_path = method_out_dir / "2_Clustering_Report.md"
        with open(md_file_path, "w", encoding="utf-8") as md_file:
            md_file.write(f"# Analysis Report for: `{data_file.name}`\n\n")

            for model_name, model in models.items():
                print(f"  - Running {model_name}...")
                cluster_col = f"{model_name}_Cluster"
                
                # Fit & Predict on Reduced Space
                labels = model.fit_predict(X_reduced)
                df_reduced[cluster_col] = labels
                
                # Map labels back to Original Data
                df_orig[cluster_col] = labels
                models_cols[model_name] = cluster_col

                md_file.write(f"## Model: {model_name} (Optimal K={best_ks[model_name]})\n\n")
                md_file.write("### Contingency Tables & Alignment Metrics\n")
                
                # Contingency Tables & Alignment Metrics
                for meta_col in ['Phase', 'Cohort', 'Round', 'Role']:
                    if meta_col in df_reduced.columns:
                        ct = pd.crosstab(df_reduced[cluster_col], df_reduced[meta_col])
                        ari = adjusted_rand_score(df_reduced[meta_col].astype(str), df_reduced[cluster_col])
                        nmi = normalized_mutual_info_score(df_reduced[meta_col].astype(str), df_reduced[cluster_col])
                        chi2, p, _, _ = chi2_contingency(ct)
                        cramer_v = np.sqrt(chi2 / (ct.sum().sum() * (min(ct.shape) - 1))) if min(ct.shape) > 1 else 0

                        md_file.write(f"**Target: {meta_col}**  \n")
                        md_file.write(f"ARI: `{ari:.4f}` | NMI: `{nmi:.4f}` | p-value: `{p:.4e}` | Cramer's V: `{cramer_v:.4f}`\n\n")
                        md_file.write(ct.to_markdown() + "\n\n")

                # Feature Discriminability on ORIGINAL Data
                md_file.write("### Top 10 Discriminating Original Features (Mann-Whitney U)\n")
                mw_results = []
                for c in range(best_ks[model_name]):
                    c_data = df_orig[df_orig[cluster_col] == c]
                    rest_data = df_orig[df_orig[cluster_col] != c]
                    
                    for f in orig_bio_features:
                        if c_data[f].isna().all() or rest_data[f].isna().all(): continue
                        c_vals, r_vals = c_data[f].dropna(), rest_data[f].dropna()
                        if len(c_vals) == 0 or len(r_vals) == 0: continue

                        stat, p = mannwhitneyu(c_vals, r_vals, alternative='two-sided')
                        d = cohens_d(c_vals, r_vals)
                        mw_results.append({'Cluster': c, 'Feature': f, 'p-val': p, 'Cohens_d': abs(d)})

                if mw_results:
                    mw_df = pd.DataFrame(mw_results).sort_values(by=['Cluster', 'Cohens_d'], ascending=[True, False])
                    # Select Top 10 features per cluster
                    md_file.write(mw_df.groupby('Cluster').head(10).to_markdown(index=False) + "\n\n")
                
                md_file.write("---\n\n")

        # Visualizations (using the first 2 dimensions of the reduced space for plotting)
        print("  - Generating combined scatter plots...")
        dim1_name = reduced_feature_cols[0] if len(reduced_feature_cols) > 0 else "Dim 1"
        dim2_name = reduced_feature_cols[1] if len(reduced_feature_cols) > 1 else "Dim 2"
        visualizations_plot_path = method_out_dir / "3_visualizations_combined.png"
        save_combined_cluster_plots(df_reduced, X_reduced[:, :2], models_cols, 'Phase', visualizations_plot_path, dim1_name, dim2_name)
        
    print("\nPipeline execution completed successfully! All results are saved in the output directory.")