import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Ellipse
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from scipy.stats import chi2_contingency, mannwhitneyu
import warnings

warnings.filterwarnings('ignore')
sns.set_theme(style="whitegrid")

def cohens_d(x, y):
    nx, ny = len(x), len(y)
    dof = nx + ny - 2
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
        ell = Ellipse(mean, v[0], v[1], angle=180+angle, color=colors[idx])
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
        ax_gt.set_title(f"Ground Truth Phases")
        ax_gt.set_xlabel(xlabel)
        ax_gt.set_ylabel(ylabel)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def evaluate_optimal_k_and_save(X_scaled, title, save_path, max_k=5):
    k_range = list(range(2, max_k + 1))
    bics, sil_kmeans, sil_kmedoids = [], [], []

    for k in k_range:
        bics.append(GaussianMixture(n_components=k, random_state=42).fit(X_scaled).bic(X_scaled))
        sil_kmeans.append(silhouette_score(X_scaled, KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(X_scaled)))
        sil_kmedoids.append(silhouette_score(X_scaled, KMedoids(n_clusters=k, random_state=42, method='pam').fit_predict(X_scaled)))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Optimal K Analysis - {title}", fontsize=14, fontweight='bold')

    axes[0].plot(k_range, bics, marker='o', color='purple')
    axes[0].set_title('GMM: BIC Score (Lower is better)')
    axes[0].set_xticks(k_range)

    axes[1].plot(k_range, sil_kmeans, marker='o', label='K-Means')
    axes[1].plot(k_range, sil_kmedoids, marker='s', label='K-Medoids')
    axes[1].set_title('Silhouette Score (Higher is better)')
    axes[1].set_xticks(k_range)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    return {
        'GMM': k_range[np.argmin(bics)], 
        'K-Means': k_range[np.argmax(sil_kmeans)], 
        'K-Medoids': k_range[np.argmax(sil_kmedoids)]
    }

def run_analysis_pipeline(df_reduced, df_orig, file_name, out_dir):
    if 'phase' in df_reduced.columns:
        df_reduced.rename(columns={'phase': 'Phase'}, inplace=True)
    if 'phase' in df_orig.columns:
        df_orig.rename(columns={'phase': 'Phase'}, inplace=True)
        
    metadata_cols = ['original ID', 'raw_data Path', 'Team ID', 'Individual', 'Phase', 'Cohort', 'Round', 'Role', 'Puzzler']
    QUEST_COLS = ['Frustrated', 'upset', 'hostile', 'alert', 'ashamed', 'inspired', 'nervous', 'attentive', 'afraid', 'active', 'determined']
    
    bio_out_dir = out_dir / 'biosignals'
    quest_out_dir = out_dir / 'questionnaires'
    bio_out_dir.mkdir(parents=True, exist_ok=True)
    quest_out_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------
    # PART A: BIOSIGNALS
    # ---------------------------------------------------------
    reduced_num_cols = df_reduced.select_dtypes(include=[np.number]).columns
    reduced_features = [c for c in reduced_num_cols if c not in metadata_cols and c not in QUEST_COLS]
    
    orig_num_cols = df_orig.select_dtypes(include=[np.number]).columns
    orig_bio_features = [c for c in orig_num_cols if c not in metadata_cols and c not in QUEST_COLS]
    
    if len(reduced_features) >= 2:
        dim1_col, dim2_col = reduced_features[0], reduced_features[1]
        X_bio = df_reduced[[dim1_col, dim2_col]].fillna(df_reduced[[dim1_col, dim2_col]].median())
        X_bio_scaled = StandardScaler().fit_transform(X_bio)
        
        best_ks_bio = evaluate_optimal_k_and_save(X_bio_scaled, "Biosignals", bio_out_dir / "1_optimal_k.png", max_k=5)
        
        models_bio = {
            'K-Means': KMeans(n_clusters=best_ks_bio['K-Means'], random_state=42, n_init=10),
            'K-Medoids': KMedoids(n_clusters=best_ks_bio['K-Medoids'], random_state=42, method='pam'),
            'GMM': GaussianMixture(n_components=best_ks_bio['GMM'], random_state=42, covariance_type='diag')
        }
        
        models_cols_bio = {}
        
        with open(bio_out_dir / "2_Biosignals_Report.md", "w", encoding="utf-8") as md_file:
            md_file.write(f"# Biosignals Analysis Report: {file_name}\n\n")
            
            for model_name, model in models_bio.items():
                cluster_col = f"{model_name}_Bio_Cluster"
                
                # Fit on reduced space
                labels = model.fit_predict(X_bio_scaled)
                df_reduced[cluster_col] = labels
                df_orig[cluster_col] = labels
                models_cols_bio[model_name] = cluster_col
                
                md_file.write(f"## Model: {model_name} (K={best_ks_bio[model_name]})\n\n")
                md_file.write("### Contingency Tables & Metrics\n")
                
                for meta_col in ['Phase', 'Cohort', 'Round', 'Role']:
                    if meta_col in df_reduced.columns:
                        ct = pd.crosstab(df_reduced[cluster_col], df_reduced[meta_col])
                        ari = adjusted_rand_score(df_reduced[meta_col].astype(str), df_reduced[cluster_col])
                        nmi = normalized_mutual_info_score(df_reduced[meta_col].astype(str), df_reduced[cluster_col])
                        chi2, p, _, _ = chi2_contingency(ct)
                        cramer_v = np.sqrt(chi2 / (ct.sum().sum() * (min(ct.shape) - 1))) if min(ct.shape) > 1 else 0
                        
                        md_file.write(f"**Target: {meta_col}**  \n")
                        md_file.write(f"- ARI: `{ari:.4f}` | NMI: `{nmi:.4f}` | p-value: `{p:.4e}` | Cramer's V: `{cramer_v:.4f}`\n\n")
                        md_file.write(ct.to_markdown() + "\n\n")
                
                md_file.write("### Top 5 Discriminating Original Features (Mann-Whitney U)\n")
                mw_results = []
                for c in range(best_ks_bio[model_name]):
                    # Evaluate on Original Features mapping from reduced labels
                    c_data = df_orig[df_orig[cluster_col] == c]
                    rest_data = df_orig[df_orig[cluster_col] != c]
                    
                    for f in orig_bio_features:
                        if c_data[f].isna().all() or rest_data[f].isna().all(): continue
                        
                        c_vals = c_data[f].dropna()
                        r_vals = rest_data[f].dropna()
                        
                        if len(c_vals) == 0 or len(r_vals) == 0: continue
                            
                        stat, p = mannwhitneyu(c_vals, r_vals, alternative='two-sided')
                        d = cohens_d(c_vals, r_vals)
                        mw_results.append({'Cluster': c, 'Feature': f, 'p-val': p, 'Cohens_d': abs(d)})
                
                if mw_results:
                    mw_df = pd.DataFrame(mw_results).sort_values(by=['Cluster', 'Cohens_d'], ascending=[True, False])
                    top_feats = mw_df.groupby('Cluster').head(5)
                    md_file.write(top_feats.to_markdown(index=False) + "\n\n")
                    
        save_combined_cluster_plots(df_reduced, X_bio_scaled, models_cols_bio, 'Phase', bio_out_dir / "3_visualizations_combined.png", dim1_col, dim2_col)

    # ---------------------------------------------------------
    # PART B: QUESTIONNAIRES
    # ---------------------------------------------------------
    available_quest = [c for c in QUEST_COLS if c in df_orig.columns]
    
    if available_quest:
        X_quest = df_orig[available_quest].fillna(df_orig[available_quest].median())
        X_quest_scaled = StandardScaler().fit_transform(X_quest)
        pca_coords = PCA(n_components=2, random_state=42).fit_transform(X_quest_scaled)
        
        best_ks_quest = evaluate_optimal_k_and_save(X_quest_scaled, "Questionnaires", quest_out_dir / "1_optimal_k.png", max_k=5)
        
        models_quest = {
            'K-Means': KMeans(n_clusters=best_ks_quest['K-Means'], random_state=42, n_init=10),
            'K-Medoids': KMedoids(n_clusters=best_ks_quest['K-Medoids'], random_state=42, method='pam'),
            'GMM': GaussianMixture(n_components=best_ks_quest['GMM'], random_state=42)
        }
        
        models_cols_quest = {}
        
        with open(quest_out_dir / "2_Questionnaires_Report.md", "w", encoding="utf-8") as md_file:
            md_file.write(f"# Questionnaires Analysis Report: {file_name}\n\n")
            
            fig_heatmap, axes_heat = plt.subplots(len(models_quest), 1, figsize=(10, 4 * len(models_quest)))
            
            for i, (model_name, model) in enumerate(models_quest.items()):
                cluster_col = f"{model_name}_Quest_Cluster"
                labels = model.fit_predict(X_quest_scaled)
                df_orig[cluster_col] = labels
                df_reduced[cluster_col] = labels
                models_cols_quest[model_name] = cluster_col
                
                md_file.write(f"## Model: {model_name} (K={best_ks_quest[model_name]})\n\n")
                md_file.write("### Contingency Tables & Metrics\n")
                
                for meta_col in ['Phase', 'Cohort', 'Round', 'Role']:
                    if meta_col in df_orig.columns:
                        ct = pd.crosstab(df_orig[cluster_col], df_orig[meta_col])
                        ari = adjusted_rand_score(df_orig[meta_col].astype(str), df_orig[cluster_col])
                        nmi = normalized_mutual_info_score(df_orig[meta_col].astype(str), df_orig[cluster_col])
                        chi2, p, _, _ = chi2_contingency(ct)
                        cramer_v = np.sqrt(chi2 / (ct.sum().sum() * (min(ct.shape) - 1))) if min(ct.shape) > 1 else 0
                        
                        md_file.write(f"**Target: {meta_col}**  \n")
                        md_file.write(f"- ARI: `{ari:.4f}` | NMI: `{nmi:.4f}` | p-value: `{p:.4e}` | Cramer's V: `{cramer_v:.4f}`\n\n")
                        md_file.write(ct.to_markdown() + "\n\n")
                
                md_file.write("### Mean Cluster Profiles (Z-Scores)\n")
                cluster_profiles = df_orig.groupby(cluster_col)[available_quest].mean()
                md_file.write(cluster_profiles.to_markdown() + "\n\n")
                
                ax_h = axes_heat[i] if len(models_quest) > 1 else axes_heat
                sns.heatmap(cluster_profiles, annot=True, cmap='coolwarm', fmt=".2f", ax=ax_h, cbar=False)
                ax_h.set_title(f"{model_name} - Mean Profile")

            plt.tight_layout()
            plt.savefig(quest_out_dir / "4_profiles_heatmap.png", dpi=300)
            plt.close()
            
        save_combined_cluster_plots(df_orig, pca_coords, models_cols_quest, 'Phase', quest_out_dir / "3_visualizations_combined.png", "PCA 1", "PCA 2")


if __name__ == "__main__":
    PROJECT_ROOT = Path(os.getcwd()).resolve()
    BASE_OUTPUT_DIR = PROJECT_ROOT / 'final' / 'clustering' / 'results' 
    
    orig_data_path = PROJECT_ROOT / 'data' / 'processed' / 'HR_data_2.csv'
    
    if not orig_data_path.exists():
        sys.exit(1)
        
    df_original = pd.read_csv(orig_data_path)

    data_paths = [
        PROJECT_ROOT / 'data' / 'processed' / 'final' / 'HR_data_pca.csv', 
        PROJECT_ROOT / 'data' / 'processed' / 'final' / 'HR_data_spca.csv', 
        PROJECT_ROOT / 'data' / 'processed' / 'final' / 'HR_data_umap.csv'
    ]

    for data_file in data_paths:
        if data_file.exists():
            dataset_out_dir = BASE_OUTPUT_DIR / data_file.stem
            df_reduced = pd.read_csv(data_file)
            
            if len(df_reduced) == len(df_original):
                run_analysis_pipeline(df_reduced, df_original.copy(), data_file.name, dataset_out_dir)