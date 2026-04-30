import os
import warnings
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score

warnings.filterwarnings('ignore')

def evaluate_best_combinations(data_paths, target_col='Phase', max_k=5):
    results = []
    meta_cols = ['original ID', 'raw_data Path', 'Team ID', 'Individual', 'Phase', 'Cohort', 'Round', 'Role', 'Puzzler']

    for path in data_paths:
        if not path.exists():
            continue
            
        df = pd.read_csv(path)
        if 'phase' in df.columns:
            df.rename(columns={'phase': 'Phase'}, inplace=True)

        features = [c for c in df.select_dtypes(include=[np.number]).columns if c not in meta_cols]
        if len(features) < 2:
            continue
            
        X = StandardScaler().fit_transform(df[features].fillna(df[features].median()))

        for model_name in ['K-Means', 'K-Medoids', 'GMM']:
            best_k = 2
            best_val_score = -np.inf if model_name != 'GMM' else np.inf

            # 1. Find optimal K
            for k in range(2, max_k + 1):
                if model_name == 'K-Means':
                    labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(X)
                    score = silhouette_score(X, labels)
                    if score > best_val_score: best_k, best_val_score = k, score
                elif model_name == 'K-Medoids':
                    labels = KMedoids(n_clusters=k, random_state=42, method='pam').fit_predict(X)
                    score = silhouette_score(X, labels)
                    if score > best_val_score: best_k, best_val_score = k, score
                else:
                    gmm = GaussianMixture(n_components=k, random_state=42).fit(X)
                    score = gmm.bic(X)
                    if score < best_val_score: best_k, best_val_score = k, score

            # 2. Fit with best K and evaluate external alignment
            if model_name == 'K-Means':
                final_labels = KMeans(n_clusters=best_k, random_state=42, n_init=10).fit_predict(X)
            elif model_name == 'K-Medoids':
                final_labels = KMedoids(n_clusters=best_k, random_state=42, method='pam').fit_predict(X)
            else:
                final_labels = GaussianMixture(n_components=best_k, random_state=42).fit_predict(X)

            ari = adjusted_rand_score(df[target_col].astype(str), final_labels)
            nmi = normalized_mutual_info_score(df[target_col].astype(str), final_labels)

            results.append({
                'Decomposition': path.stem,
                'Model': model_name,
                'Optimal_K': best_k,
                'Phase_ARI': ari,
                'Phase_NMI': nmi
            })

    return pd.DataFrame(results).sort_values(by=['Phase_ARI', 'Phase_NMI'], ascending=[False, False])

if __name__ == "__main__":
    PROJECT_ROOT = Path(os.getcwd()).resolve().parents[1] 
    
    data_paths = [
        PROJECT_ROOT / 'data' / 'processed' / 'final' / 'HR_data_pca.csv', 
        PROJECT_ROOT / 'data' / 'processed' / 'final' / 'HR_data_spca.csv', 
        PROJECT_ROOT / 'data' / 'processed' / 'final' / 'HR_data_umap.csv'
    ]

    print("Evaluating combinations...")
    leaderboard = evaluate_best_combinations(data_paths, target_col='Phase', max_k=5)
    
    print("\n--- LEADERBOARD (Sorted by best Phase Alignment) ---")
    print(leaderboard.to_string(index=False))
    
    # Save the leaderboard
    out_dir = PROJECT_ROOT / 'final' / 'clustering' / 'results'
    out_dir.mkdir(parents=True, exist_ok=True)
    leaderboard.to_csv(out_dir / '0_combination_leaderboard.csv', index=False)