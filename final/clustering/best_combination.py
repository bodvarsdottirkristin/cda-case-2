import os
import sys
import warnings
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from scipy.stats import chi2_contingency

warnings.filterwarnings('ignore')

def calculate_cramers_v(contingency_table):
    chi2 = chi2_contingency(contingency_table)[0]
    n = contingency_table.sum().sum()
    if n == 0: return 0
    phi2 = chi2 / n
    r, k = contingency_table.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    denom = min((kcorr-1), (rcorr-1))
    return np.sqrt(phi2corr / denom) if denom > 0 else 0

def evaluate_best_combinations(data_paths, target_cols=['Phase', 'Cohort', 'Puzzler', 'Round'], max_k=5):
    results = []
    
    # Define columns to exclude from the clustering features
    questionnaire_cols = ['Frustrated', 'upset', 'hostile', 'alert', 'ashamed', 'inspired', 'nervous', 'attentive', 'afraid', 'active', 'determined']
    meta_cols = ['original ID', 'raw_data Path', 'Team ID', 'Individual', 'Phase', 'phase', 'Cohort', 'Round', 'Role', 'Puzzler'] + questionnaire_cols

    for path in data_paths:
        if not path.exists():
            print(f"File not found, skipping: {path.name}")
            continue
            
        df = pd.read_csv(path)
        if 'phase' in df.columns: 
            df.rename(columns={'phase': 'Phase'}, inplace=True)
        
        # Extract ONLY the reduced dimensions (e.g., PC1, PC2, UMAP1)
        # CRITICAL: NO StandardScaler() applied here!
        features = [c for c in df.select_dtypes(include=[np.number]).columns if c not in meta_cols]
        if len(features) < 2: 
            continue
            
        X = df[features].fillna(df[features].median()).values

        for model_name in ['K-Means', 'K-Medoids', 'GMM']:
            best_k = 2
            best_val_score = -np.inf if model_name != 'GMM' else np.inf

            # 1. Find Optimal K on the reduced space
            for k in range(2, max_k + 1):
                try:
                    if model_name == 'K-Means':
                        m = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X)
                        score = silhouette_score(X, m.labels_)
                        if score > best_val_score: best_k, best_val_score = k, score
                    elif model_name == 'K-Medoids':
                        m = KMedoids(n_clusters=k, random_state=42, method='pam').fit(X)
                        score = silhouette_score(X, m.labels_)
                        if score > best_val_score: best_k, best_val_score = k, score
                    else:
                        gmm = GaussianMixture(n_components=k, random_state=42).fit(X)
                        score = gmm.bic(X)
                        if score < best_val_score: best_k, best_val_score = k, score
                except Exception as e:
                    continue

            # 2. Fit the best model
            if model_name == 'K-Means':
                final_labels = KMeans(n_clusters=best_k, random_state=42, n_init=10).fit_predict(X)
            elif model_name == 'K-Medoids':
                final_labels = KMedoids(n_clusters=best_k, random_state=42, method='pam').fit_predict(X)
            else:
                final_labels = GaussianMixture(n_components=best_k, random_state=42).fit_predict(X)

            # 3. Evaluate against all target variables
            for target in target_cols:
                if target not in df.columns: 
                    continue
                
                y_true = df[target].astype(str)
                ari = adjusted_rand_score(y_true, final_labels)
                nmi = normalized_mutual_info_score(y_true, final_labels)
                
                contingency_table = pd.crosstab(y_true, final_labels)
                chi2, p_val, _, _ = chi2_contingency(contingency_table)
                v_cramer = calculate_cramers_v(contingency_table)

                results.append({
                    'Target_Variable': target,
                    'Decomposition': path.stem,
                    'Model': model_name,
                    'Optimal_K': best_k,
                    'ARI': ari,
                    'NMI': nmi,
                    'Chi2_p-value': p_val,
                    'Cramers_V': v_cramer
                })

    if not results: 
        return pd.DataFrame()
    
    return pd.DataFrame(results).sort_values(by=['Target_Variable', 'ARI', 'Cramers_V'], ascending=[True, False, False])

if __name__ == "__main__":
    
    # Handle paths dynamically based on your folder structure
    current_file = Path(__file__).resolve()
    PROJECT_ROOT = current_file.parents[2]
    
    data_paths = [
        PROJECT_ROOT / 'data' / 'processed' / 'final' / 'HR_data_pca.csv', 
        PROJECT_ROOT / 'data' / 'processed' / 'final' / 'HR_data_spca.csv', 
        PROJECT_ROOT / 'data' / 'processed' / 'final' / 'HR_data_umap.csv'
    ]

    targets = ['Phase', 'Cohort', 'Puzzler', 'Round']
    print("Evaluating best combinations on pre-reduced data (No Scaling)...")
    leaderboard = evaluate_best_combinations(data_paths, target_cols=targets)
    
    if not leaderboard.empty:
        out_dir = current_file.parent / 'results'
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_path = out_dir / '0_combination_leaderboard.csv'
        
        # Save to CSV
        leaderboard.to_csv(csv_path, index=False)
        
        # Print a nice summary to terminal (Top 15 sorted by ARI globally)
        print("\n--- GLOBAL LEADERBOARD (Top 15 sorted by ARI) ---")
        top_15 = leaderboard.sort_values(by='ARI', ascending=False).head(15)
        print(top_15.to_string(index=False))
        
        print(f"\nFull leaderboard successfully saved in: {csv_path}")
    else:
        print("No results generated. Please check if the file paths are correct.")