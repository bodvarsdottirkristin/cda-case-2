import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIGURES_DIR = Path(__file__).resolve().parent / 'figures'
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
sys.path.append(str(PROJECT_ROOT))

from dim_reduction.utils.high_corr import highly_corr

# =========================
# 1. Load data
# =========================
df = pd.read_csv(PROJECT_ROOT / 'data' / 'processed' / 'HR_data_2.csv')

meta_cols = ['Round', 'Phase', 'Individual', 'Puzzler', 'Cohort']
questionnaire_cols = ['Frustrated', 'upset', 'hostile', 'alert', 'ashamed', 'inspired',
                      'nervous', 'attentive', 'afraid', 'active', 'determined']

numeric_cols = df.select_dtypes(include='number').columns.tolist()
biosignal_cols = [c for c in numeric_cols if c not in meta_cols + questionnaire_cols]

X = StandardScaler().fit_transform(df[biosignal_cols])

# =========================
# 3. Find optimal eps using k-distance plot
# =========================
k = 5
nbrs = NearestNeighbors(n_neighbors=k).fit(X)
distances, _ = nbrs.kneighbors(X)
k_distances = np.sort(distances[:, k - 1])[::-1]

plt.figure(figsize=(8, 5))
plt.plot(k_distances)
plt.xlabel('Points sorted by distance')
plt.ylabel(f'{k}-NN distance')
plt.title('k-distance plot (use elbow to pick eps)')
plt.grid(True)
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'dbscan_kdistance.png', dpi=300)

# =========================
# 4. Reduce dimensions with PCA before clustering
# =========================
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
print(f"Variance explained by 2 PCA components: {pca.explained_variance_ratio_.sum():.2%}")

# Scan for good eps on PCA-reduced data
for eps in [0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0]:
    for min_s in [3, 5]:
        labels = DBSCAN(eps=eps, min_samples=min_s).fit_predict(X_pca)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = (labels == -1).sum()
        print(f"eps={eps}, min_samples={min_s}: clusters={n_clusters}, noise={n_noise} ({n_noise/len(labels)*100:.1f}%)")

# =========================
# 5. Fit DBSCAN on PCA-reduced data
# =========================
eps = 1.0       # adjust based on scan above
min_samples = 5
labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X_pca)

n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = (labels == -1).sum()
print(f"\nFinal — Clusters: {n_clusters}, Noise: {n_noise} ({n_noise/len(labels)*100:.1f}%)")

X_2d = X_pca[:, :2]  # use first 2 PCA components for plotting

plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap='tab10', s=20)
plt.colorbar(scatter, label='Cluster')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title(f'DBSCAN clusters (eps={eps}, min_samples={min_samples})')
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'dbscan_clusters.png', dpi=300)

# =========================
# 6. Save results
# =========================
df_out = df.copy()
df_out['cluster'] = labels
df_out.to_csv(PROJECT_ROOT / 'data' / 'processed' / 'HR_data_dbscan.csv', index=False)
print(f"Saved with cluster labels: {df_out.shape}")

plt.show()
