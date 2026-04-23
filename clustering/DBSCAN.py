import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIGURES_DIR = Path(__file__).resolve().parent / 'figures'
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
sys.path.append(str(PROJECT_ROOT))

# =========================
# 1. Load SparsePCA-reduced data
# =========================
df = pd.read_csv(PROJECT_ROOT / 'data' / 'processed' / 'HR_data_PCA2.csv')

meta_cols = ['Round', 'Phase', 'Individual', 'Puzzler', 'Cohort']
pc_cols = [c for c in df.columns if c.startswith('PC')]

X = df[pc_cols].values
print(f"Using {len(pc_cols)} PCA components, {len(df)} samples")

# =========================
# 2. k-distance plot to guide eps selection
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
# 3. Scan for good eps/min_samples
# =========================
for eps in [0.1, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0]:
    for min_s in [3, 5]:
        labels = DBSCAN(eps=eps, min_samples=min_s).fit_predict(X)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = (labels == -1).sum()
        print(f"eps={eps}, min_samples={min_s}: clusters={n_clusters}, noise={n_noise} ({n_noise/len(labels)*100:.1f}%)")

# =========================
# 4. Fit DBSCAN with chosen parameters
# =========================
eps = 5.0
min_samples = 5
labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X)

n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = (labels == -1).sum()
print(f"\nFinal — Clusters: {n_clusters}, Noise: {n_noise} ({n_noise/len(labels)*100:.1f}%)")

# =========================
# 5. Plot clusters on first two PCA components
# =========================
plt.figure(figsize=(8, 6))
scatter = plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='tab10', s=20)
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
