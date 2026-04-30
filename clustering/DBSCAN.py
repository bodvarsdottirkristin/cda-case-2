import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIGURES_DIR = Path(__file__).resolve().parent / 'figures'
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
sys.path.append(str(PROJECT_ROOT))

if len(sys.argv) < 2:
    print("Usage: python DBSCAN.py NAME_OF_CSV_FILE")
    sys.exit(1)

csv_name = sys.argv[1]

# =========================
# 1. Load SparsePCA-reduced data
# =========================
df = pd.read_csv(PROJECT_ROOT / 'data' / 'processed' / 'final' / f'{csv_name}.csv')

meta_cols = ['Round', 'Phase', 'Individual', 'Puzzler', 'Cohort']
if 'umap' in csv_name.lower():
    feature_cols = [c for c in df.columns if c.startswith('UMAP')]
else:
    feature_cols = [c for c in df.columns if c.startswith('PC')]

X = df[feature_cols].values
print(f"Using {len(feature_cols)} features, {len(df)} samples")

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
plt.savefig(FIGURES_DIR / f'dbscan_kdistance_{csv_name}.png', dpi=300)

# =========================
# 3. Scan for good eps/min_samples using silhouette score
# =========================
best_score = -1
best_params = None
best_labels = None

for eps in [0.1, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0]:
    for min_s in [3, 5]:
        labels = DBSCAN(eps=eps, min_samples=min_s).fit_predict(X)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = (labels == -1).sum()
        # silhouette requires at least 2 clusters and at least 1 non-noise point
        mask = labels != -1
        if n_clusters >= 2 and mask.sum() > 1:
            score = silhouette_score(X[mask], labels[mask])
        else:
            score = -1
        print(f"eps={eps}, min_samples={min_s}: clusters={n_clusters}, noise={n_noise} ({n_noise/len(labels)*100:.1f}%), silhouette={score:.3f}")
        if score > best_score:
            best_score = score
            best_params = (eps, min_s)
            best_labels = labels

eps, min_samples = best_params
labels = best_labels
print(f"\nBest params: eps={eps}, min_samples={min_samples}, silhouette={best_score:.3f}")

# =========================
# 4. Fit DBSCAN with best parameters
# =========================

n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = (labels == -1).sum()
print(f"\nFinal — Clusters: {n_clusters}, Noise: {n_noise} ({n_noise/len(labels)*100:.1f}%)")

# =========================
# 5. Plot: DBSCAN clusters vs Phase labels (side by side)
# =========================
ax1_label = feature_cols[0]
ax2_label = feature_cols[1]

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Left: coloured by DBSCAN cluster assignment
scatter = axes[0].scatter(X[:, 0], X[:, 1], c=labels, cmap='tab10', s=20, alpha=0.8)
plt.colorbar(scatter, ax=axes[0], label='Cluster')
axes[0].set_xlabel(ax1_label)
axes[0].set_ylabel(ax2_label)
axes[0].set_title(f'DBSCAN clusters (eps={eps}, min_samples={min_samples})')

# Right: coloured by Phase variable
phases = sorted(df['Phase'].dropna().unique())
cmap_phase = plt.get_cmap('tab10')
for i, phase in enumerate(phases):
    mask = (df['Phase'] == phase).values
    axes[1].scatter(X[mask, 0], X[mask, 1], label=phase,
                    color=cmap_phase(i), s=20, alpha=0.8)
axes[1].set_xlabel(ax1_label)
axes[1].set_ylabel(ax2_label)
axes[1].set_title('Coloured by Phase')
axes[1].legend(title='Phase')

plt.tight_layout()
plt.savefig(FIGURES_DIR / f'dbscan_clusters_{csv_name}.png', dpi=300)

# =========================
# 6. Save results
# =========================
df_out = df.copy()
df_out['cluster'] = labels
df_out.to_csv(PROJECT_ROOT / 'data' / 'processed' / 'final'/ f'{csv_name}_dbscan.csv', index=False)
print(f"Saved with cluster labels: {df_out.shape}")

plt.show()
