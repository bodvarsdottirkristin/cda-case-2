import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import umap

# =========================
# 0. Paths and imports
# =========================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from dim_reduction.utils.high_corr import highly_corr

FIGURES_DIR = Path(__file__).resolve().parent / 'figures'
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

PROCESSED_DIR = PROJECT_ROOT / 'data' / 'processed'
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# 1. Load already processed data
# =========================
input_path = PROJECT_ROOT / 'data' / 'processed' / 'HR_data_2.csv'
df = pd.read_csv(input_path)

meta_cols = ['Round', 'Phase', 'Individual', 'Puzzler', 'Cohort']
questionnaire_cols = [
    'Frustrated', 'upset', 'hostile', 'alert', 'ashamed', 'inspired',
    'nervous', 'attentive', 'afraid', 'active', 'determined'
]

numeric_cols = df.select_dtypes(include='number').columns.tolist()
biosignal_cols = [c for c in numeric_cols if c not in meta_cols + questionnaire_cols]


# =========================
# 2. Drop highly correlated features
# =========================
redundant = highly_corr(df[biosignal_cols], perf=0.95)
remaining_biosignals = [c for c in biosignal_cols if c not in redundant]

print(f"Original biosignal features: {len(biosignal_cols)}")
print(f"Features after correlation drop: {len(remaining_biosignals)}")


# =========================
# 3. Scale features phase-wise
# =========================
phase_df = df.copy()

def safe_standardize(series):
    std = series.std(ddof=0)
    if pd.isna(std) or std == 0:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - series.mean()) / std

for col in remaining_biosignals:
    phase_df[col] = phase_df.groupby('Phase')[col].transform(safe_standardize)

X = phase_df[remaining_biosignals]

print("Remaining NaNs before UMAP:", X.isna().sum().sum())
if X.isna().sum().sum() > 0:
    raise ValueError("There are still NaNs in X before UMAP.")


# =========================
# 4. UMAP in 10D for clustering
# =========================
reducer_10d = umap.UMAP(
    n_components=10,
    n_neighbors=15,
    min_dist=0.1,
    metric='euclidean',
    random_state=42
)

X_umap_10d = reducer_10d.fit_transform(X)

df_umap_10d = pd.DataFrame(
    X_umap_10d,
    columns=[f'UMAP{i+1}' for i in range(10)],
    index=phase_df.index
)

df_umap_10d = pd.concat([df_umap_10d, phase_df[meta_cols]], axis=1)
df_umap_10d.to_csv(PROCESSED_DIR / 'HR_data_umap_10d.csv', index=False)


# =========================
# 5. UMAP in 2D for visualization
# =========================
reducer_2d = umap.UMAP(
    n_components=2,
    n_neighbors=15,
    min_dist=0.1,
    metric='euclidean',
    random_state=42
)

X_umap_2d = reducer_2d.fit_transform(X)

df_umap_2d = pd.DataFrame(
    X_umap_2d,
    columns=['UMAP1', 'UMAP2'],
    index=phase_df.index
)

df_umap_2d = pd.concat([df_umap_2d, phase_df[meta_cols]], axis=1)


# =========================
# 6. Plot 2D UMAP colored by Phase
# =========================
plt.figure(figsize=(8, 6))

for phase in sorted(df_umap_2d['Phase'].dropna().unique()):
    mask = df_umap_2d['Phase'] == phase
    plt.scatter(
        df_umap_2d.loc[mask, 'UMAP1'],
        df_umap_2d.loc[mask, 'UMAP2'],
        label=str(phase),
        alpha=0.8
    )

plt.xlabel('UMAP1')
plt.ylabel('UMAP2')
plt.title('2D UMAP embedding colored by Phase')
plt.legend(title='Phase')
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'umap_2d_by_phase.png', dpi=300)
plt.show()


# =========================
# 7. KMeans on 10D UMAP
# =========================
n_clusters = 3

kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
cluster_labels = kmeans.fit_predict(X_umap_10d)

sil_score = silhouette_score(X_umap_10d, cluster_labels)
print(f"Silhouette score on 10D UMAP with k={n_clusters}: {sil_score:.3f}")

df_umap_10d['Cluster'] = cluster_labels
df_umap_2d['Cluster'] = cluster_labels


# =========================
# 8. Plot 2D UMAP colored by KMeans cluster
# =========================
plt.figure(figsize=(8, 6))

for cluster in sorted(df_umap_2d['Cluster'].unique()):
    mask = df_umap_2d['Cluster'] == cluster
    plt.scatter(
        df_umap_2d.loc[mask, 'UMAP1'],
        df_umap_2d.loc[mask, 'UMAP2'],
        label=f'Cluster {cluster}',
        alpha=0.8
    )

plt.xlabel('UMAP1')
plt.ylabel('UMAP2')
plt.title(f'2D UMAP visualization + KMeans on 10D UMAP (k={n_clusters})')
plt.legend(title='Cluster')
plt.tight_layout()
plt.savefig(FIGURES_DIR / f'umap_2d_kmeans_from_10d_k{n_clusters}.png', dpi=300)
plt.show()


# =========================
# 9. Optional: test several k values
# =========================
print("\nSilhouette scores for different k (on 10D UMAP):")
for k in [2, 3, 4, 5, 6]:
    km = KMeans(n_clusters=k, random_state=42, n_init=20)
    labels = km.fit_predict(X_umap_10d)
    score = silhouette_score(X_umap_10d, labels)
    print(f"k={k}: silhouette={score:.3f}")


# =========================
# 10. Final summary
# =========================
print("\nSaved files:")
print(PROCESSED_DIR / 'HR_data_umap_10d.csv')
print(FIGURES_DIR / 'umap_2d_by_phase.png')
print(FIGURES_DIR / f'umap_2d_kmeans_from_10d_k{n_clusters}.png')