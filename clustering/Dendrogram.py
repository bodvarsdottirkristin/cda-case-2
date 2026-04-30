import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════════════════════
# CHANGE K HERE
K = 5
# ══════════════════════════════════════════════════════════════════════════════

# ── Load & prepare data ───────────────────────────────────────────────────────
df = pd.read_csv('../data/processed/HR_data_2.csv')

phys_cols = [c for c in df.columns if any(c.startswith(p) for p in ['HR_TD', 'TEMP_TD', 'EDA_TD'])]
emotion_cols = ['upset', 'hostile', 'alert', 'ashamed', 'inspired', 'nervous',
                'attentive', 'afraid', 'active', 'determined', 'Frustrated']
all_feat_cols = phys_cols + emotion_cols

X = df[all_feat_cols].dropna()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Compute linkage once (used in multiple plots)
sample_linkage = linkage(X_scaled, method='ward', metric='euclidean')
feat_linkage   = linkage(X_scaled.T, method='ward', metric='euclidean')

# Assign cluster labels
labels = fcluster(sample_linkage, t=K, criterion='maxclust')
df = df.copy()
df['Cluster'] = labels

# ── PLOT 1: Dendrogram on FEATURES (columns) ──────────────────────────────────
fig, ax = plt.subplots(figsize=(16, 6))
dendrogram(feat_linkage, labels=all_feat_cols, ax=ax, leaf_rotation=90,
           leaf_font_size=7.5, color_threshold=0.7 * max(feat_linkage[:, 2]))
ax.set_title('Hierarchical Clustering of Features\n(Ward linkage, Euclidean distance)', fontsize=14, fontweight='bold')
ax.set_xlabel('Features', fontsize=11)
ax.set_ylabel('Distance', fontsize=11)
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
plt.savefig(f'figures/dendrogram_features.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Plot 1 saved: figures/dendrogram_features.png")

# ── PLOT 2: Dendrogram on OBSERVATIONS coloured by Phase ──────────────────────
phase_map    = {'phase1': '#4C72B0', 'phase2': '#DD8452', 'phase3': '#55A868'}
phase_colors = df['Phase'].map(phase_map).values

fig, ax = plt.subplots(figsize=(18, 6))
dend = dendrogram(sample_linkage, ax=ax, no_labels=True,
                  color_threshold=0.4 * max(sample_linkage[:, 2]),
                  above_threshold_color='grey')
ax.set_title(f'Hierarchical Clustering of Observations (n={len(df)})\nPhase shown in strip below',
             fontsize=14, fontweight='bold')
ax.set_xlabel('Observations', fontsize=11)
ax.set_ylabel('Distance', fontsize=11)
ax.spines[['top', 'right']].set_visible(False)

leaves  = dend['leaves']
strip_h = ax.get_ylim()[1] * 0.04
for i, idx in enumerate(leaves):
    ax.add_patch(mpatches.Rectangle((10 * i, -strip_h * 1.5), 10, strip_h,
                 color=phase_colors[idx], transform=ax.transData, clip_on=False))

legend_handles = [mpatches.Patch(color=v, label=k) for k, v in phase_map.items()]
ax.legend(handles=legend_handles, title='Phase', loc='upper right', fontsize=9)
plt.tight_layout()
plt.savefig(f'figures/dendrogram_observations.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Plot 2 saved: figures/dendrogram_observations.png")

# ── PLOT 3: Clustermap ────────────────────────────────────────────────────────
phase_lut   = {'phase1': '#4C72B0', 'phase2': '#DD8452', 'phase3': '#55A868'}
puzzler_lut = {0: '#AECDE8', 1: '#F4A582'}
row_colors  = pd.DataFrame({
    'Phase':   df['Phase'].map(phase_lut),
    'Puzzler': df['Puzzler'].map(puzzler_lut)
})

g = sns.clustermap(
    pd.DataFrame(X_scaled, columns=all_feat_cols),
    method='ward', metric='euclidean',
    row_colors=row_colors,
    cmap='RdBu_r', center=0,
    xticklabels=True, yticklabels=False,
    figsize=(20, 14),
    dendrogram_ratio=(0.15, 0.2),
    cbar_pos=(0.02, 0.8, 0.03, 0.15)
)
g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xticklabels(), fontsize=7, rotation=90)
g.figure.suptitle('Clustermap: Observations × Features\n(Ward linkage)', fontsize=15, fontweight='bold', y=1.01)

patches  = [mpatches.Patch(color=v, label=k) for k, v in phase_lut.items()]
patches += [mpatches.Patch(color=v, label=f'Puzzler={k}') for k, v in puzzler_lut.items()]
g.ax_heatmap.legend(handles=patches, loc='lower right', bbox_to_anchor=(1.25, 0),
                    fontsize=8, title='Groups', title_fontsize=9)
plt.savefig(f'figures/clustermap.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Plot 3 saved: figures/clustermap.png")

# ── PLOT 4: Elbow + Cluster Emotion Profile ───────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

last = sample_linkage[-20:, 2][::-1]
axes[0].plot(range(1, 21), last, 'o-', color='#4C72B0', linewidth=2, markersize=6)
axes[0].axvline(x=K, color='tomato', linestyle='--', label=f'k={K}')
axes[0].set_title('Elbow Plot\n(Last 20 merging distances)', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Number of Clusters', fontsize=11)
axes[0].set_ylabel('Merge Distance', fontsize=11)
axes[0].legend()
axes[0].spines[['top', 'right']].set_visible(False)

df_plot   = df[all_feat_cols + ['Cluster']].copy()
profile   = df_plot.groupby('Cluster')[emotion_cols].mean()
profile_z = (profile - profile.mean()) / profile.std()

im = axes[1].imshow(profile_z.values, cmap='coolwarm', aspect='auto')
axes[1].set_xticks(range(len(emotion_cols)))
axes[1].set_xticklabels(emotion_cols, rotation=45, ha='right', fontsize=9)
axes[1].set_yticks(range(K))
axes[1].set_yticklabels([f'Cluster {i+1}' for i in range(K)], fontsize=10)
axes[1].set_title(f'Emotion Profile per Cluster (k={K})\n(Z-scored means)', fontsize=13, fontweight='bold')
plt.colorbar(im, ax=axes[1], label='Z-score')

plt.tight_layout()
plt.savefig(f'figures/elbow_and_profile_k{K}.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Plot 4 saved: figures/elbow_and_profile_k{K}.png")

# ── PLOT 5: Phase distribution within clusters ────────────────────────────────
phase_cluster     = df.groupby(['Cluster', 'Phase']).size().unstack(fill_value=0)
phase_cluster_pct = phase_cluster.div(phase_cluster.sum(axis=1), axis=0) * 100

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
colors = [phase_map.get(p, '#999999') for p in phase_cluster.columns]

# Left: raw counts
phase_cluster.plot(kind='bar', ax=axes[0], color=colors, edgecolor='white', width=0.7)
axes[0].set_title(f'Phase Counts per Cluster (k={K})', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Cluster', fontsize=11)
axes[0].set_ylabel('Count', fontsize=11)
axes[0].set_xticklabels([f'Cluster {i}' for i in phase_cluster.index], rotation=0)
axes[0].legend(title='Phase', fontsize=9)
axes[0].spines[['top', 'right']].set_visible(False)

# Right: proportions stacked 100%
phase_cluster_pct.plot(kind='bar', stacked=True, ax=axes[1], color=colors, edgecolor='white', width=0.7)
axes[1].set_title(f'Phase Proportions per Cluster (k={K})', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Cluster', fontsize=11)
axes[1].set_ylabel('Percentage (%)', fontsize=11)
axes[1].set_xticklabels([f'Cluster {i}' for i in phase_cluster_pct.index], rotation=0)
axes[1].legend(title='Phase', fontsize=9, bbox_to_anchor=(1.01, 1), loc='upper left')
axes[1].spines[['top', 'right']].set_visible(False)

plt.tight_layout()
plt.savefig(f'figures/phase_distribution_k{K}.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Plot 5 saved: figures/phase_distribution_k{K}.png")

print(f"\nCluster sizes (k={K}):")
print(pd.Series(labels).value_counts().sort_index())