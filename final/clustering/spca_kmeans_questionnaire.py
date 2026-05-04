import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from math import pi

# --- PATH CONFIGURATION ---
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

FIGURES_DIR = Path(__file__).resolve().parent / 'figures_triangulation'
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR = PROJECT_ROOT / 'data' / 'processed' / 'final'

# --- EMOTION DEFINITIONS ---
ALL_EMOTIONS = [
    'inspired', 'alert', 'attentive', 'active', 'determined', # Positive
    'Frustrated', 'upset', 'hostile', 'ashamed', 'nervous', 'afraid' # Negative
]

def plot_radar_chart(df_pivot, title):
    """
    Creates a radar chart comparing the emotional profiles of the clusters.
    """
    categories = list(df_pivot.columns)
    N = len(categories)
    
    # Repeat the first value to close the circular graph
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    
    plt.figure(figsize=(10, 10))
    ax = plt.subplot(111, polar=True)
    
    # Draw one axe per variable + add labels
    plt.xticks(angles[:-1], categories, color='grey', size=10)
    
    # Plot each cluster
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    for i, (idx, row) in enumerate(df_pivot.iterrows()):
        values = row.values.flatten().tolist()
        values += values[:1]
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=f"Cluster {idx}", color=colors[i % 5])
        ax.fill(angles, values, color=colors[i % 5], alpha=0.1)
    
    plt.title(title, size=15, color='black', y=1.1)
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'radar_emotional_profile.png', dpi=300)

def main():
    # 1. Load SPCA Data (Best performing model)
    path_spca = PROCESSED_DIR / 'HR_data_spca.csv'
    if not path_spca.exists():
        print(f"Error: File {path_spca} not found!")
        return
    df = pd.read_csv(path_spca)
    
    # 2. Data Cleaning & Normalization
    if 'phase' in df.columns: 
        df.rename(columns={'phase': 'Phase'}, inplace=True)

    # 3. Clustering Execution (SPCA + K-Means)
    meta_cols = ['original ID', 'raw_data Path', 'Team ID', 'Individual', 'Phase', 'Cohort', 'Round', 'Role', 'Puzzler'] + ALL_EMOTIONS
    features = [c for c in df.select_dtypes(include=[np.number]).columns if c not in meta_cols]
    
    X_scaled = StandardScaler().fit_transform(df[features].fillna(df[features].median()))
    
    # Using K=2 as identified in your previous leaderboard
    best_k = 2 
    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    df['bio_cluster'] = kmeans.fit_predict(X_scaled)

    # 4. Detailed Emotional Profiling
    # Calculate average score for each of the 11 emotions per cluster
    cluster_profiles = df.groupby('bio_cluster')[ALL_EMOTIONS].mean()
    
    # Normalize by the global mean to see relative prominence (over/under representation)
    global_means = df[ALL_EMOTIONS].mean()
    relative_profiles = cluster_profiles / global_means

    # 5. Visualization: Detailed Heatmap
    plt.figure(figsize=(14, 6))
    sns.heatmap(relative_profiles, annot=True, fmt='.2f', cmap='coolwarm', center=1.0)
    plt.title("Relative Emotional Significance (1.0 = Average)")
    plt.xlabel("Individual Emotion")
    plt.ylabel("Bio-Physiological Cluster")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'detailed_emotional_heatmap.png', dpi=300)

    # 6. Visualization: Radar Chart
    # We use absolute means for the radar chart to show raw intensities
    plot_radar_chart(cluster_profiles, "Emotional Fingerprint of Bio-Clusters")

    # 7. Triangulation with Experiment Phases
    phase_triangulation = pd.crosstab(df['bio_cluster'], df['Phase'], normalize='columns')
    
    plt.figure(figsize=(10, 6))
    sns.heatmap(phase_triangulation, annot=True, fmt='.2f', cmap='Greens')
    plt.title("Cluster Distribution across Experiment Phases")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'cluster_phase_alignment.png', dpi=300)

    # 8. Save Final Report
    output_path = PROCESSED_DIR / 'HR_data_detailed_triangulation.csv'
    df.to_csv(output_path, index=False)
    
    print("\n--- ANALYSIS COMPLETE ---")
    print(f"1. Detailed Heatmap saved to figures_triangulation/detailed_emotional_heatmap.png")
    print(f"2. Radar Chart saved to figures_triangulation/radar_emotional_profile.png")
    print(f"3. Dataset with clusters and emotional scores saved to {output_path}")

if __name__ == '__main__':
    main()