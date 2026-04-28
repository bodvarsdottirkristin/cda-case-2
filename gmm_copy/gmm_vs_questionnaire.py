# PANAS and comparison
# this codes serves to evaluate the performance of gmm clustering - it compares the state obtained with possible clustering
# obtained from grouping together similar emotions. 
# Comparison is visual and metric
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Ellipse
from sklearn.mixture import GaussianMixture

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

FIGURES_DIR = Path(__file__).resolve().parent / 'figures_panas'
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR = PROJECT_ROOT / 'data' / 'processed'
RESULTS_DIR = PROJECT_ROOT / 'results' / 'tables'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Positive / negative 
PA_ITEMS = ['inspired', 'alert', 'attentive', 'active', 'determined']
# 'Frustrated' retains its capital-F as it appears in HR_data_2.csv
NA_ITEMS = ['Frustrated', 'upset', 'hostile', 'ashamed', 'nervous', 'afraid']
LABELS_PANA = {
    (True,  False): 'Engaged/Calm',
    (False, True):  'Tense/Stressed',
    (False, False): 'Drained/Disengaged',
    (True,  True):  'Alert/Anxious',
}

# Focuses on physical intensity (HR/EDA spikes) vs. quieter emotions
HIGH_AROUSAL_ITEMS = ['alert', 'active', 'nervous', 'afraid', 'Frustrated']
LOW_AROUSAL_ITEMS = ['inspired', 'attentive', 'determined', 'ashamed', 'upset']
LABELS_AROUSAL = {
    (True,  False): 'Reactive/Physiological', # High Arousal, Low Quiet
    (False, True):  'Reflective/Internal',    # Low Arousal, High Quiet
    (False, False): 'Baseline/Resting',       # Low both
    (True,  True):  'Peak Involvement',       # High both
}

# Focuses on interpersonal/competitive stress vs. internal feelings
SOCIAL_STRESS_ITEMS = ['hostile', 'Frustrated', 'ashamed']
INTERNAL_STRESS_ITEMS = ['nervous', 'afraid', 'upset']
LABELS_SOCIAL = {
    (True,  False): 'Competitive/Frustrated', # High Social, Low Internal
    (False, True):  'Internally Anxious',     # Low Social, High Internal
    (False, False): 'Neutral/Cooperative',    # Low both
    (True,  True):  'Overwhelmed/Conflict',   # High both
}

# Focuses on focus/determination vs. emotional distress
ENGAGEMENT_ITEMS = ['attentive', 'determined', 'active', 'alert']
DISTRESS_ITEMS = ['Frustrated', 'nervous', 'hostile', 'upset']
LABELS_ENGAGEMENT = {
    (True,  False): 'Focused/Locked-in',  # High Engagement, Low Distress
    (False, True):  'Stressed/Struggling', # Low Engagement, High Distress
    (False, False): 'Disengaged/Resting',  # Low both
    (True,  True):  'High-Pressure Flow',  # High both
}

META_COLS = ['Round', 'Phase', 'Individual', 'Puzzler', 'Cohort']


# uploading raw data and reduced ones
def load_data():
    X = pd.read_csv(PROCESSED_DIR / 'HR_data_2.csv')
    X_pca2 = pd.read_csv(PROCESSED_DIR / 'HR_data_PCA2.csv')
    return X, X_pca2

# compute panas scores on original data, but attach cols to reduced dataset (we do not keep full 66 features)
def compute_panas_scores(df, df_reduced, pos_ITEMS, neg_ITEMS):
    X = df_reduced.copy()
    X['pa_score'] = df[pos_ITEMS].mean(axis=1)
    X['na_score'] = df[neg_ITEMS].mean(axis=1)
    return X

def label_clusters(means, EMOTIONAL_LABELS):
    # means: array (k, 2), col 0 = pa_score, col 1 = na_score
    # Uses >= for median split; ties fall into the high side.
    pa_med = np.median(means[:, 0])
    na_med = np.median(means[:, 1])
    base_labels = [EMOTIONAL_LABELS[(pa >= pa_med, na >= na_med)] for pa, na in means]
    # Disambiguate duplicates when k > 4 (multiple clusters in same quadrant)
    counts: dict = {}
    result = []
    for label in base_labels:
        counts[label] = counts.get(label, 0) + 1
        result.append(label if counts[label] == 1 else f"{label} ({counts[label]})")
    return result

def main():
    # 1. Load raw data and PCA-reduced features
    raw, processed = load_data()

    # 2. Load the results of your previous GMM clustering (Biological States)
    # This file contains your 'cluster' labels and 'prob_cluster_X' columns
    bio_df = pd.read_csv(PROCESSED_DIR / 'HR_data_gmm.csv')

    # 3. Compute PA/NA scores from the questionnaire items
    # These items are based on the I-PANAS-SF scale [cite: 111, 112]
    df_with_scores = compute_panas_scores(raw, processed, PA_ITEMS, NA_ITEMS)

    # 4. Integrate Biological and Psychological data
    # We ensure rows align by merging on experimental keys [cite: 100, 106]
    df = pd.merge(
        bio_df, 
        df_with_scores[['pa_score', 'na_score']], 
        left_index=True, 
        right_index=True
    )

    # 5. Fit GMM to the PANAS space to find Psychological Clusters
    X_panas = df[['pa_score', 'na_score']].values
    
    # We use 'full' covariance to capture the shape of emotional quadrants
    gmm_panas = GaussianMixture(n_components=4, covariance_type='full', random_state=42, n_init=10)
    gmm_panas.fit(X_panas)
    
    # 6. Assign labels based on the PA/NA quadrants
    # Uses the median split logic to define 'Tense', 'Calm', etc.
    panas_labels = label_clusters(gmm_panas.means_, LABELS_PANA)
    df['emotional_cluster_id'] = gmm_panas.predict(X_panas)
    df['emotional_label'] = [panas_labels[i] for i in df['emotional_cluster_id']]

    # 7. THE TRIANGULATION: Compare Body (cluster) vs Mind (emotional_label)
    # We calculate the agreement percentage between the two methods
    comparison = pd.crosstab(
        df['emotional_label'], 
        df['cluster'], 
        normalize='index' 
    )

    # 8. Visualize the Comparison
    plt.figure(figsize=(10, 7))
    sns.heatmap(comparison, annot=True, fmt='.2f', cmap='YlGnBu')
    plt.title("Triangulation: Biological Clusters vs. Reported Emotions")
    plt.ylabel("Subjective Report (PANAS)")
    plt.xlabel("Biological Cluster (GMM on PCA)")
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'final_triangulation_heatmap.png', dpi=300)
    print(f"✅ Final Comparison saved to {FIGURES_DIR}")
    
    # 9. Output Summary for the Case Study
    print("\n--- Final Alignment Summary ---")
    for emo_state in comparison.index:
        best_match = comparison.columns[comparison.loc[emo_state].argmax()]
        strength = comparison.loc[emo_state].max()
        print(f"Emotion '{emo_state}' aligns with Bio-Cluster {best_match} ({strength:.1%})")

    # 10. Save Master Dataset for reporting
    df.to_csv(PROCESSED_DIR / 'HR_data_triangulated.csv', index=False)

if __name__ == '__main__':
    main()
