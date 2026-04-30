import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

# Importing hierarchical clustering utilities instead of GMM utilities
from final.dendrogram.utils import (
    fit_hierarchical, 
    plot_dendrogram,
    plot_silhouette_scores,
    plot_clusters_with_hulls,
    plot_phases, 
    contingency_heatmap
)

PROCESSED_DIR = PROJECT_ROOT / 'data' / 'processed'
REPORT_DIR = Path(__file__).resolve().parent / 'report' / 'questionnaire'

META_COLS = ['Round', 'Phase', 'Individual', 'Puzzler', 'Cohort']
QUEST_COLS = ['Frustrated', 'upset', 'hostile', 'alert', 'ashamed',
              'inspired', 'nervous', 'attentive', 'afraid', 'active', 'determined']


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    
    # ── 1. Load and prepare questionnaire data ────────────────────────────────
    df = pd.read_csv(PROCESSED_DIR / 'HR_data_2.csv')
    df = df.dropna(subset=QUEST_COLS).reset_index(drop=True)
    X_quest = df[QUEST_COLS].values
    meta = df[META_COLS].copy()
    meta['Role'] = meta['Puzzler'].map({1: 'Active', 0: 'Instructor'})
    print(f"Questionnaire: {len(df)} rows, {len(QUEST_COLS)} features")

    # ── 2. Fit hierarchical clustering, choose k via silhouette ───────────────
    Z, labels, best_k, sil_scores = fit_hierarchical(X_quest, k_range=range(2, 6))
    print(f"Best: k={best_k}, silhouette={sil_scores[best_k]:.3f} (scores: {sil_scores})")

    # ── 3. Dimensionality reduction for visualization ─────────────────────────
    pca_2d = PCA(n_components=2, random_state=42)
    canvas_2d = pca_2d.fit_transform(X_quest)

    # ── 4. Dendrogram with cut line ───────────────────────────────────────────
    plot_dendrogram(
        Z, best_k,
        title=f'Dendrogram — Questionnaire (k={best_k}, Ward linkage)',
        output_path=REPORT_DIR / 'questionnaire_dendrogram.png'
    )

    # ── 5. Silhouette scores (k-selection rationale) ──────────────────────────
    plot_silhouette_scores(
        sil_scores, best_k,
        title='Silhouette scores — Questionnaire',
        output_path=REPORT_DIR / 'questionnaire_silhouette_scores.png'
    )

    # ── 6. 2-D scatter coloured by cluster (convex hulls) ─────────────────────
    plot_clusters_with_hulls(
        canvas_2d, labels, best_k,
        title=f'Hierarchical clusters — Questionnaire (k={best_k})',
        xlabel='PC1', ylabel='PC2',
        output_path=REPORT_DIR / 'questionnaire_clusters_hulls.png'
    )

    # ── 7. Phase overlay on the same 2-D canvas ───────────────────────────────
    plot_phases(
        canvas_2d, meta['Phase'].values,
        title='Phase overlay — Questionnaire',
        xlabel='PC1', ylabel='PC2',
        output_path=REPORT_DIR / 'questionnaire_phases.png'
    )

    # ── 8. Contingency heatmaps ───────────────────────────────────────────────
    for var_name, col in [
        ('Phase', meta['Phase']),
        ('Cohort', meta['Cohort']),
        ('Round', meta['Round']),
        ('Role', meta['Role']),
    ]:
        contingency_heatmap(
            labels, col.values, var_name,
            title=f'Questionnaire clusters × {var_name}',
            output_path=REPORT_DIR / f'questionnaire_contingency_{var_name}.png'
        )
        
    print(f"Questionnaire Dendrogram complete. Figures → {REPORT_DIR}")


if __name__ == '__main__':
    main()