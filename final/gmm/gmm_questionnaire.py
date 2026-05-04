import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from final.gmm.utils import (fit_gmm_bic, plot_clusters_with_ellipses,
                               plot_phases, contingency_heatmap)

PROCESSED_DIR = PROJECT_ROOT / 'data' / 'processed'
REPORT_DIR = Path(__file__).resolve().parent / 'report' / 'questionnaire'

META_COLS = ['Round', 'Phase', 'Individual', 'Puzzler', 'Cohort']
QUEST_COLS = ['Frustrated', 'upset', 'hostile', 'alert', 'ashamed',
              'inspired', 'nervous', 'attentive', 'afraid', 'active', 'determined']


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(PROCESSED_DIR / 'HR_data_2.csv')
    df = df.dropna(subset=QUEST_COLS).reset_index(drop=True)
    X_quest = df[QUEST_COLS].values
    meta = df[META_COLS].copy()
    meta['Role'] = meta['Puzzler'].map({1: 'Active', 0: 'Instructor'})
    print(f"Questionnaire: {len(df)} rows, {len(QUEST_COLS)} features")

    best_model, labels = fit_gmm_bic(X_quest)
    k = best_model.n_components
    cov_type = best_model.covariance_type
    print(f"Best: k={k}, cov={cov_type}, BIC={best_model.bic(X_quest):.1f}")

    pca_2d = PCA(n_components=2, random_state=42)
    canvas_2d = pca_2d.fit_transform(X_quest)

    plot_clusters_with_ellipses(
        canvas_2d, labels, k, cov_type,
        title=f'GMM clusters — Questionnaire (k={k})',
        xlabel='PC1', ylabel='PC2',
        output_path=REPORT_DIR / 'questionnaire_clusters_ellipses.png'
    )
    plot_phases(
        canvas_2d, meta['Phase'].values,
        title='Phase overlay — Questionnaire',
        xlabel='PC1', ylabel='PC2',
        output_path=REPORT_DIR / 'questionnaire_phases.png'
    )
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
    print(f"Questionnaire GMM complete. Figures → {REPORT_DIR}")


if __name__ == '__main__':
    main()
