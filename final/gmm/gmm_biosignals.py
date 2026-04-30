import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from final.preprocessing import load_and_clean, normalize_by_individual
from final.gmm.utils import (fit_gmm_bic, plot_clusters_with_ellipses,
                               plot_phases, contingency_heatmap,
                               discriminating_features)

PROCESSED_DIR = PROJECT_ROOT / 'data' / 'processed'
REPORT_DIR = Path(__file__).resolve().parent / 'report' / 'biosignals'

META_COLS = ['Round', 'Phase', 'Individual', 'Puzzler', 'Cohort']


def load_normalized_biosignals():
    df, biosig, _, _ = load_and_clean(PROCESSED_DIR / 'HR_data_2.csv')
    df_norm = normalize_by_individual(df, biosig)
    return df_norm, biosig


def run_dataset(name, df_norm, biosig_cols):
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(PROCESSED_DIR / 'final' / f'HR_data_{name}.csv')

    round_idx = df.columns.get_loc('Round')
    feature_cols = df.columns[:round_idx].tolist()
    X = df[feature_cols].values
    meta = df[META_COLS].copy()
    meta['Role'] = meta['Puzzler'].map({1: 'Active', 0: 'Instructor'})

    best_model, labels = fit_gmm_bic(X)
    k = best_model.n_components
    cov_type = best_model.covariance_type
    print(f"  {name.upper()}: k={k}, cov={cov_type}, BIC={best_model.bic(X):.1f}")

    canvas_2d = X[:, :2]
    xlabel = 'UMAP1' if name == 'umap' else 'PC1'
    ylabel = 'UMAP2' if name == 'umap' else 'PC2'

    plot_clusters_with_ellipses(
        canvas_2d, labels, k, cov_type,
        title=f'GMM clusters — {name.upper()} (k={k})',
        xlabel=xlabel, ylabel=ylabel,
        output_path=REPORT_DIR / f'{name}_clusters_ellipses.png'
    )
    plot_phases(
        canvas_2d, meta['Phase'].values,
        title=f'Phase overlay — {name.upper()}',
        xlabel=xlabel, ylabel=ylabel,
        output_path=REPORT_DIR / f'{name}_phases.png'
    )
    for var_name, col in [
        ('Phase', meta['Phase']),
        ('Cohort', meta['Cohort']),
        ('Round', meta['Round']),
        ('Role', meta['Role']),
    ]:
        contingency_heatmap(
            labels, col.values, var_name,
            title=f'{name.upper()} clusters × {var_name}',
            output_path=REPORT_DIR / f'{name}_contingency_{var_name}.png'
        )
    discriminating_features(
        labels, df_norm, biosig_cols,
        output_dir=REPORT_DIR, prefix=name
    )
    print(f"  Saved {name} outputs → {REPORT_DIR}")


def main():
    print("Loading normalized biosignals...")
    df_norm, biosig_cols = load_normalized_biosignals()
    for name in ['pca', 'spca', 'umap']:
        print(f"\n=== {name.upper()} ===")
        run_dataset(name, df_norm, biosig_cols)
    print("\nBiosignals GMM complete.")


if __name__ == '__main__':
    main()
