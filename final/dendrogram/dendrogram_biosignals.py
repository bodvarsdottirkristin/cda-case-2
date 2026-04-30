import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from final.preprocessing import load_and_clean, normalize_by_individual
from final.dendrogram.utils import (fit_hierarchical, plot_dendrogram,
                                     plot_silhouette_scores,
                                     plot_clusters_with_hulls,
                                     plot_phases, contingency_heatmap,
                                     discriminating_features)

PROCESSED_DIR = PROJECT_ROOT / 'data' / 'processed'
REPORT_DIR    = Path(__file__).resolve().parent / 'report' / 'biosignals'

META_COLS = ['Round', 'Phase', 'Individual', 'Puzzler', 'Cohort']


def load_normalized_biosignals():
    """Load HR_data_2.csv, clean and normalize by individual.

    Returns the normalized DataFrame and the biosignal column list,
    used later for the discriminating-features analysis.
    """
    df, biosig, _, _ = load_and_clean(PROCESSED_DIR / 'HR_data_2.csv')
    df_norm = normalize_by_individual(df, biosig)
    return df_norm, biosig


def run_dataset(name, df_norm, biosig_cols):
    """Run the full dendrogram pipeline for one dimensionality-reduction dataset.

    Steps
    -----
    1. Load the pre-reduced CSV (pca / spca / umap).
    2. Fit Ward linkage; select k via silhouette score (range 2–5).
    3. Plot: dendrogram, silhouette bar chart, 2-D scatter with convex hulls,
             phase overlay, contingency heatmaps (Phase / Cohort / Round / Role),
             discriminating biosignal features.

    Args:
        name      : one of 'pca', 'spca', 'umap'.
        df_norm   : individual-normalized DataFrame (for discriminating features).
        biosig_cols: list of biosignal column names.
    """
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(PROCESSED_DIR / 'final' / f'HR_data_{name}.csv')

    # Feature columns are everything to the left of 'Round'
    round_idx    = df.columns.get_loc('Round')
    feature_cols = df.columns[:round_idx].tolist()
    X    = df[feature_cols].values
    meta = df[META_COLS].copy()
    meta['Role'] = meta['Puzzler'].map({1: 'Active', 0: 'Instructor'})

    # ── 1. Fit hierarchical clustering, choose k via silhouette ──────────────
    Z, labels, best_k, sil_scores = fit_hierarchical(X, k_range=range(2, 6))
    print(f"  {name.upper()}: k={best_k}, "
          f"silhouette={sil_scores[best_k]:.3f}  "
          f"(scores: {sil_scores})")

    # Axis labels depend on the reduction method
    xlabel = 'UMAP1' if name == 'umap' else 'PC1'
    ylabel = 'UMAP2' if name == 'umap' else 'PC2'
    canvas_2d = X[:, :2]

    # ── 2. Dendrogram with cut line ───────────────────────────────────────────
    plot_dendrogram(
        Z, best_k,
        title=f'Dendrogram — {name.upper()} (k={best_k}, Ward linkage)',
        output_path=REPORT_DIR / f'{name}_dendrogram.png',
    )

    # ── 3. Silhouette scores (k-selection rationale) ──────────────────────────
    plot_silhouette_scores(
        sil_scores, best_k,
        title=f'Silhouette scores — {name.upper()}',
        output_path=REPORT_DIR / f'{name}_silhouette_scores.png',
    )

    # ── 4. 2-D scatter coloured by cluster (convex hulls) ────────────────────
    plot_clusters_with_hulls(
        canvas_2d, labels, best_k,
        title=f'Hierarchical clusters — {name.upper()} (k={best_k})',
        xlabel=xlabel, ylabel=ylabel,
        output_path=REPORT_DIR / f'{name}_clusters_hulls.png',
    )

    # ── 5. Phase overlay on the same 2-D canvas ───────────────────────────────
    plot_phases(
        canvas_2d, meta['Phase'].values,
        title=f'Phase overlay — {name.upper()}',
        xlabel=xlabel, ylabel=ylabel,
        output_path=REPORT_DIR / f'{name}_phases.png',
    )

    # ── 6. Contingency heatmaps ───────────────────────────────────────────────
    for var_name, col in [
        ('Phase',  meta['Phase']),
        ('Cohort', meta['Cohort']),
        ('Round',  meta['Round']),
        ('Role',   meta['Role']),
    ]:
        contingency_heatmap(
            labels, col.values, var_name,
            title=f'{name.upper()} clusters × {var_name}',
            output_path=REPORT_DIR / f'{name}_contingency_{var_name}.png',
        )

    # ── 7. Discriminating biosignal features ──────────────────────────────────
    discriminating_features(
        labels, df_norm, biosig_cols,
        output_dir=REPORT_DIR, prefix=name,
    )

    print(f"  Saved {name} outputs → {REPORT_DIR}")


def main():
    print("Loading normalized biosignals...")
    df_norm, biosig_cols = load_normalized_biosignals()

    for name in ['pca', 'spca', 'umap']:
        print(f"\n=== {name.upper()} ===")
        run_dataset(name, df_norm, biosig_cols)

    print("\nBiosignals dendrogram complete.")


if __name__ == '__main__':
    main()