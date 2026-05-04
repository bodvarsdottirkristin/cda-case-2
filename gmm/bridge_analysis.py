import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

FIGURES_DIR = Path(__file__).resolve().parent / 'figures'
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR = PROJECT_ROOT / 'data' / 'processed'
RESULTS_DIR = PROJECT_ROOT / 'results' / 'tables'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MERGE_COLS = ['Round', 'Phase', 'Individual']


def load_and_merge():
    biosig = pd.read_csv(PROCESSED_DIR / 'HR_data_gmm_umap.csv')
    panas = pd.read_csv(PROCESSED_DIR / 'HR_data_panas.csv')
    questionnaire = pd.read_csv(PROCESSED_DIR / 'HR_data_questionnaire_clusters.csv')

    df = (
        biosig[MERGE_COLS + ['cluster', 'Puzzler']]
        .rename(columns={'cluster': 'biosig_cluster'})
        .merge(
            panas[MERGE_COLS + ['emotional_cluster']]
            .rename(columns={'emotional_cluster': 'panas_cluster'}),
            on=MERGE_COLS, how='inner'
        )
        .merge(
            questionnaire[MERGE_COLS + ['q_cluster']],
            on=MERGE_COLS, how='inner'
        )
    )

    print(f"Merged: {len(df)} rows "
          f"(biosig={len(biosig)}, panas={len(panas)}, q={len(questionnaire)})")
    return df


def compute_alignment(df):
    pairs = [
        ('biosig_cluster', 'panas_cluster', 'Biosignal vs PANAS (PA/NA composite)'),
        ('biosig_cluster', 'q_cluster',     'Biosignal vs Questionnaire (11-item)'),
        ('biosig_cluster', 'Phase',          'Biosignal vs Experimental Phase'),
        ('biosig_cluster', 'Puzzler',        'Biosignal vs Role (Puzzler/Instructor)'),
    ]

    records = []
    for col_a, col_b, label in pairs:
        a = df[col_a].astype(str).values
        b = df[col_b].astype(str).values
        ari = adjusted_rand_score(a, b)
        nmi = normalized_mutual_info_score(a, b)
        records.append({'comparison': label, 'ARI': round(ari, 4), 'NMI': round(nmi, 4)})
        print(f"{label}:  ARI={ari:.4f}  NMI={nmi:.4f}")

    summary = pd.DataFrame(records)
    summary.to_csv(RESULTS_DIR / 'bridge_summary.csv', index=False)
    return summary


def plot_confusion(df, col_a, col_b, fname, title):
    ct = pd.crosstab(df[col_a], df[col_b], normalize='index')
    fig_w = max(6, len(ct.columns) * 0.9)
    fig_h = max(4, len(ct) * 0.6 + 1.5)
    plt.figure(figsize=(fig_w, fig_h))
    sns.heatmap(ct, annot=True, fmt='.2f', cmap='Blues', vmin=0, vmax=1,
                linewidths=0.3)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / fname, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {fname}")


def main():
    df = load_and_merge()

    print("\n--- Alignment Metrics ---")
    summary = compute_alignment(df)
    print()
    print(summary.to_string(index=False))

    plot_confusion(
        df, 'biosig_cluster', 'panas_cluster',
        'bridge_biosig_vs_panas.png',
        'Biosignal cluster vs PANAS emotional cluster\n(row-normalised: each row sums to 1)'
    )
    plot_confusion(
        df, 'biosig_cluster', 'q_cluster',
        'bridge_biosig_vs_questionnaire.png',
        'Biosignal cluster vs Questionnaire cluster (11-item)\n(row-normalised)'
    )
    plot_confusion(
        df, 'biosig_cluster', 'Phase',
        'bridge_biosig_vs_phase.png',
        'Biosignal cluster vs Experimental Phase\n(row-normalised)'
    )
    df_role = df.copy()
    df_role['Role'] = df_role['Puzzler'].map({1: 'Active', 0: 'Instructor'})
    plot_confusion(
        df_role, 'biosig_cluster', 'Role',
        'bridge_biosig_vs_role.png',
        'Biosignal cluster vs Role\n(row-normalised)'
    )

    print(f"\nBridge analysis complete.")
    print(f"  Figures: {FIGURES_DIR}")
    print(f"  Tables:  {RESULTS_DIR / 'bridge_summary.csv'}")


if __name__ == '__main__':
    main()
