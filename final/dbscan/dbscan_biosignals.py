"""
dbscan_biosignals.py
=====================
Runs DBSCAN on all three reductions (PCA, SparsePCA, UMAP).
Grid-searches eps and min_samples via silhouette score on non-noise points.
Saves cluster assignments, evaluation metrics, and plots to final/dbscan/report/.
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR     = PROJECT_ROOT / "data/processed/final"
REPORT_DIR   = Path(__file__).parent / "report"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

META  = ["Round", "Phase", "Individual", "Puzzler", "Cohort",
         "Frustrated", "upset", "hostile", "alert", "ashamed", "inspired",
         "nervous", "attentive", "afraid", "active", "determined"]
LABELS_COLS = ["Phase", "Cohort", "Round", "Puzzler"]
REDUCTIONS  = [("pca", "PCA"), ("spca", "SparsePCA"), ("umap", "UMAP")]

EPS_GRID      = [0.1, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0]
MIN_SAMP_GRID = [3, 5, 10]

summary_rows = []

for red_name, red_label in REDUCTIONS:
    print(f"\n{'='*50}")
    print(f"  {red_label}")
    print(f"{'='*50}")

    df = pd.read_csv(DATA_DIR / f"HR_data_{red_name}.csv")
    feat_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                 if c not in META]
    X = StandardScaler().fit_transform(
        df[feat_cols].fillna(df[feat_cols].median()).values
    )

    # ── grid search ──────────────────────────────────────────────────────────
    best_score  = -1
    best_params = (1.0, 5)
    best_labels = np.full(len(X), -1)

    for eps in EPS_GRID:
        for min_s in MIN_SAMP_GRID:
            lbl = DBSCAN(eps=eps, min_samples=min_s).fit_predict(X)
            n_cl    = len(set(lbl)) - (1 if -1 in lbl else 0)
            n_noise = (lbl == -1).sum()
            mask    = lbl != -1
            if n_cl >= 2 and mask.sum() > 1:
                score = silhouette_score(X[mask], lbl[mask])
            else:
                score = -1
            print(f"  eps={eps:<5} min_s={min_s:<3} "
                  f"clusters={n_cl}  noise={n_noise} ({n_noise/len(lbl)*100:.0f}%)  "
                  f"sil={score:.3f}")
            if score > best_score:
                best_score  = score
                best_params = (eps, min_s)
                best_labels = lbl

    eps_best, min_s_best = best_params
    labels   = best_labels
    n_cl     = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise  = (labels == -1).sum()
    print(f"\n  Best: eps={eps_best}, min_samples={min_s_best}  "
          f"clusters={n_cl}  noise={n_noise}  silhouette={best_score:.3f}")

    # ── evaluation ───────────────────────────────────────────────────────────
    valid = labels >= 0
    row   = {"Reduction": red_label, "eps": eps_best, "min_samples": min_s_best,
             "n_clusters": n_cl, "n_noise": n_noise,
             "noise_pct": round(n_noise / len(labels) * 100, 1),
             "silhouette": round(best_score, 4)}

    print(f"\n  Alignment metrics (non-noise points only: {valid.sum()}):")
    for col in LABELS_COLS:
        if col in df.columns:
            if valid.sum() > 1:
                ari = adjusted_rand_score(df[col].astype(str)[valid], labels[valid])
                nmi = normalized_mutual_info_score(df[col].astype(str)[valid], labels[valid])
            else:
                ari, nmi = float("nan"), float("nan")
            row[f"{col}_ARI"] = round(ari, 4)
            row[f"{col}_NMI"] = round(nmi, 4)
            print(f"    {col:10} ARI={ari:.4f}  NMI={nmi:.4f}")

    summary_rows.append(row)

    # ── save cluster assignments ──────────────────────────────────────────────
    df_out = df.copy()
    df_out["DBSCAN_cluster"] = labels
    df_out.to_csv(REPORT_DIR / f"{red_name}_assignments.csv", index=False)

    # ── k-distance plot ───────────────────────────────────────────────────────
    k     = 5
    dists = np.sort(NearestNeighbors(n_neighbors=k).fit(X).kneighbors(X)[0][:, k-1])[::-1]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(dists)
    ax.axhline(eps_best, color="red", linestyle="--", label=f"best eps={eps_best}")
    ax.set_xlabel("Points sorted by distance")
    ax.set_ylabel(f"{k}-NN distance")
    ax.set_title(f"k-distance plot — {red_label}")
    ax.legend()
    plt.tight_layout()
    plt.savefig(REPORT_DIR / f"{red_name}_kdistance.png", dpi=150)
    plt.close()

    # ── cluster scatter ───────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 5))
    sc = ax.scatter(X[:, 0], X[:, 1], c=labels, cmap="tab10", s=15, alpha=0.7)
    plt.colorbar(sc, ax=ax, label="Cluster (-1 = noise)")
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.set_title(f"DBSCAN — {red_label}  (eps={eps_best}, min_s={min_s_best})")
    plt.tight_layout()
    plt.savefig(REPORT_DIR / f"{red_name}_clusters.png", dpi=150)
    plt.close()

# ── summary CSV ───────────────────────────────────────────────────────────────
summary = pd.DataFrame(summary_rows)
summary.to_csv(REPORT_DIR / "dbscan_summary.csv", index=False)
print("\n\n=== SUMMARY ===")
print(summary.to_string(index=False))
print(f"\nSaved to {REPORT_DIR}")
