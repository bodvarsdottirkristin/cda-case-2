"""
Cluster questionnaire variables only.

Purpose:
    Sanity check whether self-reported questionnaire answers themselves
    separate the experimental phases.

If questionnaire-only clusters do not align with Phase, then expecting
physiology-only clusters to recover Phase may be unrealistic.

Input:
    <input-dir>/phase_metadata_with_clusters.csv

Output:
    advanced/outputs/questionnaire_clustering/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


ADVANCED_DIR = Path(__file__).resolve().parent


EXCLUDED_COLS = {
    "Cohort",
    "Individual",
    "Round",
    "Phase",
    "Cluster",
    "Puzzler",
    "puzzler",
    "parent",
    "Parent",
    "WindowStartIndex",
    "WindowEndIndex",
    "WindowStartTime",
    "WindowEndTime",
}


def get_questionnaire_columns(df: pd.DataFrame) -> list[str]:
    numeric_cols = df.select_dtypes(include="number").columns
    return [col for col in numeric_cols if col not in EXCLUDED_COLS]


def run_kmeans_sweep(
    X: np.ndarray,
    k_min: int,
    k_max: int,
    random_state: int,
) -> tuple[pd.DataFrame, int, np.ndarray]:
    rows = []
    labels_by_k = {}

    for k in range(k_min, k_max + 1):
        model = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = model.fit_predict(X)

        sil = silhouette_score(X, labels) if len(np.unique(labels)) > 1 else np.nan

        rows.append({"k": k, "silhouette": sil})
        labels_by_k[k] = labels

    scores = pd.DataFrame(rows)
    best_k = int(scores.loc[scores["silhouette"].idxmax(), "k"])
    scores["selected"] = scores["k"] == best_k

    return scores, best_k, labels_by_k[best_k]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cluster questionnaire variables only."
    )

    parser.add_argument(
        "--input-dir",
        type=Path,
        default=ADVANCED_DIR / "outputs" / "conv_ae_32_cohort_norm",
        help="Folder containing phase_metadata_with_clusters.csv.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ADVANCED_DIR / "outputs" / "questionnaire_clustering",
        help="Output folder.",
    )

    parser.add_argument("--k-min", type=int, default=2)
    parser.add_argument("--k-max", type=int, default=6)
    parser.add_argument("--random-state", type=int, default=42)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_path = args.input_dir / "phase_metadata_with_clusters.csv"

    if not input_path.exists():
        raise FileNotFoundError(f"Could not find {input_path}")

    df = pd.read_csv(input_path)

    questionnaire_cols = get_questionnaire_columns(df)

    if not questionnaire_cols:
        raise ValueError("No questionnaire-like numeric columns found.")

    print("Questionnaire columns used:")
    for col in questionnaire_cols:
        print(f"- {col}")

    X = df[questionnaire_cols].copy()

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    X_imputed = imputer.fit_transform(X)
    X_scaled = scaler.fit_transform(X_imputed)

    scores, best_k, labels = run_kmeans_sweep(
        X=X_scaled,
        k_min=args.k_min,
        k_max=args.k_max,
        random_state=args.random_state,
    )

    out_df = df.copy()
    out_df["Cluster"] = labels

    args.output_dir.mkdir(parents=True, exist_ok=True)

    out_df.to_csv(
        args.output_dir / "phase_metadata_with_clusters.csv",
        index=False,
    )

    np.save(
        args.output_dir / "phase_latents.npy",
        X_scaled.astype(np.float32),
    )

    scores.to_csv(
        args.output_dir / "questionnaire_kmeans_scores.csv",
        index=False,
    )

    metrics = {
        "model": "QuestionnaireKMeans",
        "input_dir": str(args.input_dir),
        "questionnaire_columns": questionnaire_cols,
        "k_min": args.k_min,
        "k_max": args.k_max,
        "best_k": best_k,
        "best_silhouette": float(
            scores.loc[scores["k"] == best_k, "silhouette"].iloc[0]
        ),
        "note": (
            "Clusters were fitted using questionnaire variables only. "
            "This is a sanity check to see whether self-reported emotions "
            "separate experimental phases."
        ),
    }

    with open(args.output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nBest k: {best_k}")
    print(f"Saved to: {args.output_dir}")


if __name__ == "__main__":
    main()