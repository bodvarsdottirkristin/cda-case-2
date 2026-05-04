"""
Cluster questionnaire variables only using fixed k = 2.

Purpose:
    Check whether questionnaire data itself naturally splits into two groups.
    This helps understand whether emotional structure is even present.

Input:
    <input-dir>/phase_metadata_with_clusters.csv

Output:
    advanced/outputs/questionnaire_clustering_k2/
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cluster questionnaire variables only using fixed k=2."
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
        default=ADVANCED_DIR / "outputs" / "questionnaire_clustering_k2",
        help="Output folder.",
    )

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

    k = 2

    model = KMeans(
        n_clusters=k,
        random_state=args.random_state,
        n_init=10,
    )

    labels = model.fit_predict(X_scaled)

    silhouette = (
        float(silhouette_score(X_scaled, labels))
        if len(np.unique(labels)) > 1
        else None
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

    metrics = {
        "model": "QuestionnaireKMeans",
        "input_dir": str(args.input_dir),
        "questionnaire_columns": questionnaire_cols,
        "n_clusters": k,
        "selection": "fixed_k",
        "silhouette": silhouette,
        "note": (
            "Clusters were fitted using questionnaire variables only with fixed k=2."
        ),
    }

    with open(args.output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nUsed fixed k: {k}")
    if silhouette is not None:
        print(f"Silhouette: {silhouette:.4f}")
    print(f"Saved to: {args.output_dir}")


if __name__ == "__main__":
    main()