"""
Run Gaussian Mixture Model clustering on an existing phase-level representation.

This script does NOT train a new representation model.

It loads:
    <input-dir>/phase_latents.npy
    <input-dir>/phase_metadata_with_clusters.csv

It applies GMM clustering to the phase-level representation, replaces the
Cluster column, and saves a new experiment folder.

By default, the number of GMM components is selected using BIC.

Example:
    python advanced/v4_gmm.py \
        --input-dir advanced/outputs/conv_ae_32_cohort_norm

Then evaluate:
    python advanced/utils/evaluate/evaluate_clusters.py \
        --output-dir advanced/outputs/conv_ae_32_cohort_norm_gmm_bic \
        --update-metrics-json

    python advanced/utils/evaluate/check_questionnaire_profiles.py \
        --output-dir advanced/outputs/conv_ae_32_cohort_norm_gmm_bic
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
def infer_advanced_dir() -> Path:
    here = Path(__file__).resolve()

    for parent in here.parents:
        if parent.name == "advanced":
            return parent

    cwd = Path.cwd()
    if (cwd / "advanced").exists():
        return cwd / "advanced"

    return here.parent


ADVANCED_DIR = infer_advanced_dir()


# ---------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------
def load_input_representation(input_dir: Path) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Load phase-level representation and metadata.

    Expected files:
        phase_latents.npy
        phase_metadata_with_clusters.csv
    """
    phase_latents_path = input_dir / "phase_latents.npy"
    phase_meta_path = input_dir / "phase_metadata_with_clusters.csv"

    if not phase_latents_path.exists():
        raise FileNotFoundError(f"Could not find {phase_latents_path}")

    if not phase_meta_path.exists():
        raise FileNotFoundError(f"Could not find {phase_meta_path}")

    X = np.load(phase_latents_path)
    phase_meta = pd.read_csv(phase_meta_path)

    if len(X) != len(phase_meta):
        raise ValueError(
            "phase_latents.npy and phase_metadata_with_clusters.csv "
            "must contain the same number of samples."
        )

    return X.astype(np.float32), phase_meta


# ---------------------------------------------------------------------
# GMM clustering
# ---------------------------------------------------------------------
def run_gmm_sweep(
    X: np.ndarray,
    k_min: int,
    k_max: int,
    covariance_type: str,
    random_state: int,
    selection_metric: str,
) -> tuple[pd.DataFrame, int, np.ndarray]:
    """
    Fit GMM models for k_min...k_max.

    For each k, this computes:
        - silhouette score
        - BIC
        - AIC

    Model selection:
        - bic:        choose k with lowest BIC
        - aic:        choose k with lowest AIC
        - silhouette: choose k with highest silhouette

    For this project, BIC is the recommended default because GMM is a
    probabilistic model and BIC penalizes unnecessary complexity.
    """
    n_samples = len(X)

    if n_samples < 3:
        raise ValueError("Need at least 3 phase-level samples for clustering.")

    k_max = min(k_max, n_samples - 1)

    if k_min > k_max:
        raise ValueError(f"Invalid k range: k_min={k_min}, k_max={k_max}")

    rows: list[dict[str, float | int | bool]] = []
    fitted_models: dict[int, GaussianMixture] = {}

    for k in range(k_min, k_max + 1):
        gmm = GaussianMixture(
            n_components=k,
            covariance_type=covariance_type,
            random_state=random_state,
            n_init=10,
        )

        labels = gmm.fit_predict(X)

        if len(np.unique(labels)) > 1:
            silhouette = float(silhouette_score(X, labels))
        else:
            silhouette = np.nan

        bic = float(gmm.bic(X))
        aic = float(gmm.aic(X))

        rows.append(
            {
                "k": int(k),
                "silhouette": silhouette,
                "bic": bic,
                "aic": aic,
            }
        )

        fitted_models[k] = gmm

    scores_df = pd.DataFrame(rows)

    if selection_metric == "bic":
        best_k = int(scores_df.loc[scores_df["bic"].idxmin(), "k"])
    elif selection_metric == "aic":
        best_k = int(scores_df.loc[scores_df["aic"].idxmin(), "k"])
    elif selection_metric == "silhouette":
        if not scores_df["silhouette"].notna().any():
            raise ValueError("No valid silhouette scores could be computed.")
        best_k = int(scores_df.loc[scores_df["silhouette"].idxmax(), "k"])
    else:
        raise ValueError(f"Unknown selection metric: {selection_metric}")

    best_model = fitted_models[best_k]
    final_labels = best_model.predict(X)

    scores_df["selected"] = scores_df["k"] == best_k

    return scores_df, best_k, final_labels


# ---------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------
def save_outputs(
    output_dir: Path,
    input_dir: Path,
    X_original: np.ndarray,
    X_used_for_clustering: np.ndarray,
    phase_meta: pd.DataFrame,
    scores_df: pd.DataFrame,
    best_k: int,
    args: argparse.Namespace,
) -> None:
    """
    Save GMM clustering outputs.

    We save:
        - phase_latents.npy
        - phase_metadata_with_clusters.csv
        - gmm_model_selection.csv
        - metrics.json

    The phase_latents.npy file stores the original input representation,
    not necessarily the standardized version used for clustering.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / "phase_latents.npy", X_original.astype(np.float32))

    phase_meta.to_csv(
        output_dir / "phase_metadata_with_clusters.csv",
        index=False,
    )

    scores_df.to_csv(
        output_dir / "gmm_model_selection.csv",
        index=False,
    )

    best_row = scores_df.loc[scores_df["k"] == best_k].iloc[0]

    metrics = {
        "model": "GMM",
        "input_representation_dir": str(input_dir),
        "clustering_method": "GaussianMixture",
        "covariance_type": args.covariance_type,
        "standardize_features": bool(args.standardize_features),
        "selection_metric": args.selection_metric,
        "k_min": int(args.k_min),
        "k_max": int(args.k_max),
        "best_k": int(best_k),
        "best_silhouette": None
        if pd.isna(best_row["silhouette"])
        else float(best_row["silhouette"]),
        "best_bic": float(best_row["bic"]),
        "best_aic": float(best_row["aic"]),
        "n_samples": int(X_original.shape[0]),
        "n_features": int(X_original.shape[1]),
        "note": (
            "GMM was fitted on an existing phase-level representation. "
            "The number of components was selected using the chosen "
            "selection_metric. Evaluation should be run with evaluate_clusters.py."
        ),
    }

    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nSaved GMM outputs to: {output_dir}")


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    default_input_dir = ADVANCED_DIR / "outputs" / "conv_ae_32_cohort_norm"

    parser = argparse.ArgumentParser(
        description="Run GMM clustering on an existing phase-level representation."
    )

    parser.add_argument(
        "--input-dir",
        type=Path,
        default=default_input_dir,
        help=(
            "Folder containing phase_latents.npy and "
            "phase_metadata_with_clusters.csv."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Output folder. If omitted, the name is generated from input-dir "
            "and selection metric."
        ),
    )

    parser.add_argument("--k-min", type=int, default=2)
    parser.add_argument("--k-max", type=int, default=8)
    parser.add_argument("--random-state", type=int, default=42)

    parser.add_argument(
        "--covariance-type",
        choices=["full", "tied", "diag", "spherical"],
        default="full",
        help="GMM covariance type.",
    )

    parser.add_argument(
        "--selection-metric",
        choices=["bic", "aic", "silhouette"],
        default="bic",
        help=(
            "Metric used to select the number of GMM components. "
            "Recommended default: bic."
        ),
    )

    parser.add_argument(
        "--standardize-features",
        action="store_true",
        help=(
            "Standardize phase-level representation before GMM clustering. "
            "The original unstandardized phase_latents.npy is still saved."
        ),
    )

    return parser.parse_args()


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main() -> None:
    args = parse_args()

    input_dir = args.input_dir

    if args.output_dir is None:
        args.output_dir = (
            input_dir.parent
            / f"{input_dir.name}_gmm_{args.selection_metric}"
        )

    print(f"Loading representation from: {input_dir}")
    X_original, phase_meta = load_input_representation(input_dir)

    print(
        f"Loaded phase representation: "
        f"{X_original.shape[0]} samples x {X_original.shape[1]} features"
    )

    if args.standardize_features:
        print("Standardizing phase-level features before GMM...")
        scaler = StandardScaler()
        X_for_clustering = scaler.fit_transform(X_original)
    else:
        X_for_clustering = X_original

    print(
        f"Running GMM sweep from k={args.k_min} to k={args.k_max} "
        f"using covariance_type='{args.covariance_type}'..."
    )

    scores_df, best_k, labels = run_gmm_sweep(
        X=X_for_clustering,
        k_min=args.k_min,
        k_max=args.k_max,
        covariance_type=args.covariance_type,
        random_state=args.random_state,
        selection_metric=args.selection_metric,
    )

    phase_meta["Cluster"] = labels

    print("\nGMM model selection results:")
    print(scores_df.to_string(index=False))

    print(
        f"\nSelected k={best_k} using {args.selection_metric.upper()}."
    )

    save_outputs(
        output_dir=args.output_dir,
        input_dir=input_dir,
        X_original=X_original,
        X_used_for_clustering=X_for_clustering,
        phase_meta=phase_meta,
        scores_df=scores_df,
        best_k=best_k,
        args=args,
    )

    print("\nNext, evaluate with:")
    print(
        f"python advanced/utils/evaluate/evaluate_clusters.py "
        f"--output-dir {args.output_dir} --update-metrics-json"
    )
    print(
        f"python advanced/utils/evaluate/check_questionnaire_profiles.py "
        f"--output-dir {args.output_dir}"
    )


if __name__ == "__main__":
    main()