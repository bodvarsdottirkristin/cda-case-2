"""
Run Agglomerative clustering on an existing phase-level representation.

This script does NOT train a new representation model.

It loads:
    <input-dir>/phase_latents.npy
    <input-dir>/phase_metadata_with_clusters.csv

It applies Agglomerative clustering to the phase-level representation,
replaces the Cluster column, and saves a new experiment folder.

This version is designed for interpretability experiments:
    - use --n-clusters 2 for broad low/high state separation
    - use --n-clusters 3 to match rest / puzzle-stress / recovery
    - use --n-clusters 4 to allow one additional latent state

Example:
    python advanced/v5_agglomerative.py \
        --input-dir advanced/outputs/conv_ae_32_cohort_norm \
        --n-clusters 3

Then evaluate:
    python advanced/utils/evaluate/evaluate_clusters.py \
        --output-dir advanced/outputs/conv_ae_32_cohort_norm_agglomerative_k3 \
        --update-metrics-json

    python advanced/utils/evaluate/check_questionnaire_profiles.py \
        --output-dir advanced/outputs/conv_ae_32_cohort_norm_agglomerative_k3
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
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
# Agglomerative clustering
# ---------------------------------------------------------------------
def run_agglomerative(
    X: np.ndarray,
    n_clusters: int,
    linkage: str,
) -> tuple[np.ndarray, float | None]:
    """
    Run Agglomerative clustering with a fixed number of clusters.

    Agglomerative clustering is hierarchical:
        each sample starts as its own cluster,
        then clusters are progressively merged until n_clusters remain.

    For ward linkage, sklearn automatically uses Euclidean distance.
    """
    if n_clusters < 2:
        raise ValueError("n_clusters must be at least 2.")

    if n_clusters >= len(X):
        raise ValueError(
            f"n_clusters={n_clusters} must be smaller than number of samples={len(X)}."
        )

    model = AgglomerativeClustering(
        n_clusters=n_clusters,
        linkage=linkage,
    )

    labels = model.fit_predict(X)

    if len(np.unique(labels)) > 1:
        sil = float(silhouette_score(X, labels))
    else:
        sil = None

    return labels, sil


# ---------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------
def save_outputs(
    output_dir: Path,
    input_dir: Path,
    X_original: np.ndarray,
    phase_meta: pd.DataFrame,
    n_clusters: int,
    silhouette: float | None,
    args: argparse.Namespace,
) -> None:
    """
    Save Agglomerative clustering outputs.

    Saved:
        phase_latents.npy
        phase_metadata_with_clusters.csv
        agglomerative_model_selection.csv
        metrics.json
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / "phase_latents.npy", X_original.astype(np.float32))

    phase_meta.to_csv(
        output_dir / "phase_metadata_with_clusters.csv",
        index=False,
    )

    selection_df = pd.DataFrame(
        {
            "n_clusters": [int(n_clusters)],
            "silhouette": [silhouette],
            "linkage": [args.linkage],
            "standardize_features": [bool(args.standardize_features)],
        }
    )

    selection_df.to_csv(
        output_dir / "agglomerative_model_selection.csv",
        index=False,
    )

    metrics = {
        "model": "AgglomerativeClustering",
        "input_representation_dir": str(input_dir),
        "clustering_method": "AgglomerativeClustering",
        "linkage": args.linkage,
        "standardize_features": bool(args.standardize_features),
        "n_clusters": int(n_clusters),
        "best_k": int(n_clusters),
        "silhouette": silhouette,
        "selection_metric": "fixed_n_clusters",
        "n_samples": int(X_original.shape[0]),
        "n_features": int(X_original.shape[1]),
        "note": (
            "Agglomerative clustering was fitted on an existing phase-level "
            "representation using a fixed number of clusters. Silhouette is saved "
            "only as a geometric diagnostic, not as the model-selection criterion."
        ),
    }

    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nSaved Agglomerative outputs to: {output_dir}")


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    default_input_dir = ADVANCED_DIR / "outputs" / "conv_ae_32_cohort_norm"

    parser = argparse.ArgumentParser(
        description=(
            "Run fixed-k Agglomerative clustering on an existing phase-level "
            "representation."
        )
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
            "Output folder. If omitted, default is "
            "<input-dir>_agglomerative_k<n_clusters>."
        ),
    )

    parser.add_argument(
        "--n-clusters",
        type=int,
        default=3,
        help=(
            "Fixed number of clusters. Recommended values: "
            "2, 3, or 4. Use 3 to match the three experimental phases."
        ),
    )

    parser.add_argument(
        "--linkage",
        choices=["ward", "complete", "average", "single"],
        default="ward",
        help=(
            "Agglomerative linkage strategy. "
            "Ward is the recommended default for compact clusters."
        ),
    )

    parser.add_argument(
        "--standardize-features",
        action="store_true",
        help=(
            "Standardize phase-level representation before clustering. "
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
            / f"{input_dir.name}_agglomerative_k{args.n_clusters}"
        )

    print(f"Loading representation from: {input_dir}")
    X_original, phase_meta = load_input_representation(input_dir)

    print(
        f"Loaded phase representation: "
        f"{X_original.shape[0]} samples x {X_original.shape[1]} features"
    )

    if args.standardize_features:
        print("Standardizing phase-level features before clustering...")
        scaler = StandardScaler()
        X_for_clustering = scaler.fit_transform(X_original)
    else:
        X_for_clustering = X_original

    print(
        f"Running Agglomerative clustering with "
        f"n_clusters={args.n_clusters}, linkage='{args.linkage}'..."
    )

    labels, silhouette = run_agglomerative(
        X=X_for_clustering,
        n_clusters=args.n_clusters,
        linkage=args.linkage,
    )

    phase_meta["Cluster"] = labels

    print(f"\nCreated {args.n_clusters} clusters.")

    if silhouette is not None:
        print(f"Silhouette diagnostic: {silhouette:.4f}")
    else:
        print("Silhouette diagnostic could not be computed.")

    save_outputs(
        output_dir=args.output_dir,
        input_dir=input_dir,
        X_original=X_original,
        phase_meta=phase_meta,
        n_clusters=args.n_clusters,
        silhouette=silhouette,
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