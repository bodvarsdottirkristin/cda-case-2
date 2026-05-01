"""
PCA baseline for latent-state discovery.

This script uses the same processed window data as the Conv1D autoencoder,
but replaces the neural encoder with PCA.

Pipeline:

    data/processed/autoencoder/autoencoder_windows.npz
    -> flatten each 60 x 3 window
    -> PCA
    -> average PCA scores per phase
    -> K-Means clustering
    -> save results
    -> evaluate with evaluate_clusters.py

Example:

    python advanced/v3_PCA.py --n-components 32

Then evaluate:

    python advanced/utils/evaluate/evaluate_clusters.py \
        --output-dir advanced/outputs/pca_32
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Paths and local imports
# ---------------------------------------------------------------------
ADVANCED_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ADVANCED_DIR.parent

sys.path.insert(0, str(ADVANCED_DIR))

from utils.PCA import (  # noqa: E402
    load_processed_autoencoder_file,
    flatten_windows,
    fit_window_pca,
    aggregate_window_features_per_phase,
    run_kmeans_sweep,
    explained_variance_dataframe,
)


# ---------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------
def plot_pca_summary(
    explained_variance_df: pd.DataFrame,
    silhouette_scores: dict[int, float],
    best_k: int,
    figures_dir: Path,
) -> None:
    """
    Save PCA explained variance and K-Means silhouette diagnostic figure.
    """
    figures_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(
        explained_variance_df["component"],
        explained_variance_df["cumulative_explained_variance_ratio"],
        marker="o",
    )
    axes[0].set_xlabel("PCA component")
    axes[0].set_ylabel("Cumulative explained variance ratio")
    axes[0].set_title("PCA Explained Variance")

    axes[1].bar(list(silhouette_scores.keys()), list(silhouette_scores.values()))
    axes[1].axvline(best_k, linestyle="--", label=f"Best k={best_k}")
    axes[1].set_xlabel("k")
    axes[1].set_ylabel("Silhouette Score")
    axes[1].set_title("K-Means Silhouette Scores")
    axes[1].legend()

    plt.tight_layout()

    out = figures_dir / "pca_summary.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)

    print(f"Figure saved -> {out}")


def plot_pca_2d(
    phase_features: np.ndarray,
    phase_meta: pd.DataFrame,
    figures_dir: Path,
) -> None:
    """
    Save simple PC1/PC2 plots colored by cluster and cohort.

    This is only available if n_components >= 2.
    """
    if phase_features.shape[1] < 2:
        return

    figures_dir.mkdir(parents=True, exist_ok=True)

    # Colored by cluster
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(
        phase_features[:, 0],
        phase_features[:, 1],
        c=phase_meta["Cluster"],
        cmap="tab10",
        s=40,
        alpha=0.8,
    )
    plt.colorbar(scatter, ax=ax, label="Cluster")
    ax.set_title("Phase-level PCA scores colored by cluster")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    plt.tight_layout()

    out = figures_dir / "pca_clusters.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Figure saved -> {out}")

    # Colored by cohort
    if "Cohort" in phase_meta.columns:
        fig, ax = plt.subplots(figsize=(8, 6))
        cohorts = sorted(phase_meta["Cohort"].dropna().unique())
        cmap = plt.get_cmap("tab10")

        for i, cohort in enumerate(cohorts):
            mask = phase_meta["Cohort"] == cohort
            ax.scatter(
                phase_features[mask, 0],
                phase_features[mask, 1],
                label=cohort,
                color=cmap(i % 10),
                s=40,
                alpha=0.8,
            )

        ax.legend(title="Cohort")
        ax.set_title("Phase-level PCA scores colored by cohort")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        plt.tight_layout()

        out = figures_dir / "pca_cohort.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"Figure saved -> {out}")


# ---------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------
def save_outputs(
    output_dir: Path,
    phase_features: np.ndarray,
    phase_meta: pd.DataFrame,
    explained_variance_df: pd.DataFrame,
    silhouette_scores: dict[int, float],
    best_k: int,
    args: argparse.Namespace,
) -> None:
    """
    Save useful PCA baseline outputs.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Keep same filename as autoencoder so evaluation/comparison is easy.
    np.save(output_dir / "phase_latents.npy", phase_features.astype(np.float32))

    phase_meta.to_csv(output_dir / "phase_metadata_with_clusters.csv", index=False)

    explained_variance_df.to_csv(
        output_dir / "pca_explained_variance.csv",
        index=False,
    )

    pd.DataFrame(
        {
            "k": list(silhouette_scores.keys()),
            "silhouette": list(silhouette_scores.values()),
        }
    ).to_csv(output_dir / "kmeans_silhouette_scores.csv", index=False)

    metrics = {
        "model": "PCA",
        "n_components": int(args.n_components),
        "representation": "flattened 60-second HR/EDA/TEMP windows",
        "aggregation": "mean PCA scores per Cohort/participant_ID/Round/Phase",
        "processed_file": str(args.processed_file),
        "output_dir": str(args.output_dir),
        "standardize_flattened_features": bool(args.standardize_flattened_features),
        "k_min": int(args.k_min),
        "k_max": int(args.k_max),
        "best_k": int(best_k),
        "best_silhouette": float(silhouette_scores[best_k]),
        "cumulative_explained_variance": float(
            explained_variance_df["cumulative_explained_variance_ratio"].iloc[-1]
        ),
        "note": "K-Means was fitted on phase-level PCA scores.",
    }

    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"Outputs saved to: {output_dir}")


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    default_processed_file = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "autoencoder"
        / "autoencoder_windows.npz"
    )

    parser = argparse.ArgumentParser(
        description="Run PCA baseline on processed autoencoder windows."
    )

    parser.add_argument(
        "--processed-file",
        type=Path,
        default=default_processed_file,
        help="Processed autoencoder window file.",
    )

    parser.add_argument(
        "--n-components",
        type=int,
        default=32,
        help="Number of PCA components.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Default: advanced/outputs/pca_<n_components>.",
    )

    parser.add_argument("--k-min", type=int, default=2)
    parser.add_argument("--k-max", type=int, default=8)
    parser.add_argument("--random-state", type=int, default=42)

    parser.add_argument(
        "--standardize-flattened-features",
        action="store_true",
        help=(
            "Standardize each flattened feature before PCA. "
            "Usually not needed because X_scaled is already standardized by signal."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.output_dir is None:
        args.output_dir = ADVANCED_DIR / "outputs" / f"pca_{args.n_components}"

    print(f"Loading processed data from: {args.processed_file}")

    X_scaled, window_meta = load_processed_autoencoder_file(args.processed_file)

    if "participant_ID" not in window_meta.columns:
        raise ValueError(
            "Expected column 'participant_ID' in window metadata, but it was not found. "
            f"Available columns: {list(window_meta.columns)}"
        )

    print(
        f"Loaded windows: {X_scaled.shape[0]} windows x "
        f"{X_scaled.shape[1]} seconds x {X_scaled.shape[2]} signals"
    )

    X_flat = flatten_windows(X_scaled)

    print(f"Flattened window matrix: {X_flat.shape}")

    window_pca_scores, pca, _ = fit_window_pca(
        X_flat=X_flat,
        n_components=args.n_components,
        standardize_flattened_features=args.standardize_flattened_features,
        random_state=args.random_state,
    )

    explained_variance_df = explained_variance_dataframe(pca)

    cumulative_ev = explained_variance_df[
        "cumulative_explained_variance_ratio"
    ].iloc[-1]

    print(
        f"PCA window scores: {window_pca_scores.shape}, "
        f"cumulative explained variance={cumulative_ev:.3f}"
    )

    phase_features, phase_meta = aggregate_window_features_per_phase(
        window_features=window_pca_scores,
        window_meta=window_meta,
    )

    print(f"Phase-level PCA matrix: {phase_features.shape}")

    silhouette_scores, best_k, labels = run_kmeans_sweep(
        phase_features=phase_features,
        k_min=args.k_min,
        k_max=args.k_max,
        random_state=args.random_state,
    )

    phase_meta["Cluster"] = labels

    print(f"Silhouette scores: { {k: f'{v:.3f}' for k, v in silhouette_scores.items()} }")
    print(f"Best k: {best_k}")

    save_outputs(
        output_dir=args.output_dir,
        phase_features=phase_features,
        phase_meta=phase_meta,
        explained_variance_df=explained_variance_df,
        silhouette_scores=silhouette_scores,
        best_k=best_k,
        args=args,
    )

    figures_dir = args.output_dir / "figures"

    plot_pca_summary(
        explained_variance_df=explained_variance_df,
        silhouette_scores=silhouette_scores,
        best_k=best_k,
        figures_dir=figures_dir,
    )

    plot_pca_2d(
        phase_features=phase_features,
        phase_meta=phase_meta,
        figures_dir=figures_dir,
    )

    print("\nDone.")
    print(f"Output folder: {args.output_dir}")


if __name__ == "__main__":
    main()