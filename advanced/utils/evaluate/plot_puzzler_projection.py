"""
Plot top-2D latent representation colored by Cluster and Puzzler.

This script creates a side-by-side figure:
    1. Top-2D representation colored by cluster, with empirical ellipses.
    2. Top-2D representation colored by Puzzler role.

Inputs:
    <output-dir>/phase_latents.npy
    <output-dir>/phase_metadata_with_clusters.csv

Example:
    python advanced/utils/evaluate/plot_puzzler_projection.py \
        --output-dir advanced/outputs/new/LSTM/lstm_ae_16_cohort_round_norm
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Ellipse
from sklearn.decomposition import PCA


def get_top2_projection(X: np.ndarray) -> np.ndarray:
    """
    Return a 2D representation for visualization.

    If X already has 2 dimensions, use it directly.
    Otherwise, use PCA to project to 2 dimensions.
    """
    if X.shape[1] == 2:
        return X

    return PCA(n_components=2, random_state=42).fit_transform(X)


def draw_empirical_ellipses(
    ax,
    X: np.ndarray,
    labels: np.ndarray,
    palette: str = "viridis",
    alpha: float = 0.2,
) -> None:
    """
    Draw empirical covariance ellipses for each cluster.

    These ellipses are estimated from the assigned cluster points.
    They are only for visualization and do not change the clustering.
    """
    unique_labels = np.unique(labels)
    colors = sns.color_palette(palette, n_colors=len(unique_labels))

    for idx, cluster in enumerate(unique_labels):
        X_cluster = X[labels == cluster]

        if len(X_cluster) < 2:
            continue

        mean = np.mean(X_cluster, axis=0)
        cov = np.cov(X_cluster, rowvar=False)

        if cov.shape != (2, 2):
            continue

        vals, vecs = np.linalg.eigh(cov)

        order = vals.argsort()[::-1]
        vals = vals[order]
        vecs = vecs[:, order]

        angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))

        width, height = 2.0 * np.sqrt(2.0) * np.sqrt(vals)

        ellipse = Ellipse(
            xy=mean,
            width=width,
            height=height,
            angle=angle,
            color=colors[idx],
            alpha=alpha,
        )

        ax.add_patch(ellipse)


def plot_top2_cluster_and_puzzler(
    Z2: np.ndarray,
    labels: np.ndarray,
    df: pd.DataFrame,
    out_path: Path,
    cluster_palette: str = "viridis",
    puzzler_palette: str = "Set1",
) -> None:
    """
    Save a figure with two aligned subplots:
        1. Top-2D representation colored by Cluster.
        2. Top-2D representation colored by Puzzler.
    """
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(16, 6),
        sharex=True,
        sharey=True,
    )

    ax_cluster, ax_puzzler = axes

    # ------------------------------------------------------------
    # Left subplot: clusters with empirical ellipses
    # ------------------------------------------------------------
    unique_clusters = np.unique(labels)
    cluster_colors = sns.color_palette(
        cluster_palette,
        n_colors=len(unique_clusters),
    )

    cluster_to_color = dict(zip(unique_clusters, cluster_colors))

    for cluster in unique_clusters:
        mask = labels == cluster

        ax_cluster.scatter(
            Z2[mask, 0],
            Z2[mask, 1],
            s=35,
            alpha=0.8,
            color=cluster_to_color[cluster],
            label=f"Cluster {cluster}",
        )

    draw_empirical_ellipses(
        ax=ax_cluster,
        X=Z2,
        labels=labels,
        palette=cluster_palette,
        alpha=0.2,
    )

    ax_cluster.set_title("Top 2 dimensions colored by Cluster")
    ax_cluster.set_xlabel("Dimension 1")
    ax_cluster.set_ylabel("Dimension 2")
    ax_cluster.legend(title="Cluster")

    # ------------------------------------------------------------
    # Right subplot: Puzzler role
    # ------------------------------------------------------------
    if "Puzzler" not in df.columns:
        ax_puzzler.scatter(
            Z2[:, 0],
            Z2[:, 1],
            s=35,
            alpha=0.8,
        )
        ax_puzzler.set_title("Top 2 dimensions (Puzzler not available)")

    else:
        puzzler_values = df["Puzzler"].copy()
        puzzler_values = puzzler_values.astype(str).str.strip()

        unique_roles = sorted(puzzler_values.dropna().unique())
        role_colors = sns.color_palette(
            puzzler_palette,
            n_colors=len(unique_roles),
        )

        role_to_color = dict(zip(unique_roles, role_colors))

        for role in unique_roles:
            mask = puzzler_values == role

            ax_puzzler.scatter(
                Z2[mask, 0],
                Z2[mask, 1],
                s=35,
                alpha=0.8,
                color=role_to_color[role],
                label=str(role),
            )

        ax_puzzler.legend(title="Puzzler")
        ax_puzzler.set_title("Top 2 dimensions colored by Puzzler")

    ax_puzzler.set_xlabel("Dimension 1")
    ax_puzzler.set_ylabel("Dimension 2")

    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot top-2D representation colored by Cluster and Puzzler."
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Experiment output folder containing phase_latents.npy and metadata.",
    )

    parser.add_argument(
        "--eval-subdir",
        type=str,
        default="eval/final_biosignal_report",
        help="Subfolder where the plot will be saved.",
    )

    parser.add_argument(
        "--filename",
        type=str,
        default="top2_cluster_and_puzzler.png",
        help="Output figure filename.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    output_dir = args.output_dir
    eval_dir = output_dir / args.eval_subdir
    eval_dir.mkdir(parents=True, exist_ok=True)

    latent_path = output_dir / "phase_latents.npy"
    meta_path = output_dir / "phase_metadata_with_clusters.csv"

    if not latent_path.exists():
        raise FileNotFoundError(f"Missing {latent_path}")

    if not meta_path.exists():
        raise FileNotFoundError(f"Missing {meta_path}")

    X = np.load(latent_path)
    df = pd.read_csv(meta_path)

    if "Cluster" not in df.columns:
        raise ValueError("phase_metadata_with_clusters.csv must contain Cluster.")

    if len(X) != len(df):
        raise ValueError("phase_latents.npy and metadata must have the same rows.")

    labels = df["Cluster"].to_numpy()
    Z2 = get_top2_projection(X)

    out_path = eval_dir / args.filename

    plot_top2_cluster_and_puzzler(
        Z2=Z2,
        labels=labels,
        df=df,
        out_path=out_path,
    )

    print(f"Saved Puzzler projection plot to: {out_path}")


if __name__ == "__main__":
    main()