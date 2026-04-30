"""
Final biosignal cluster report.

Generates:
- top-2D plot with GMM ellipses
- top-2D plot colored by Phase
- contingency tables and heatmaps for Phase, Cohort, Round, Puzzler
- ARI, NMI, chi-square p-value, Cramer's V
- discriminating biosignal features between clusters using Mann-Whitney U + Cohen's d

Input:
    <output-dir>/phase_latents.npy
    <output-dir>/phase_metadata_with_clusters.csv

Optional input:
    --processed-file data/processed/autoencoder/autoencoder_windows.npz

If --processed-file is provided:
    discriminating_features.csv ranks real biosignal features:
        HR_mean, HR_std, HR_min, HR_max, HR_median
        EDA_mean, EDA_std, EDA_min, EDA_max, EDA_median
        TEMP_mean, TEMP_std, TEMP_min, TEMP_max, TEMP_median

If --processed-file is not provided:
    discriminating_features.csv ranks latent/PCA dimensions:
        feature_0, feature_1, ...

Example:
    python advanced/utils/evaluate/final_biosignal_cluster_report.py \
        --output-dir advanced/outputs/new/conv_ae_32 \
        --processed-file data/processed/autoencoder/autoencoder_windows.npz
"""

from __future__ import annotations

import argparse
from io import StringIO
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, mannwhitneyu
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.mixture import GaussianMixture


CATEGORICAL_VARS = ["Phase", "Cohort", "Round", "Puzzler"]
GROUP_KEYS = ["participant_ID", "Cohort", "Round", "Phase"]


# ---------------------------------------------------------------------
# Basic statistics
# ---------------------------------------------------------------------
def cramers_v(table: pd.DataFrame) -> float:
    """
    Compute Cramér's V from a contingency table.

    Interpretation:
        0.00 ≈ no association
        0.10 ≈ weak
        0.30 ≈ moderate
        0.50+ ≈ strong
    """
    chi2, _, _, _ = chi2_contingency(table)

    n = table.to_numpy().sum()
    r, k = table.shape
    denom = n * min(r - 1, k - 1)

    if denom == 0:
        return np.nan

    return float(np.sqrt(chi2 / denom))


def cohen_d(x: np.ndarray, y: np.ndarray) -> float:
    """
    Compute Cohen's d between two groups.

    Positive d means x has higher mean than y.
    """
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]

    if len(x) < 2 or len(y) < 2:
        return np.nan

    nx, ny = len(x), len(y)

    pooled = np.sqrt(
        ((nx - 1) * np.var(x, ddof=1) + (ny - 1) * np.var(y, ddof=1))
        / (nx + ny - 2)
    )

    if pooled == 0:
        return np.nan

    return float((np.mean(x) - np.mean(y)) / pooled)


# ---------------------------------------------------------------------
# Projection and plots
# ---------------------------------------------------------------------
def get_top2_projection(X: np.ndarray) -> np.ndarray:
    """
    Return a 2D representation for visualization.

    If X already has 2 dimensions, use it directly.
    Otherwise, use PCA to project to 2 dimensions.
    """
    if X.shape[1] == 2:
        return X

    return PCA(n_components=2, random_state=42).fit_transform(X)


def plot_contingency_table(
    table: pd.DataFrame,
    var: str,
    out_path: Path,
) -> None:
    """
    Save a contingency table as a heatmap figure.
    """
    fig, ax = plt.subplots(figsize=(8, max(4, 0.45 * len(table))))

    im = ax.imshow(table.values, aspect="auto")

    ax.set_xticks(np.arange(table.shape[1]))
    ax.set_yticks(np.arange(table.shape[0]))

    ax.set_xticklabels(table.columns)
    ax.set_yticklabels(table.index)

    ax.set_xlabel("Cluster")
    ax.set_ylabel(var)
    ax.set_title(f"{var} x Cluster")

    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            ax.text(
                j,
                i,
                str(table.iloc[i, j]),
                ha="center",
                va="center",
            )

    plt.colorbar(im, ax=ax, label="Count")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_gmm_ellipses(
    Z2: np.ndarray,
    labels: np.ndarray,
    out_path: Path,
) -> None:
    """
    Plot top-2D representation colored by cluster with GMM ellipses.

    The GMM is fitted only for visualization in 2D.
    It does not change the cluster labels.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    scatter = ax.scatter(
        Z2[:, 0],
        Z2[:, 1],
        c=labels,
        s=35,
        alpha=0.8,
    )

    n_clusters = len(np.unique(labels))

    gmm = GaussianMixture(
        n_components=n_clusters,
        covariance_type="full",
        random_state=42,
    )
    gmm.fit(Z2)

    for mean, cov in zip(gmm.means_, gmm.covariances_):
        vals, vecs = np.linalg.eigh(cov)

        order = vals.argsort()[::-1]
        vals = vals[order]
        vecs = vecs[:, order]

        angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))

        for scale in [1, 2]:
            width, height = 2 * scale * np.sqrt(vals)

            ellipse = plt.matplotlib.patches.Ellipse(
                xy=mean,
                width=width,
                height=height,
                angle=angle,
                fill=False,
                linewidth=1.5,
            )
            ax.add_patch(ellipse)

    ax.set_title("Top 2 dimensions with GMM ellipses")
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")
    plt.colorbar(scatter, ax=ax, label="Cluster")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_by_phase(
    Z2: np.ndarray,
    df: pd.DataFrame,
    out_path: Path,
) -> None:
    """
    Plot top-2D representation colored by experimental Phase.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    if "Phase" not in df.columns:
        ax.scatter(Z2[:, 0], Z2[:, 1], s=35, alpha=0.8)
    else:
        phases = sorted(df["Phase"].dropna().unique())

        for phase in phases:
            mask = df["Phase"] == phase
            ax.scatter(
                Z2[mask, 0],
                Z2[mask, 1],
                s=35,
                alpha=0.8,
                label=str(phase),
            )

        ax.legend(title="Phase")

    ax.set_title("Top 2 dimensions colored by Phase")
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------
# Categorical association tests
# ---------------------------------------------------------------------
def categorical_tests(df: pd.DataFrame, eval_dir: Path) -> pd.DataFrame:
    """
    For each categorical variable, save:
        - contingency table CSV
        - contingency heatmap PNG

    Also compute:
        - ARI
        - NMI
        - chi-square p-value
        - Cramér's V
    """
    rows = []

    for var in CATEGORICAL_VARS:
        if var not in df.columns:
            continue

        usable = df[[var, "Cluster"]].dropna()

        if usable[var].nunique() < 2 or usable["Cluster"].nunique() < 2:
            continue

        table = pd.crosstab(usable[var], usable["Cluster"])

        table.to_csv(eval_dir / f"contingency_{var}.csv")

        plot_contingency_table(
            table=table,
            var=var,
            out_path=eval_dir / f"contingency_{var}.png",
        )

        chi2, p_value, dof, _ = chi2_contingency(table)

        rows.append(
            {
                "variable": var,
                "ari": adjusted_rand_score(usable[var], usable["Cluster"]),
                "nmi": normalized_mutual_info_score(usable[var], usable["Cluster"]),
                "chi_square": float(chi2),
                "chi_square_dof": int(dof),
                "chi_square_p_value": float(p_value),
                "cramers_v": cramers_v(table),
                "n_unique_values": int(usable[var].nunique()),
                "n_samples_used": int(len(usable)),
            }
        )

    out = pd.DataFrame(rows)

    if not out.empty:
        out = out.sort_values("cramers_v", ascending=False)

    return out


# ---------------------------------------------------------------------
# Biosignal feature extraction
# ---------------------------------------------------------------------
def load_window_metadata_from_processed(arrays: np.lib.npyio.NpzFile, processed_file: Path) -> pd.DataFrame:
    """
    Load window metadata from processed .npz.

    Expected preferred format:
        window_meta_json

    Fallback:
        window_metadata.csv in same folder.
    """
    if "window_meta_json" in arrays:
        return pd.read_json(
            StringIO(str(arrays["window_meta_json"])),
            orient="split",
        )

    metadata_csv = processed_file.parent / "window_metadata.csv"

    if metadata_csv.exists():
        return pd.read_csv(metadata_csv)

    raise ValueError(
        "Could not find window metadata. Expected window_meta_json in the "
        "processed .npz or window_metadata.csv in the same folder."
    )


def build_phase_biosignal_features(
    processed_file: Path,
    phase_meta: pd.DataFrame,
) -> tuple[np.ndarray, list[str], np.ndarray]:
    """
    Build interpretable phase-level biosignal features from processed windows.

    Features computed per phase:
        <SIGNAL>_mean
        <SIGNAL>_std
        <SIGNAL>_min
        <SIGNAL>_max
        <SIGNAL>_median

    Example:
        HR_mean, HR_std, EDA_mean, TEMP_median, ...

    Returns:
        X_features:
            numeric feature matrix

        feature_names:
            names of biosignal features

        labels:
            Cluster labels aligned to X_features
    """
    if not processed_file.exists():
        raise FileNotFoundError(f"Missing processed file: {processed_file}")

    arrays = np.load(processed_file, allow_pickle=True)

    if "X_scaled" not in arrays:
        raise ValueError("Processed file must contain X_scaled.")

    X_windows = arrays["X_scaled"]
    window_meta = load_window_metadata_from_processed(arrays, processed_file)

    if "signals" in arrays:
        signal_names = [str(s) for s in arrays["signals"]]
    else:
        signal_names = ["HR", "EDA", "TEMP"]

    missing_keys = [col for col in GROUP_KEYS if col not in window_meta.columns]
    if missing_keys:
        raise ValueError(
            f"Processed window metadata is missing grouping columns: {missing_keys}"
        )

    missing_phase_keys = [col for col in GROUP_KEYS + ["Cluster"] if col not in phase_meta.columns]
    if missing_phase_keys:
        raise ValueError(
            f"Phase metadata is missing columns required for merging: {missing_phase_keys}"
        )

    rows = []

    for keys, group in window_meta.groupby(GROUP_KEYS, sort=True, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)

        idx = group.index.to_numpy()
        phase_windows = X_windows[idx]  # n_windows x time x channels

        row = dict(zip(GROUP_KEYS, keys))

        for c, signal in enumerate(signal_names):
            values = phase_windows[:, :, c].reshape(-1)

            row[f"{signal}_mean"] = float(np.nanmean(values))
            row[f"{signal}_std"] = float(np.nanstd(values))
            row[f"{signal}_min"] = float(np.nanmin(values))
            row[f"{signal}_max"] = float(np.nanmax(values))
            row[f"{signal}_median"] = float(np.nanmedian(values))

        rows.append(row)

    feature_df = pd.DataFrame(rows)

    merged = feature_df.merge(
        phase_meta[GROUP_KEYS + ["Cluster"]],
        on=GROUP_KEYS,
        how="inner",
    )

    feature_names = [
        col
        for col in merged.columns
        if col not in GROUP_KEYS + ["Cluster"]
    ]

    if not feature_names:
        raise ValueError("No biosignal features were created.")

    X_features = merged[feature_names].to_numpy(dtype=float)
    labels = merged["Cluster"].to_numpy()

    return X_features, feature_names, labels


# ---------------------------------------------------------------------
# Discriminating features
# ---------------------------------------------------------------------
def discriminating_features_named(
    X: np.ndarray,
    labels: np.ndarray,
    feature_names: list[str],
) -> pd.DataFrame:
    """
    Rank features that best separate clusters.

    Uses:
        Mann-Whitney U test
        Cohen's d

    For k=2:
        each cluster is compared against the other cluster.

    For k>2:
        each cluster is compared against all remaining clusters.

    The output is sorted by absolute Cohen's d.
    """
    rows = []
    unique_clusters = sorted(np.unique(labels))

    for cluster in unique_clusters:
        in_cluster = labels == cluster
        out_cluster = labels != cluster

        for j, feature_name in enumerate(feature_names):
            x = X[in_cluster, j]
            y = X[out_cluster, j]

            try:
                stat, p_value = mannwhitneyu(
                    x,
                    y,
                    alternative="two-sided",
                )
            except Exception:
                stat, p_value = np.nan, np.nan

            d = cohen_d(x, y)

            rows.append(
                {
                    "cluster_vs_rest": cluster,
                    "feature": feature_name,
                    "mean_in_cluster": float(np.nanmean(x)),
                    "mean_out_cluster": float(np.nanmean(y)),
                    "mean_difference": float(np.nanmean(x) - np.nanmean(y)),
                    "mannwhitney_u": stat,
                    "p_value": p_value,
                    "cohens_d": d,
                    "abs_cohens_d": abs(d) if not pd.isna(d) else np.nan,
                }
            )

    out = pd.DataFrame(rows)
    out = out.sort_values("abs_cohens_d", ascending=False, na_position="last")
    return out


def discriminating_latent_features(
    X: np.ndarray,
    labels: np.ndarray,
) -> pd.DataFrame:
    """
    Fallback feature ranking using latent/PCA dimensions.

    For autoencoders:
        feature_0, feature_1, ... are latent dimensions.

    For PCA:
        feature_0, feature_1, ... are PCA components.
    """
    feature_names = [f"feature_{j}" for j in range(X.shape[1])]

    return discriminating_features_named(
        X=X,
        labels=labels,
        feature_names=feature_names,
    )


# ---------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------
def write_summary(
    eval_dir: Path,
    output_dir: Path,
    tests_df: pd.DataFrame,
    features_df: pd.DataFrame,
    feature_source: str,
) -> None:
    path = eval_dir / "summary.txt"

    with open(path, "w", encoding="utf-8") as f:
        f.write("Final biosignal cluster report\n")
        f.write("=" * 31 + "\n\n")

        f.write(f"Experiment folder: {output_dir}\n\n")

        f.write("Generated figures\n")
        f.write("-" * 17 + "\n")
        f.write("gmm_top2_ellipses.png\n")
        f.write("top2_by_phase.png\n")
        for var in CATEGORICAL_VARS:
            f.write(f"contingency_{var}.png\n")
        f.write("\n")

        f.write("Categorical association tests\n")
        f.write("-" * 31 + "\n")
        if tests_df.empty:
            f.write("No categorical tests available.\n\n")
        else:
            cols = [
                "variable",
                "ari",
                "nmi",
                "chi_square_p_value",
                "cramers_v",
                "n_samples_used",
            ]
            f.write(tests_df[cols].to_string(index=False))
            f.write("\n\n")

        f.write("Top discriminating features\n")
        f.write("-" * 27 + "\n")
        f.write(f"Feature source: {feature_source}\n\n")

        if features_df.empty:
            f.write("No feature tests available.\n\n")
        else:
            cols = [
                "cluster_vs_rest",
                "feature",
                "mean_difference",
                "p_value",
                "cohens_d",
            ]
            f.write(features_df[cols].head(20).to_string(index=False))
            f.write("\n\n")

        f.write("Interpretation guide\n")
        f.write("-" * 20 + "\n")
        f.write(
            "ARI/NMI evaluate how similar cluster labels are to known metadata labels.\n"
            "Chi-square p-values test whether cluster membership is associated with a categorical variable.\n"
            "Cramer's V measures association strength: near 0 = weak, around 0.3 = moderate, 0.5+ = strong.\n"
            "Mann-Whitney U tests whether feature values differ between a cluster and the remaining clusters.\n"
            "Cohen's d measures the size of that difference. Larger absolute values indicate stronger discriminating features.\n"
            "If --processed-file was provided, discriminating features are biosignal summaries such as HR_mean or EDA_std.\n"
            "Otherwise, discriminating features are latent/PCA dimensions.\n"
        )


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create final biosignal cluster report."
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Experiment output folder.",
    )

    parser.add_argument(
        "--processed-file",
        type=Path,
        default=None,
        help=(
            "Optional processed .npz file used to compute real HR/EDA/TEMP "
            "biosignal features for Mann-Whitney U + Cohen's d."
        ),
    )

    parser.add_argument(
        "--eval-subdir",
        type=str,
        default="eval/final_biosignal_report",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
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

    labels = df["Cluster"].to_numpy()

    if len(X) != len(df):
        raise ValueError("phase_latents.npy and metadata must have same rows.")

    # ------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------
    Z2 = get_top2_projection(X)

    plot_gmm_ellipses(
        Z2=Z2,
        labels=labels,
        out_path=eval_dir / "gmm_top2_ellipses.png",
    )

    plot_by_phase(
        Z2=Z2,
        df=df,
        out_path=eval_dir / "top2_by_phase.png",
    )

    # ------------------------------------------------------------
    # Categorical tests + contingency tables/heatmaps
    # ------------------------------------------------------------
    tests_df = categorical_tests(df, eval_dir)
    tests_df.to_csv(eval_dir / "categorical_tests.csv", index=False)

    # ------------------------------------------------------------
    # Discriminating features
    # ------------------------------------------------------------
    if args.processed_file is not None:
        X_features, feature_names, feature_labels = build_phase_biosignal_features(
            processed_file=args.processed_file,
            phase_meta=df,
        )

        features_df = discriminating_features_named(
            X=X_features,
            labels=feature_labels,
            feature_names=feature_names,
        )

        feature_source = "biosignal_features_from_processed_windows"

    else:
        features_df = discriminating_latent_features(
            X=X,
            labels=labels,
        )

        feature_source = "latent_or_pca_dimensions"

    features_df["feature_source"] = feature_source
    features_df.to_csv(eval_dir / "discriminating_features.csv", index=False)

    # ------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------
    write_summary(
        eval_dir=eval_dir,
        output_dir=output_dir,
        tests_df=tests_df,
        features_df=features_df,
        feature_source=feature_source,
    )

    print(f"Final biosignal report saved to: {eval_dir}")


if __name__ == "__main__":
    main()