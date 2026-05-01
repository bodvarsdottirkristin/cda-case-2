"""
Check how many PCA components are needed to explain 80% of variance.

Example:
    python advanced/check_pca_80_variance.py \
        --processed-file data/processed/autoencoder/autoencoder_windows_cohort_norm.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA


def load_pca_input(processed_file: Path) -> np.ndarray:
    """
    Load processed windows and flatten them for PCA.

    Input shape:
        N x T x C

    PCA input shape:
        N x (T*C)
    """
    if not processed_file.exists():
        raise FileNotFoundError(f"Could not find {processed_file}")

    arrays = np.load(processed_file, allow_pickle=True)

    if "X_scaled" not in arrays:
        raise ValueError("Processed file must contain X_scaled.")

    X = arrays["X_scaled"].astype(np.float32)

    if X.ndim != 3:
        raise ValueError("X_scaled must have shape N x T x C.")

    n_windows, window_size, n_channels = X.shape

    X_flat = X.reshape(n_windows, window_size * n_channels)

    return X_flat


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Find number of PCA components needed for 80% variance."
    )

    parser.add_argument(
        "--processed-file",
        type=Path,
        required=True,
        help="Processed .npz file containing X_scaled.",
    )

    parser.add_argument(
        "--target-variance",
        type=float,
        default=0.80,
        help="Target cumulative explained variance. Default: 0.80.",
    )

    args = parser.parse_args()

    X = load_pca_input(args.processed_file)

    print(f"PCA input shape: {X.shape}")

    pca = PCA()
    pca.fit(X)

    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

    n_components = int(np.searchsorted(cumulative_variance, args.target_variance) + 1)

    print(
        f"Components needed for {args.target_variance:.0%} explained variance: "
        f"{n_components}"
    )

    print(
        f"Explained variance with {n_components} components: "
        f"{cumulative_variance[n_components - 1]:.4f}"
    )

    print("\nFirst 20 cumulative explained variance values:")
    for i, value in enumerate(cumulative_variance[:20], start=1):
        print(f"{i:2d} components: {value:.4f}")


if __name__ == "__main__":
    main()