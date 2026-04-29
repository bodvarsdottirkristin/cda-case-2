"""
PCA utilities for the raw time-series autoencoder project.

This module provides a PCA baseline using the same processed window data
used by the Conv1D autoencoder.

Pipeline:

    processed autoencoder windows
    → flatten each 60 x 3 window into 180 features
    → PCA on window-level data
    → average PCA scores per phase
    → K-Means clustering on phase-level PCA representation

The goal is to compare a simple linear representation against the nonlinear
Conv1D autoencoder.
"""

from __future__ import annotations

from io import StringIO
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


META_KEYS: tuple[str, ...] = ("Cohort", "Individual", "Round", "Phase")


def load_processed_autoencoder_file(
    processed_file: Path,
) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Load the reusable processed autoencoder file.

    Expected file:
        data/processed/autoencoder/autoencoder_windows.npz

    Returns
    -------
    X_scaled:
        Shape: n_windows x window_size x n_signals

    window_meta:
        One row per window.
    """
    processed_file = Path(processed_file)

    if not processed_file.exists():
        raise FileNotFoundError(f"Processed file not found: {processed_file}")

    data = np.load(processed_file, allow_pickle=True)

    if "X_scaled" not in data:
        raise ValueError(f"{processed_file} does not contain X_scaled.")

    X_scaled = data["X_scaled"]

    if "window_meta_json" in data:
        window_meta_json = str(data["window_meta_json"])
        window_meta = pd.read_json(StringIO(window_meta_json), orient="split")
    else:
        raise ValueError(
            f"{processed_file} does not contain window_meta_json. "
            "Rebuild the processed file with the current pipeline."
        )

    return X_scaled.astype(np.float32), window_meta


def flatten_windows(X: np.ndarray) -> np.ndarray:
    """
    Flatten windows from:

        n_windows x window_size x n_signals

    to:

        n_windows x (window_size * n_signals)
    """
    if X.ndim != 3:
        raise ValueError("X must have shape (n_windows, window_size, n_signals).")

    n_windows = X.shape[0]
    return X.reshape(n_windows, -1).astype(np.float32)


def fit_window_pca(
    X_flat: np.ndarray,
    n_components: int,
    standardize_flattened_features: bool = False,
    random_state: int = 42,
) -> tuple[np.ndarray, PCA, StandardScaler | None]:
    """
    Fit PCA on flattened window-level data.

    Parameters
    ----------
    X_flat:
        Shape: n_windows x flattened_features.

    n_components:
        Number of PCA components.

    standardize_flattened_features:
        If True, standardize each flattened feature before PCA.
        Usually False is fine because X_scaled was already standardized per signal
        before flattening.

    Returns
    -------
    window_pca_scores:
        Shape: n_windows x n_components.

    pca:
        Fitted sklearn PCA object.

    scaler:
        Fitted StandardScaler if standardize_flattened_features=True, else None.
    """
    if X_flat.ndim != 2:
        raise ValueError("X_flat must have shape (n_windows, n_features).")

    if n_components < 1:
        raise ValueError("n_components must be >= 1.")

    max_components = min(X_flat.shape[0], X_flat.shape[1])

    if n_components > max_components:
        raise ValueError(
            f"n_components={n_components} is too large. "
            f"Maximum allowed is {max_components}."
        )

    scaler = None

    if standardize_flattened_features:
        scaler = StandardScaler()
        X_for_pca = scaler.fit_transform(X_flat)
    else:
        X_for_pca = X_flat

    pca = PCA(n_components=n_components, random_state=random_state)
    window_pca_scores = pca.fit_transform(X_for_pca)

    return window_pca_scores.astype(np.float32), pca, scaler


def aggregate_window_features_per_phase(
    window_features: np.ndarray,
    window_meta: pd.DataFrame,
    meta_keys: Iterable[str] = META_KEYS,
) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Average window-level features per Cohort/Individual/Round/Phase.

    This mirrors the autoencoder pipeline:
        window representation → phase-level representation
    """
    meta_keys = list(meta_keys)

    missing = [col for col in meta_keys if col not in window_meta.columns]
    if missing:
        raise ValueError(f"Missing metadata columns: {missing}")

    if len(window_features) != len(window_meta):
        raise ValueError(
            "window_features and window_meta must have the same number of rows."
        )

    phase_features: list[np.ndarray] = []
    phase_meta_rows: list[dict[str, object]] = []

    for keys, group in window_meta.groupby(meta_keys, sort=True, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)

        phase_features.append(window_features[group.index].mean(axis=0))

        row = dict(zip(meta_keys, keys))

        # Preserve useful extra phase-level metadata, for example questionnaire values.
        for col in window_meta.columns:
            if col in meta_keys or col.startswith("Window"):
                continue

            values = group[col].dropna()
            row[col] = values.iloc[0] if len(values) else np.nan

        phase_meta_rows.append(row)

    return np.stack(phase_features).astype(np.float32), pd.DataFrame(phase_meta_rows)


def run_kmeans_sweep(
    phase_features: np.ndarray,
    k_min: int,
    k_max: int,
    random_state: int,
) -> tuple[dict[int, float], int, np.ndarray]:
    """
    Run K-Means over a range of k values and select k by silhouette score.
    """
    n_samples = len(phase_features)

    if n_samples < 3:
        raise ValueError("Need at least 3 phase-level samples for clustering.")

    k_max = min(k_max, n_samples - 1)

    if k_min > k_max:
        raise ValueError(
            f"Invalid k range after sample-size check: k_min={k_min}, k_max={k_max}"
        )

    silhouette_scores: dict[int, float] = {}

    for k in range(k_min, k_max + 1):
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = kmeans.fit_predict(phase_features)
        silhouette_scores[k] = float(silhouette_score(phase_features, labels))

    best_k = max(silhouette_scores, key=silhouette_scores.get)

    final_kmeans = KMeans(n_clusters=best_k, random_state=random_state, n_init=10)
    final_labels = final_kmeans.fit_predict(phase_features)

    return silhouette_scores, best_k, final_labels


def explained_variance_dataframe(pca: PCA) -> pd.DataFrame:
    """
    Return PCA explained variance information as a dataframe.
    """
    explained_ratio = pca.explained_variance_ratio_
    cumulative = np.cumsum(explained_ratio)

    return pd.DataFrame(
        {
            "component": np.arange(1, len(explained_ratio) + 1),
            "explained_variance_ratio": explained_ratio,
            "cumulative_explained_variance_ratio": cumulative,
        }
    )