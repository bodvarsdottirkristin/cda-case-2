"""
features.py
===========
Feature engineering utilities for the EmoPairCompete dataset.

Functions
---------
compute_delta_features(df, feature_cols)
    Compute within-participant, within-round change between phases.
compute_ratio_features(df, numerator_cols, denominator_cols)
    Compute element-wise ratios between two sets of feature columns.
select_features_variance(df, feature_cols, threshold)
    Remove low-variance features.
get_biosignal_feature_cols()
    Return the canonical list of biosignal feature column names.
get_questionnaire_cols()
    Return the canonical list of questionnaire column names.
"""

from __future__ import annotations

from typing import Iterable, List

import numpy as np
import pandas as pd

from src.data_loader import (
    BIOSIGNAL_FEATURE_COLS,
    QUESTIONNAIRE_COLS,
)


def get_biosignal_feature_cols() -> List[str]:
    """Return the canonical list of biosignal feature column names."""
    return list(BIOSIGNAL_FEATURE_COLS)


def get_questionnaire_cols() -> List[str]:
    """Return the canonical list of questionnaire column names."""
    return list(QUESTIONNAIRE_COLS)


def compute_delta_features(
    df: pd.DataFrame,
    feature_cols: Iterable[str],
    from_phase: str = "pre",
    to_phase: str = "puzzle",
    group_cols: Iterable[str] = ("cohort", "participant_id", "round"),
) -> pd.DataFrame:
    """Compute within-participant per-round delta (to_phase − from_phase).

    A new column ``<feature>_delta`` is added for each feature in
    *feature_cols*.  Rows that do not correspond to *to_phase* are dropped.

    Parameters
    ----------
    df:
        Unified DataFrame as returned by :func:`src.data_loader.load_dataset`.
    feature_cols:
        Feature columns to compute deltas for.
    from_phase:
        The baseline phase (default ``"pre"``).
    to_phase:
        The target phase (default ``"puzzle"``).
    group_cols:
        Columns that identify a unique participant-round combination.

    Returns
    -------
    pd.DataFrame
        DataFrame restricted to *to_phase* rows with additional ``_delta``
        columns.
    """
    feature_cols = list(feature_cols)
    group_cols = list(group_cols)

    baseline = df[df["phase"] == from_phase][group_cols + feature_cols].copy()
    baseline = baseline.rename(columns={c: f"{c}_baseline" for c in feature_cols})

    target = df[df["phase"] == to_phase].copy()
    merged = target.merge(baseline, on=group_cols, how="left")

    for col in feature_cols:
        merged[f"{col}_delta"] = merged[col] - merged[f"{col}_baseline"]
        merged = merged.drop(columns=[f"{col}_baseline"])

    return merged.reset_index(drop=True)


def compute_ratio_features(
    df: pd.DataFrame,
    numerator_cols: Iterable[str],
    denominator_cols: Iterable[str],
    suffix: str = "_ratio",
    fill_value: float = np.nan,
) -> pd.DataFrame:
    """Compute element-wise ratios between two lists of feature columns.

    Parameters
    ----------
    df:
        Input DataFrame.
    numerator_cols:
        Column names for numerators.
    denominator_cols:
        Column names for denominators (same length as *numerator_cols*).
    suffix:
        Suffix appended to the numerator column name for the new ratio column.
    fill_value:
        Value used when the denominator is zero.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with additional ratio columns.
    """
    numerator_cols = list(numerator_cols)
    denominator_cols = list(denominator_cols)
    if len(numerator_cols) != len(denominator_cols):
        raise ValueError("numerator_cols and denominator_cols must have the same length.")

    df = df.copy()
    for num_col, den_col in zip(numerator_cols, denominator_cols):
        ratio_col = f"{num_col}{suffix}"
        denom = df[den_col].replace(0, np.nan)
        df[ratio_col] = df[num_col] / denom
        df[ratio_col] = df[ratio_col].fillna(fill_value)

    return df


def select_features_variance(
    df: pd.DataFrame,
    feature_cols: Iterable[str],
    threshold: float = 0.0,
) -> List[str]:
    """Return feature columns whose variance exceeds *threshold*.

    Parameters
    ----------
    df:
        Input DataFrame.
    feature_cols:
        Candidate feature column names.
    threshold:
        Minimum variance required to keep a feature (default ``0.0`` removes
        only zero-variance features).

    Returns
    -------
    list[str]
        Names of features that pass the variance threshold.
    """
    feature_cols = [c for c in feature_cols if c in df.columns]
    variances = df[feature_cols].var(numeric_only=True)
    return list(variances[variances > threshold].index)
