"""
preprocessing.py
================
Cleaning and transformation utilities for the EmoPairCompete dataset.

Functions
---------
drop_missing(df, threshold)
    Drop rows or columns that exceed a missing-value threshold.
impute_missing(df, strategy)
    Fill remaining NaN values using the given strategy.
encode_phase(df)
    Ordinal-encode the ``phase`` column (pre=0, puzzle=1, post=2).
normalise_features(df, feature_cols)
    Z-score normalise the specified feature columns in place.
"""

from __future__ import annotations

import logging
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

PHASE_ORDER = {"pre": 0, "puzzle": 1, "post": 2}


def drop_missing(
    df: pd.DataFrame,
    row_threshold: float = 0.5,
    col_threshold: float = 0.5,
) -> pd.DataFrame:
    """Drop columns then rows with too many missing values.

    Parameters
    ----------
    df:
        Input DataFrame.
    row_threshold:
        Drop rows where the fraction of NaN values exceeds this threshold.
    col_threshold:
        Drop columns where the fraction of NaN values exceeds this threshold.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame (copy).
    """
    df = df.copy()
    n_rows = len(df)
    n_cols = len(df.columns)

    col_mask = df.isnull().mean(axis=0) <= col_threshold
    df = df.loc[:, col_mask]
    dropped_cols = n_cols - col_mask.sum()
    if dropped_cols:
        logging.getLogger(__name__).info("Dropped %d columns with >%.0f%% missing", dropped_cols, col_threshold * 100)

    row_mask = df.isnull().mean(axis=1) <= row_threshold
    df = df.loc[row_mask]
    dropped_rows = n_rows - row_mask.sum()
    if dropped_rows:
        logging.getLogger(__name__).info("Dropped %d rows with >%.0f%% missing", dropped_rows, row_threshold * 100)

    return df.reset_index(drop=True)


def impute_missing(
    df: pd.DataFrame,
    strategy: str = "median",
    numeric_only: bool = True,
) -> pd.DataFrame:
    """Impute NaN values in numeric columns.

    Parameters
    ----------
    df:
        Input DataFrame.
    strategy:
        One of ``"mean"``, ``"median"``, or ``"zero"``.
    numeric_only:
        When True (default) only numeric columns are imputed.

    Returns
    -------
    pd.DataFrame
        DataFrame with NaN values filled (copy).
    """
    df = df.copy()
    num_cols = df.select_dtypes(include=[np.number]).columns if numeric_only else df.columns

    if strategy == "mean":
        df[num_cols] = df[num_cols].fillna(df[num_cols].mean())
    elif strategy == "median":
        df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    elif strategy == "zero":
        df[num_cols] = df[num_cols].fillna(0)
    else:
        raise ValueError(f"Unknown imputation strategy: '{strategy}'. Choose 'mean', 'median', or 'zero'.")

    return df


def encode_phase(df: pd.DataFrame, col: str = "phase") -> pd.DataFrame:
    """Ordinal-encode the phase column (pre=0, puzzle=1, post=2).

    Parameters
    ----------
    df:
        Input DataFrame containing a ``phase`` column.
    col:
        Name of the phase column (default ``"phase"``).

    Returns
    -------
    pd.DataFrame
        DataFrame with a new ``phase_encoded`` integer column appended.
    """
    df = df.copy()
    df["phase_encoded"] = df[col].map(PHASE_ORDER)
    return df


def normalise_features(
    df: pd.DataFrame,
    feature_cols: Iterable[str],
) -> pd.DataFrame:
    """Z-score normalise the specified feature columns.

    Parameters
    ----------
    df:
        Input DataFrame.
    feature_cols:
        Column names to normalise.

    Returns
    -------
    pd.DataFrame
        DataFrame with normalised feature columns (copy). Original column
        values are replaced in-place within the copy.
    """
    df = df.copy()
    cols = [c for c in feature_cols if c in df.columns]
    if not cols:
        return df

    scaler = StandardScaler()
    df[cols] = scaler.fit_transform(df[cols].astype(float))
    return df

# global or phase-wise alternatives
def normalise_features(
    df: pd.DataFrame,
    feature_cols: Iterable[str],
    within_phase: bool = True
) -> pd.DataFrame:
    """Z-score normalise the specified feature columns.

    Parameters
    ----------
    df:
        Input DataFrame.
    feature_cols:
        Column names to normalise.
    within_phase:
        If True, normalise relative to the mean/std of each phase [cite: 54, 59-61].
        If False, normalise relative to the entire dataset (Global).

    Returns
    -------
    pd.DataFrame
        DataFrame with normalised feature columns (copy).
    """
    df = df.copy()
    cols = [c for c in feature_cols if c in df.columns]
    if not cols:
        return df

    scaler = StandardScaler()
    
    if within_phase:
        # Standardize relative to each experimental phase [cite: 59-61, 66-67]
        # This removes the "Phase Effect" so clustering focuses on latent states [cite: 110-112]
        df[cols] = df.groupby('Phase')[cols].transform(
            lambda x: scaler.fit_transform(x.values.reshape(-1, 1)).flatten()
        )
    else:
        # Standardize relative to the whole dataset
        df[cols] = scaler.fit_transform(df[cols].astype(float))
        
    return df