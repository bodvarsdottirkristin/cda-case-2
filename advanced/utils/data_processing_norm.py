"""
Create normalized processed data for the autoencoder pipeline.

This script creates a reusable processed-data file with the same structure
expected by v1_autoencoding.py, but with extra baseline normalization.

Important metadata behavior:
    data_processing.py standardizes:
        participant_ID / particpant_ID -> participant_ID
        puzzler / Puzzler / parent / Parent -> Puzzler

Therefore, this normalized pipeline automatically uses the corrected metadata
because it loads raw data through load_raw_dataset().

It saves files such as:

    data/processed/autoencoder/autoencoder_windows_participant_norm.npz
    data/processed/autoencoder/autoencoder_windows_cohort_norm.npz
    data/processed/autoencoder/autoencoder_windows_participant_round_norm.npz

Use this when the normal autoencoder pipeline finds mostly cohort/session or
participant-baseline effects.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Paths and local imports
# ---------------------------------------------------------------------
UTILS_DIR = Path(__file__).resolve().parent
ADVANCED_DIR = UTILS_DIR.parent
PROJECT_ROOT = ADVANCED_DIR.parent

sys.path.insert(0, str(ADVANCED_DIR))

from utils.data_processing import (  # noqa: E402
    SIGNALS,
    META_KEYS,
    load_raw_dataset,
    create_windows,
    to_conv1d_format,
)


# ---------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------
def normalize_signals_by_group(
    df: pd.DataFrame,
    signal_cols: Sequence[str],
    group_cols: Sequence[str],
    eps: float = 1e-8,
) -> pd.DataFrame:
    """
    Normalize signal columns within groups.

    For each group:
        x_norm = (x - group_mean) / group_std

    Recommended groups:
        ["participant_ID"]
        ["participant_ID", "Round"]
        ["Cohort"]

    participant_ID is preferred over folder-level Individual because
    Individual is only a local folder name and is not globally reliable.
    """
    df = df.copy()

    missing = [
        col
        for col in list(signal_cols) + list(group_cols)
        if col not in df.columns
    ]

    if missing:
        raise ValueError(f"Missing columns required for normalization: {missing}")

    for signal in signal_cols:
        group_mean = df.groupby(list(group_cols))[signal].transform("mean")
        group_std = df.groupby(list(group_cols))[signal].transform("std")

        df[signal] = (df[signal] - group_mean) / (group_std + eps)

    return df


def save_processed_file(
    processed_file: Path,
    X_raw: np.ndarray,
    X_scaled: np.ndarray,
    X_conv1d: np.ndarray,
    window_meta: pd.DataFrame,
    signals: Sequence[str],
    window_size: int,
    step_size: int,
    normalization: str,
    normalization_group_cols: Sequence[str],
) -> None:
    """
    Save one reusable normalized processed-data file.

    This uses the same .npz structure expected by v1_autoencoding.py /
    v2_autoencoding.py.
    """
    processed_file.parent.mkdir(parents=True, exist_ok=True)

    window_meta_json = window_meta.to_json(orient="split")

    np.savez_compressed(
        processed_file,
        X_raw=X_raw.astype(np.float32),
        X_scaled=X_scaled.astype(np.float32),
        X_conv1d=X_conv1d.astype(np.float32),
        window_meta_json=np.array(window_meta_json),
        signals=np.array(list(signals)),
        window_size=np.array(window_size),
        step_size=np.array(step_size),
        normalization=np.array(normalization),
        normalization_group_cols=np.array(list(normalization_group_cols)),
    )

    print(f"Saved normalized processed data to: {processed_file}")


def build_normalized_autoencoder_file(
    dataset_dir: Path,
    processed_file: Path,
    signals: Sequence[str],
    window_size: int,
    step_size: int,
    normalization: str,
    resample_rule: str = "1s",
) -> None:
    """
    Build normalized processed data and save it as one .npz file.

    The raw dataset is loaded through data_processing.load_raw_dataset(),
    so metadata is already standardized:

        participant_ID / particpant_ID -> participant_ID
        puzzler / Puzzler / parent / Parent -> Puzzler
    """
    normalization_to_groups = {
        # Recommended participant-level normalization.
        "participant": ["participant_ID"],

        # More aggressive: removes participant baseline separately per round.
        "participant_round": ["participant_ID", "Round"],

        # Session-level normalization.
        "cohort": ["Cohort"],

        # Optional session-round normalization.
        "cohort_round": ["Cohort", "Round"],

        # Backward-compatible options, but less recommended.
        # Individual is the folder-level ID, not the real participant identifier.
        "individual": ["Individual"],
        "cohort_participant": ["Cohort", "participant_ID"],
        "cohort_individual_round": ["Cohort", "Individual", "Round"],
    }

    if normalization not in normalization_to_groups:
        raise ValueError(
            f"Unknown normalization: {normalization}. "
            f"Choose from {list(normalization_to_groups)}"
        )

    group_cols = normalization_to_groups[normalization]

    print(f"Loading raw dataset from: {dataset_dir}")

    full_df = load_raw_dataset(
        dataset_dir=dataset_dir,
        signals=signals,
        resample_rule=resample_rule,
    )

    print(f"Loaded rows before normalization: {len(full_df):,}")
    print(f"Applying normalization: {normalization}")
    print(f"Normalization groups: {group_cols}")
    print(f"Window grouping keys from data_processing.META_KEYS: {META_KEYS}")

    if "participant_ID" in full_df.columns:
        print("participant_ID values before windowing:")
        print(full_df["participant_ID"].value_counts(dropna=False))
    else:
        print("Warning: participant_ID column not found in loaded raw dataframe.")

    if "Puzzler" in full_df.columns:
        print("Puzzler values before windowing:")
        print(full_df["Puzzler"].value_counts(dropna=False))
    else:
        print("Warning: Puzzler column not found in loaded raw dataframe.")

    if "parent" in full_df.columns or "Parent" in full_df.columns:
        print(
            "Warning: parent/Parent column is still present. "
            "It should normally be merged into Puzzler by data_processing.py."
        )

    normalized_df = normalize_signals_by_group(
        df=full_df,
        signal_cols=signals,
        group_cols=group_cols,
    )

    X_raw, window_meta = create_windows(
        df=normalized_df,
        signal_cols=signals,
        meta_keys=META_KEYS,
        window_size=window_size,
        step_size=step_size,
    )

    # Here X_raw is already normalized because it comes from normalized_df.
    # To stay compatible with v1_autoencoding.py / v2_autoencoding.py,
    # we store it both as X_raw and X_scaled.
    X_scaled = X_raw.astype(np.float32)
    X_conv1d = to_conv1d_format(X_scaled)

    save_processed_file(
        processed_file=processed_file,
        X_raw=X_raw,
        X_scaled=X_scaled,
        X_conv1d=X_conv1d,
        window_meta=window_meta,
        signals=signals,
        window_size=window_size,
        step_size=step_size,
        normalization=normalization,
        normalization_group_cols=group_cols,
    )

    print(
        f"Created windows: {X_scaled.shape[0]:,} windows x "
        f"{X_scaled.shape[1]} seconds x {X_scaled.shape[2]} signals"
    )

    if "participant_ID" in window_meta.columns:
        print("participant_ID values in window metadata:")
        print(window_meta["participant_ID"].value_counts(dropna=False))

    if "Individual" in window_meta.columns:
        print("Folder-level Individual values in window metadata:")
        print(window_meta["Individual"].value_counts(dropna=False))

    if "Puzzler" in window_meta.columns:
        print("Puzzler values in window metadata:")
        print(window_meta["Puzzler"].value_counts(dropna=False))

    if "parent" in window_meta.columns or "Parent" in window_meta.columns:
        print(
            "Warning: parent/Parent is still present in window metadata. "
            "It should normally be merged into Puzzler."
        )


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    default_dataset_dir = PROJECT_ROOT / "data" / "raw" / "data" / "dataset"

    parser = argparse.ArgumentParser(
        description="Create normalized processed data for autoencoder models."
    )

    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=default_dataset_dir,
        help="Path to raw dataset folder.",
    )

    parser.add_argument(
        "--normalization",
        choices=[
            "participant",
            "participant_round",
            "cohort",
            "cohort_round",
            "individual",
            "cohort_participant",
            "cohort_individual_round",
        ],
        default="participant",
        help=(
            "Which baseline normalization to apply. "
            "Recommended: participant or cohort."
        ),
    )

    parser.add_argument("--window-size", type=int, default=60)
    parser.add_argument("--step-size", type=int, default=30)
    parser.add_argument("--resample-rule", type=str, default="1s")

    parser.add_argument(
        "--signals",
        nargs="+",
        default=list(SIGNALS),
        help="Signal files to use. Default: HR EDA TEMP.",
    )

    parser.add_argument(
        "--processed-file",
        type=Path,
        default=None,
        help=(
            "Optional output .npz file. "
            "If omitted, a name is generated automatically."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.processed_file is None:
        args.processed_file = (
            PROJECT_ROOT
            / "data"
            / "processed"
            / "autoencoder"
            / f"autoencoder_windows_{args.normalization}_norm.npz"
        )

    build_normalized_autoencoder_file(
        dataset_dir=args.dataset_dir,
        processed_file=args.processed_file,
        signals=tuple(args.signals),
        window_size=args.window_size,
        step_size=args.step_size,
        normalization=args.normalization,
        resample_rule=args.resample_rule,
    )


if __name__ == "__main__":
    main()