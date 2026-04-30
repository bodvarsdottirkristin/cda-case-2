"""
Data extraction and preprocessing for the raw EmoPairCompete time-series data.

This module contains only reusable data-processing functions.

It does:
1. Traverse the raw dataset folder.
2. Load HR, EDA, and TEMP files for every cohort/individual/round/phase.
3. Resample each signal to 1 Hz.
4. Attach questionnaire values as phase-level metadata.
5. Standardize inconsistent role columns:
       puzzler / Puzzler / parent / Parent -> Puzzler
6. Create overlapping windows.
7. Standardize each signal channel globally.
8. Convert windows to Conv1D format when requested.

It does NOT:
- define neural network models,
- train models,
- run clustering,
- save experiment results,
- save multiple processed-data files.

The reusable processed-data cache is created by v1_autoencoding.py, not here.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


SIGNALS: tuple[str, ...] = ("HR", "EDA", "TEMP")
META_KEYS: tuple[str, ...] = ("Cohort", "Individual", "Round", "Phase")

# These columns are metadata/role columns, not questionnaire/emotion scores.
# Important: in some response.csv files the role variable appears as "puzzler",
# while in others it appears as "parent". We merge both into a single column:
# "Puzzler".
RESPONSE_META_COLS: tuple[str, ...] = (
    "particpant_ID",  # typo appears in some files
    "participant_ID",
    "puzzler",
    "Puzzler",
    "parent",
    "Parent",
    "team_ID",
    "E4_nr",
)


PUZZLER_ROLE_COLUMNS: tuple[str, ...] = (
    "puzzler",
    "Puzzler",
    "parent",
    "Parent",
)


def infer_project_root() -> Path:
    """
    Infer project root when this file is placed in:

        project_root/advanced/utils/data_processing.py

    Returns:
        project_root
    """
    here = Path(__file__).resolve()

    for parent in here.parents:
        if (parent / "data").exists() and (parent / "advanced").exists():
            return parent

    if len(here.parents) >= 3:
        return here.parents[2]

    return Path.cwd()


def _get_first_existing_column(
    df: pd.DataFrame,
    possible_names: Sequence[str],
) -> str | None:
    """
    Return the first matching column name, ignoring capitalization.

    Example:
        possible_names = ("puzzler", "Puzzler", "parent", "Parent")

    If the dataframe contains "Parent", this returns "Parent".
    """
    lower_to_original = {col.lower(): col for col in df.columns}

    for name in possible_names:
        if name.lower() in lower_to_original:
            return lower_to_original[name.lower()]

    return None


def _read_response_csv(response_csv: Path) -> pd.DataFrame:
    """
    Read response.csv robustly.

    Some files contain an unnamed index column, so we first try index_col=0,
    then fall back to a normal read.
    """
    try:
        return pd.read_csv(response_csv, index_col=0)
    except Exception:
        return pd.read_csv(response_csv)


def _read_signal_csv(csv_path: Path, signal: str) -> pd.DataFrame:
    """
    Read a raw signal CSV robustly.

    Expected columns:
        time
        <signal>

    Some files may contain an unnamed index column, so we first try reading
    with index_col=0 and then fall back to a normal read.
    """
    read_attempts = (
        {"index_col": 0},
        {},
    )

    last_error: Exception | None = None

    for kwargs in read_attempts:
        try:
            df = pd.read_csv(csv_path, **kwargs)

            if "time" in df.columns and signal in df.columns:
                out = df[["time", signal]].copy()
                out["time"] = pd.to_datetime(out["time"], errors="coerce")
                out = out.dropna(subset=["time"])
                out = out.set_index("time").sort_index()
                return out

        except Exception as exc:
            last_error = exc

    if last_error is not None:
        raise ValueError(f"Could not read {csv_path}: {last_error}")

    raise ValueError(
        f"Could not find expected columns 'time' and '{signal}' in {csv_path}"
    )


def load_phase(
    phase_dir: Path,
    cohort: str,
    individual: str,
    round_: str,
    phase: str,
    signals: Sequence[str] = SIGNALS,
    resample_rule: str = "1s",
) -> pd.DataFrame:
    """
    Load and merge signals for one phase.

    Parameters
    ----------
    phase_dir:
        Folder containing HR.csv, EDA.csv, TEMP.csv, and optionally response.csv.
    cohort, individual, round_, phase:
        Metadata from the folder hierarchy.
    signals:
        Signal files to load.
    resample_rule:
        Pandas resampling rule. Default "1s" gives 1 Hz data.

    Returns
    -------
    pd.DataFrame
        One row per resampled timestamp, including signal columns, metadata,
        one standardized role column called Puzzler, and questionnaire columns
        when available.
    """
    dfs: dict[str, pd.Series] = {}

    for signal in signals:
        csv_path = phase_dir / f"{signal}.csv"

        if csv_path.exists():
            signal_df = _read_signal_csv(csv_path, signal)
            dfs[signal] = signal_df[signal]

    if not dfs:
        return pd.DataFrame()

    resampled = {
        signal: series.resample(resample_rule).mean()
        for signal, series in dfs.items()
    }

    merged = pd.concat(resampled, axis=1)

    for signal in signals:
        if signal not in merged.columns:
            merged[signal] = np.nan

    merged = merged[list(signals)]
    merged.index.name = "time"
    merged = merged.reset_index()

    merged["Cohort"] = cohort
    merged["Individual"] = individual
    merged["Round"] = round_
    merged["Phase"] = phase

    response_csv = phase_dir / "response.csv"

    if response_csv.exists():
        resp = _read_response_csv(response_csv)

        if len(resp) > 0:
            # ---------------------------------------------------------
            # Standardize the role variable.
            #
            # Some response.csv files use "puzzler", others use "parent".
            # In this project, these refer to the same role information.
            # We merge them into one clean column: "Puzzler".
            # ---------------------------------------------------------
            puzzler_col = _get_first_existing_column(
                resp,
                PUZZLER_ROLE_COLUMNS,
            )

            if puzzler_col is not None:
                merged["Puzzler"] = resp[puzzler_col].iloc[0]
            else:
                merged["Puzzler"] = np.nan

            # ---------------------------------------------------------
            # Add questionnaire/emotion variables.
            #
            # Exclude all metadata/role columns, including parent, because
            # parent has already been merged into Puzzler.
            # ---------------------------------------------------------
            response_meta_cols_lower = {
                col.lower() for col in RESPONSE_META_COLS
            }

            questionnaire_cols = [
                col
                for col in resp.columns
                if col.lower() not in response_meta_cols_lower
            ]

            for col in questionnaire_cols:
                merged[col] = resp[col].iloc[0]
    else:
        merged["Puzzler"] = np.nan

    return merged


def load_raw_dataset(
    dataset_dir: Path,
    signals: Sequence[str] = SIGNALS,
    resample_rule: str = "1s",
) -> pd.DataFrame:
    """
    Load all available phase data from the raw dataset folder.

    Expected layout:

        dataset/
        └── Cohort/
            └── ID_x/
                └── Round/
                    └── Phase/
                        ├── HR.csv
                        ├── EDA.csv
                        ├── TEMP.csv
                        └── response.csv
    """
    dataset_dir = Path(dataset_dir)

    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {dataset_dir}")

    records: list[pd.DataFrame] = []

    for cohort_dir in sorted(dataset_dir.iterdir()):
        if not cohort_dir.is_dir():
            continue

        cohort = cohort_dir.name

        for id_dir in sorted(cohort_dir.iterdir()):
            if not id_dir.is_dir() or not id_dir.name.startswith("ID_"):
                continue

            individual = id_dir.name

            for round_dir in sorted(id_dir.iterdir()):
                if not round_dir.is_dir():
                    continue

                round_ = round_dir.name

                for phase_dir in sorted(round_dir.iterdir()):
                    if not phase_dir.is_dir():
                        continue

                    phase = phase_dir.name

                    phase_df = load_phase(
                        phase_dir=phase_dir,
                        cohort=cohort,
                        individual=individual,
                        round_=round_,
                        phase=phase,
                        signals=signals,
                        resample_rule=resample_rule,
                    )

                    if not phase_df.empty:
                        records.append(phase_df)

    if not records:
        raise ValueError(
            f"No phase data was loaded from {dataset_dir}. "
            "Check the folder layout."
        )

    return pd.concat(records, ignore_index=True)


def _safe_first_value(series: pd.Series):
    """Return the first non-null value, or NaN if none exists."""
    non_null = series.dropna()

    if len(non_null) == 0:
        return np.nan

    return non_null.iloc[0]


def create_windows(
    df: pd.DataFrame,
    signal_cols: Sequence[str] = SIGNALS,
    meta_keys: Sequence[str] = META_KEYS,
    window_size: int = 60,
    step_size: int = 30,
    keep_extra_phase_metadata: bool = True,
) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Create overlapping fixed-length windows from the resampled signals.

    Default:
        60-second windows
        30-second step size
        skip windows containing NaNs

    Returns
    -------
    X_raw:
        Array with shape:
            n_windows x window_size x n_signals

    window_meta:
        DataFrame with one row per window.
    """
    required = set(signal_cols).union(meta_keys)
    missing = sorted(required.difference(df.columns))

    if missing:
        raise ValueError(f"Missing required columns in dataframe: {missing}")

    windows: list[np.ndarray] = []
    window_meta: list[dict[str, object]] = []

    excluded_extra_cols = set(signal_cols).union(meta_keys).union({"time"})

    if keep_extra_phase_metadata:
        extra_meta_cols = [
            col for col in df.columns if col not in excluded_extra_cols
        ]
    else:
        extra_meta_cols = []

    groupby_keys = list(meta_keys)

    for keys, grp in df.groupby(groupby_keys, sort=True, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)

        if "time" in grp.columns:
            grp = grp.sort_values("time")
        else:
            grp = grp.copy()

        signal_values = grp[list(signal_cols)].to_numpy(dtype=np.float32)

        valid_row_mask = ~np.all(np.isnan(signal_values), axis=1)
        signal_values = signal_values[valid_row_mask]
        grp_valid = grp.loc[valid_row_mask].reset_index(drop=True)

        if len(signal_values) < window_size:
            continue

        base_meta = dict(zip(groupby_keys, keys))

        if keep_extra_phase_metadata:
            for col in extra_meta_cols:
                base_meta[col] = _safe_first_value(grp_valid[col])

        for start in range(0, len(signal_values) - window_size + 1, step_size):
            end = start + window_size
            window = signal_values[start:end]

            if np.any(np.isnan(window)):
                continue

            meta = dict(base_meta)
            meta["WindowStartIndex"] = start
            meta["WindowEndIndex"] = end - 1

            if "time" in grp_valid.columns:
                meta["WindowStartTime"] = grp_valid.loc[start, "time"]
                meta["WindowEndTime"] = grp_valid.loc[end - 1, "time"]

            windows.append(window)
            window_meta.append(meta)

    if not windows:
        raise ValueError(
            "No valid windows were created. "
            "Try checking missing values, window_size, or step_size."
        )

    return np.stack(windows), pd.DataFrame(window_meta)


def scale_windows_global(
    X_raw: np.ndarray,
) -> tuple[np.ndarray, StandardScaler]:
    """
    Standardize each signal channel globally across all windows and time points.

    Input shape:
        N x T x C

    Output shape:
        N x T x C
    """
    if X_raw.ndim != 3:
        raise ValueError("X_raw must have shape (n_windows, window_size, n_signals)")

    n_windows, window_size, n_channels = X_raw.shape

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(
        X_raw.reshape(-1, n_channels)
    ).reshape(n_windows, window_size, n_channels)

    return X_scaled.astype(np.float32), scaler


def to_conv1d_format(X: np.ndarray) -> np.ndarray:
    """
    Convert windows from:

        N x T x C

    to PyTorch Conv1D format:

        N x C x T
    """
    if X.ndim != 3:
        raise ValueError("X must have shape (n_windows, window_size, n_signals)")

    return X.transpose(0, 2, 1).astype(np.float32)


def build_autoencoder_input(
    dataset_dir: Path,
    signals: Sequence[str] = SIGNALS,
    meta_keys: Sequence[str] = META_KEYS,
    window_size: int = 60,
    step_size: int = 30,
    resample_rule: str = "1s",
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame, StandardScaler]:
    """
    Run the full data-only pipeline.

    Returns
    -------
    X_raw:
        Raw windows with shape:
            N x T x C

    X_scaled:
        Globally standardized windows with shape:
            N x T x C

    window_meta:
        One row of metadata per window.

    full_resampled_df:
        Full 1 Hz dataframe before windowing.

    scaler:
        Fitted StandardScaler.
    """
    full_resampled_df = load_raw_dataset(
        dataset_dir=dataset_dir,
        signals=signals,
        resample_rule=resample_rule,
    )

    X_raw, window_meta = create_windows(
        df=full_resampled_df,
        signal_cols=signals,
        meta_keys=meta_keys,
        window_size=window_size,
        step_size=step_size,
    )

    X_scaled, scaler = scale_windows_global(X_raw)

    return X_raw, X_scaled, window_meta, full_resampled_df, scaler


def parse_args() -> argparse.Namespace:
    """
    Optional CLI for checking preprocessing only.

    This does not save files. It only prints a summary.
    The actual reusable processed cache is saved by v1_autoencoding.py.
    """
    project_root = infer_project_root()
    default_dataset_dir = project_root / "data" / "raw" / "data" / "dataset"

    parser = argparse.ArgumentParser(
        description="Check raw EmoPairCompete time-series preprocessing."
    )

    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=default_dataset_dir,
        help="Path to the raw dataset folder.",
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

    return parser.parse_args()


def main() -> None:
    """
    Run preprocessing only and print a summary.

    No files are saved from this module.
    """
    args = parse_args()

    print(f"Loading raw dataset from: {args.dataset_dir}")

    X_raw, X_scaled, window_meta, full_df, _ = build_autoencoder_input(
        dataset_dir=args.dataset_dir,
        signals=tuple(args.signals),
        window_size=args.window_size,
        step_size=args.step_size,
        resample_rule=args.resample_rule,
    )

    print(f"Loaded 1 Hz rows: {len(full_df):,}")
    print(
        f"Created windows: {X_raw.shape[0]:,} windows x "
        f"{X_raw.shape[1]} seconds x {X_raw.shape[2]} signals"
    )
    print(f"Metadata rows: {len(window_meta):,}")

    if "Puzzler" in window_meta.columns:
        print("Puzzler values:")
        print(window_meta["Puzzler"].value_counts(dropna=False))

    if "parent" in window_meta.columns:
        print(
            "Warning: column 'parent' is still present in window metadata. "
            "It should normally be merged into 'Puzzler'."
        )

    print("No files were saved. v1_autoencoding.py handles the processed-data cache.")


if __name__ == "__main__":
    main()