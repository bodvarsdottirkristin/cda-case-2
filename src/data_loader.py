"""
data_loader.py
==============
Walk the raw-data directory tree and return a unified DataFrame with one row
per (cohort, participant, round, phase) combination.

Expected directory layout
-------------------------
raw_root/
  <cohort>/         e.g. D11, D12, D13 … D16
    <participant>/  e.g. P01, P02 …
      <round>/      e.g. 1, 2, 3, 4
        <phase>/    e.g. pre, puzzle, post
          biosignal.csv
          response.csv

biosignal.csv columns (one row, feature values)
------------------------------------------------
For each signal in {hr, temp, eda, eda_phasic, eda_tonic}:
  <signal>_mean, <signal>_max, <signal>_min, <signal>_std,
  <signal>_kurtosis, <signal>_skew, <signal>_slope, <signal>_auc
EDA peak metrics:
  eda_peaks, eda_rise_time, eda_recovery_time

response.csv columns (one row, questionnaire scores)
-----------------------------------------------------
frustrated, upset, hostile, alert, ashamed, inspired, nervous,
determined, attentive, afraid, active, task_difficulty
"""

from __future__ import annotations

import os
import logging
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

BIOSIGNAL_SIGNALS = ["hr", "temp", "eda", "eda_phasic", "eda_tonic"]
BIOSIGNAL_STATS = ["mean", "max", "min", "std", "kurtosis", "skew", "slope", "auc"]
EDA_PEAK_COLS = ["eda_peaks", "eda_rise_time", "eda_recovery_time"]

QUESTIONNAIRE_COLS = [
    "frustrated",
    "upset",
    "hostile",
    "alert",
    "ashamed",
    "inspired",
    "nervous",
    "determined",
    "attentive",
    "afraid",
    "active",
    "task_difficulty",
]

BIOSIGNAL_FEATURE_COLS = (
    [f"{sig}_{stat}" for sig in BIOSIGNAL_SIGNALS for stat in BIOSIGNAL_STATS]
    + EDA_PEAK_COLS
)

META_COLS = ["cohort", "participant_id", "round", "phase"]


def _read_single_csv(path: Path) -> Optional[pd.Series]:
    """Read a single-row CSV and return it as a Series, or None on failure."""
    try:
        df = pd.read_csv(path)
        if df.empty:
            logger.warning("Empty file: %s", path)
            return None
        return df.iloc[0]
    except Exception as exc:
        logger.warning("Could not read %s: %s", path, exc)
        return None


def load_dataset(raw_root: str | os.PathLike) -> pd.DataFrame:
    """Walk *raw_root* and return a unified DataFrame.

    Parameters
    ----------
    raw_root:
        Path to the top-level directory that contains cohort sub-folders.

    Returns
    -------
    pd.DataFrame
        Columns: ``cohort``, ``participant_id``, ``round``, ``phase``,
        all biosignal feature columns, all questionnaire columns.
        Rows with missing files are included with NaN values for the
        corresponding columns.
    """
    raw_root = Path(raw_root)
    if not raw_root.is_dir():
        raise FileNotFoundError(f"raw_root not found: {raw_root}")

    records: list[dict] = []

    for cohort_dir in sorted(raw_root.iterdir()):
        if not cohort_dir.is_dir():
            continue
        cohort = cohort_dir.name

        for participant_dir in sorted(cohort_dir.iterdir()):
            if not participant_dir.is_dir():
                continue
            participant_id = participant_dir.name

            for round_dir in sorted(participant_dir.iterdir()):
                if not round_dir.is_dir():
                    continue
                round_label = round_dir.name

                for phase_dir in sorted(round_dir.iterdir()):
                    if not phase_dir.is_dir():
                        continue
                    phase = phase_dir.name

                    record: dict = {
                        "cohort": cohort,
                        "participant_id": participant_id,
                        "round": round_label,
                        "phase": phase,
                    }

                    biosignal_path = phase_dir / "biosignal.csv"
                    biosignal_row = _read_single_csv(biosignal_path)
                    if biosignal_row is not None:
                        for col in BIOSIGNAL_FEATURE_COLS:
                            record[col] = biosignal_row.get(col, float("nan"))
                    else:
                        for col in BIOSIGNAL_FEATURE_COLS:
                            record[col] = float("nan")

                    response_path = phase_dir / "response.csv"
                    response_row = _read_single_csv(response_path)
                    if response_row is not None:
                        for col in QUESTIONNAIRE_COLS:
                            record[col] = response_row.get(col, float("nan"))
                    else:
                        for col in QUESTIONNAIRE_COLS:
                            record[col] = float("nan")

                    records.append(record)

    all_cols = META_COLS + BIOSIGNAL_FEATURE_COLS + QUESTIONNAIRE_COLS
    if not records:
        return pd.DataFrame(columns=all_cols)

    df = pd.DataFrame(records, columns=all_cols)
    logger.info("Loaded %d rows from %s", len(df), raw_root)
    return df
