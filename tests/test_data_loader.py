"""Unit tests for src.data_loader."""

import os
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from src.data_loader import (
    BIOSIGNAL_FEATURE_COLS,
    META_COLS,
    QUESTIONNAIRE_COLS,
    load_dataset,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_biosignal_row() -> dict:
    """Return a minimal dict with all biosignal feature columns set to 1.0."""
    return {col: 1.0 for col in BIOSIGNAL_FEATURE_COLS}


def _make_response_row() -> dict:
    """Return a minimal dict with all questionnaire columns set to 3."""
    return {col: 3 for col in QUESTIONNAIRE_COLS}


def _write_csv(path: Path, row: dict) -> None:
    pd.DataFrame([row]).to_csv(path, index=False)


def _build_tree(root: Path, cohort: str, participant: str, round_: str, phase: str) -> Path:
    """Create directory and CSV files, return the phase directory."""
    phase_dir = root / cohort / participant / round_ / phase
    phase_dir.mkdir(parents=True)
    _write_csv(phase_dir / "biosignal.csv", _make_biosignal_row())
    _write_csv(phase_dir / "response.csv", _make_response_row())
    return phase_dir


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestLoadDataset:
    def test_returns_dataframe(self, tmp_path):
        _build_tree(tmp_path, "D11", "P01", "1", "pre")
        df = load_dataset(tmp_path)
        assert isinstance(df, pd.DataFrame)

    def test_columns_present(self, tmp_path):
        _build_tree(tmp_path, "D11", "P01", "1", "pre")
        df = load_dataset(tmp_path)
        expected = set(META_COLS + BIOSIGNAL_FEATURE_COLS + QUESTIONNAIRE_COLS)
        assert expected.issubset(set(df.columns))

    def test_single_row(self, tmp_path):
        _build_tree(tmp_path, "D11", "P01", "1", "pre")
        df = load_dataset(tmp_path)
        assert len(df) == 1

    def test_metadata_values(self, tmp_path):
        _build_tree(tmp_path, "D12", "P03", "2", "puzzle")
        df = load_dataset(tmp_path)
        assert df.iloc[0]["cohort"] == "D12"
        assert df.iloc[0]["participant_id"] == "P03"
        assert df.iloc[0]["round"] == "2"
        assert df.iloc[0]["phase"] == "puzzle"

    def test_multiple_rows(self, tmp_path):
        _build_tree(tmp_path, "D11", "P01", "1", "pre")
        _build_tree(tmp_path, "D11", "P01", "1", "puzzle")
        _build_tree(tmp_path, "D11", "P02", "1", "pre")
        df = load_dataset(tmp_path)
        assert len(df) == 3

    def test_empty_root_returns_empty_df(self, tmp_path):
        df = load_dataset(tmp_path)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0

    def test_missing_biosignal_file_gives_nan(self, tmp_path):
        phase_dir = tmp_path / "D11" / "P01" / "1" / "pre"
        phase_dir.mkdir(parents=True)
        _write_csv(phase_dir / "response.csv", _make_response_row())
        df = load_dataset(tmp_path)
        assert len(df) == 1
        assert pd.isna(df.iloc[0]["hr_mean"])

    def test_missing_response_file_gives_nan(self, tmp_path):
        phase_dir = tmp_path / "D11" / "P01" / "1" / "pre"
        phase_dir.mkdir(parents=True)
        _write_csv(phase_dir / "biosignal.csv", _make_biosignal_row())
        df = load_dataset(tmp_path)
        assert len(df) == 1
        assert pd.isna(df.iloc[0]["frustrated"])

    def test_nonexistent_root_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_dataset(tmp_path / "does_not_exist")

    def test_biosignal_values_loaded(self, tmp_path):
        row = _make_biosignal_row()
        row["hr_mean"] = 75.5
        phase_dir = tmp_path / "D11" / "P01" / "1" / "pre"
        phase_dir.mkdir(parents=True)
        _write_csv(phase_dir / "biosignal.csv", row)
        _write_csv(phase_dir / "response.csv", _make_response_row())
        df = load_dataset(tmp_path)
        assert df.iloc[0]["hr_mean"] == pytest.approx(75.5)

    def test_questionnaire_values_loaded(self, tmp_path):
        row = _make_response_row()
        row["nervous"] = 5
        phase_dir = tmp_path / "D11" / "P01" / "1" / "puzzle"
        phase_dir.mkdir(parents=True)
        _write_csv(phase_dir / "biosignal.csv", _make_biosignal_row())
        _write_csv(phase_dir / "response.csv", row)
        df = load_dataset(tmp_path)
        assert df.iloc[0]["nervous"] == 5

    def test_extra_columns_in_csv_ignored(self, tmp_path):
        row = _make_biosignal_row()
        row["unknown_extra_col"] = 999
        phase_dir = tmp_path / "D11" / "P01" / "1" / "pre"
        phase_dir.mkdir(parents=True)
        _write_csv(phase_dir / "biosignal.csv", row)
        _write_csv(phase_dir / "response.csv", _make_response_row())
        df = load_dataset(tmp_path)
        assert "unknown_extra_col" not in df.columns
