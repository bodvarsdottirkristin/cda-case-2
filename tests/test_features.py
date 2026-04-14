"""Unit tests for src.features."""

import numpy as np
import pandas as pd
import pytest

from src.features import (
    compute_delta_features,
    compute_ratio_features,
    get_biosignal_feature_cols,
    get_questionnaire_cols,
    select_features_variance,
)
from src.data_loader import BIOSIGNAL_FEATURE_COLS, QUESTIONNAIRE_COLS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _two_phase_df() -> pd.DataFrame:
    """Return a small DataFrame with 'pre' and 'puzzle' rows for two participants."""
    return pd.DataFrame(
        {
            "cohort": ["D11", "D11", "D11", "D11"],
            "participant_id": ["P01", "P01", "P02", "P02"],
            "round": ["1", "1", "1", "1"],
            "phase": ["pre", "puzzle", "pre", "puzzle"],
            "hr_mean": [70.0, 80.0, 65.0, 75.0],
            "eda_mean": [1.0, 3.0, 2.0, 5.0],
        }
    )


# ---------------------------------------------------------------------------
# get_biosignal_feature_cols / get_questionnaire_cols
# ---------------------------------------------------------------------------

class TestGetCols:
    def test_biosignal_cols_is_list(self):
        cols = get_biosignal_feature_cols()
        assert isinstance(cols, list)

    def test_biosignal_cols_match_constant(self):
        assert get_biosignal_feature_cols() == list(BIOSIGNAL_FEATURE_COLS)

    def test_questionnaire_cols_is_list(self):
        cols = get_questionnaire_cols()
        assert isinstance(cols, list)

    def test_questionnaire_cols_match_constant(self):
        assert get_questionnaire_cols() == list(QUESTIONNAIRE_COLS)

    def test_hr_mean_in_biosignal_cols(self):
        assert "hr_mean" in get_biosignal_feature_cols()

    def test_eda_peaks_in_biosignal_cols(self):
        assert "eda_peaks" in get_biosignal_feature_cols()

    def test_task_difficulty_in_questionnaire_cols(self):
        assert "task_difficulty" in get_questionnaire_cols()

    def test_nervous_in_questionnaire_cols(self):
        assert "nervous" in get_questionnaire_cols()


# ---------------------------------------------------------------------------
# compute_delta_features
# ---------------------------------------------------------------------------

class TestComputeDeltaFeatures:
    def test_returns_dataframe(self):
        df = _two_phase_df()
        result = compute_delta_features(df, ["hr_mean", "eda_mean"])
        assert isinstance(result, pd.DataFrame)

    def test_delta_values(self):
        df = _two_phase_df()
        result = compute_delta_features(df, ["hr_mean"])
        # P01: 80 - 70 = 10; P02: 75 - 65 = 10
        np.testing.assert_allclose(result["hr_mean_delta"].values, 10.0)

    def test_only_puzzle_phase_rows(self):
        df = _two_phase_df()
        result = compute_delta_features(df, ["hr_mean"], from_phase="pre", to_phase="puzzle")
        assert (result["phase"] == "puzzle").all()

    def test_delta_column_added(self):
        df = _two_phase_df()
        result = compute_delta_features(df, ["hr_mean", "eda_mean"])
        assert "hr_mean_delta" in result.columns
        assert "eda_mean_delta" in result.columns

    def test_no_baseline_columns_in_result(self):
        df = _two_phase_df()
        result = compute_delta_features(df, ["hr_mean"])
        assert "hr_mean_baseline" not in result.columns

    def test_missing_pre_row_gives_nan_delta(self):
        df = pd.DataFrame(
            {
                "cohort": ["D11"],
                "participant_id": ["P01"],
                "round": ["1"],
                "phase": ["puzzle"],
                "hr_mean": [80.0],
            }
        )
        result = compute_delta_features(df, ["hr_mean"])
        assert pd.isna(result.iloc[0]["hr_mean_delta"])


# ---------------------------------------------------------------------------
# compute_ratio_features
# ---------------------------------------------------------------------------

class TestComputeRatioFeatures:
    def test_returns_dataframe(self):
        df = pd.DataFrame({"a": [2.0, 4.0], "b": [1.0, 2.0]})
        result = compute_ratio_features(df, ["a"], ["b"])
        assert isinstance(result, pd.DataFrame)

    def test_ratio_values(self):
        df = pd.DataFrame({"a": [4.0, 6.0], "b": [2.0, 3.0]})
        result = compute_ratio_features(df, ["a"], ["b"])
        np.testing.assert_allclose(result["a_ratio"].values, 2.0)

    def test_zero_denominator_gives_fill_value(self):
        df = pd.DataFrame({"a": [1.0], "b": [0.0]})
        result = compute_ratio_features(df, ["a"], ["b"], fill_value=-1.0)
        assert result["a_ratio"].iloc[0] == pytest.approx(-1.0)

    def test_mismatched_lengths_raises(self):
        df = pd.DataFrame({"a": [1.0], "b": [2.0], "c": [3.0]})
        with pytest.raises(ValueError):
            compute_ratio_features(df, ["a", "b"], ["c"])

    def test_custom_suffix(self):
        df = pd.DataFrame({"x": [2.0], "y": [1.0]})
        result = compute_ratio_features(df, ["x"], ["y"], suffix="_norm")
        assert "x_norm" in result.columns

    def test_does_not_mutate_input(self):
        df = pd.DataFrame({"a": [2.0], "b": [1.0]})
        compute_ratio_features(df, ["a"], ["b"])
        assert "a_ratio" not in df.columns


# ---------------------------------------------------------------------------
# select_features_variance
# ---------------------------------------------------------------------------

class TestSelectFeaturesVariance:
    def test_returns_list(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [1, 1, 1]})
        result = select_features_variance(df, ["a", "b"])
        assert isinstance(result, list)

    def test_removes_zero_variance(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [5, 5, 5]})
        result = select_features_variance(df, ["a", "b"], threshold=0.0)
        assert "b" not in result
        assert "a" in result

    def test_threshold_filtering(self):
        df = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": [1, 1, 1, 1, 2]})
        result = select_features_variance(df, ["a", "b"], threshold=1.0)
        assert "a" in result
        assert "b" not in result

    def test_nonexistent_cols_skipped(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        result = select_features_variance(df, ["a", "nonexistent"])
        assert "a" in result
        assert "nonexistent" not in result
