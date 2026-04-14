"""Unit tests for src.preprocessing."""

import numpy as np
import pandas as pd
import pytest

from src.preprocessing import (
    PHASE_ORDER,
    drop_missing,
    encode_phase,
    impute_missing,
    normalise_features,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "cohort": ["D11", "D11", "D12"],
            "participant_id": ["P01", "P01", "P02"],
            "round": ["1", "1", "2"],
            "phase": ["pre", "puzzle", "post"],
            "hr_mean": [70.0, np.nan, 80.0],
            "eda_mean": [1.0, 2.0, np.nan],
            "temp_mean": [36.5, 36.7, 36.9],
        }
    )


# ---------------------------------------------------------------------------
# drop_missing
# ---------------------------------------------------------------------------

class TestDropMissing:
    def test_returns_dataframe(self):
        df = _sample_df()
        result = drop_missing(df)
        assert isinstance(result, pd.DataFrame)

    def test_does_not_mutate_input(self):
        df = _sample_df()
        original_nan_count = df.isna().sum().sum()
        drop_missing(df)
        assert df.isna().sum().sum() == original_nan_count

    def test_drops_high_missing_col(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [np.nan, np.nan, np.nan]})
        result = drop_missing(df, col_threshold=0.5)
        assert "b" not in result.columns

    def test_drops_high_missing_row(self):
        df = pd.DataFrame({"a": [1, np.nan], "b": [2, np.nan], "c": [3, np.nan]})
        result = drop_missing(df, row_threshold=0.5)
        assert len(result) == 1

    def test_keeps_acceptable_missing(self):
        df = _sample_df()
        result = drop_missing(df, row_threshold=0.9, col_threshold=0.9)
        assert len(result) == len(df)


# ---------------------------------------------------------------------------
# impute_missing
# ---------------------------------------------------------------------------

class TestImputeMissing:
    def test_median_imputation(self):
        df = pd.DataFrame({"x": [1.0, np.nan, 3.0]})
        result = impute_missing(df, strategy="median")
        assert result["x"].isna().sum() == 0
        assert result["x"].iloc[1] == pytest.approx(2.0)

    def test_mean_imputation(self):
        df = pd.DataFrame({"x": [2.0, np.nan, 4.0]})
        result = impute_missing(df, strategy="mean")
        assert result["x"].iloc[1] == pytest.approx(3.0)

    def test_zero_imputation(self):
        df = pd.DataFrame({"x": [1.0, np.nan, 3.0]})
        result = impute_missing(df, strategy="zero")
        assert result["x"].iloc[1] == 0.0

    def test_invalid_strategy_raises(self):
        df = pd.DataFrame({"x": [1.0, np.nan]})
        with pytest.raises(ValueError):
            impute_missing(df, strategy="unknown")

    def test_non_numeric_columns_untouched(self):
        df = pd.DataFrame({"label": ["a", None], "value": [1.0, np.nan]})
        result = impute_missing(df, strategy="zero", numeric_only=True)
        assert pd.isna(result["label"].iloc[1])
        assert result["value"].iloc[1] == 0.0

    def test_does_not_mutate_input(self):
        df = pd.DataFrame({"x": [1.0, np.nan]})
        impute_missing(df)
        assert df["x"].isna().sum() == 1


# ---------------------------------------------------------------------------
# encode_phase
# ---------------------------------------------------------------------------

class TestEncodePhase:
    def test_adds_encoded_column(self):
        df = pd.DataFrame({"phase": ["pre", "puzzle", "post"]})
        result = encode_phase(df)
        assert "phase_encoded" in result.columns

    def test_correct_encoding(self):
        df = pd.DataFrame({"phase": ["pre", "puzzle", "post"]})
        result = encode_phase(df)
        assert result["phase_encoded"].tolist() == [0, 1, 2]

    def test_phase_order_constant(self):
        assert PHASE_ORDER["pre"] < PHASE_ORDER["puzzle"] < PHASE_ORDER["post"]

    def test_does_not_mutate_input(self):
        df = pd.DataFrame({"phase": ["pre"]})
        encode_phase(df)
        assert "phase_encoded" not in df.columns


# ---------------------------------------------------------------------------
# normalise_features
# ---------------------------------------------------------------------------

class TestNormaliseFeatures:
    def test_zero_mean(self):
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})
        result = normalise_features(df, ["x"])
        assert result["x"].mean() == pytest.approx(0.0, abs=1e-10)

    def test_unit_std(self):
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})
        result = normalise_features(df, ["x"])
        assert result["x"].std(ddof=0) == pytest.approx(1.0, abs=1e-6)

    def test_missing_cols_skipped(self):
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
        result = normalise_features(df, ["x", "nonexistent"])
        assert "nonexistent" not in result.columns

    def test_does_not_mutate_input(self):
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
        normalise_features(df, ["x"])
        assert df["x"].iloc[0] == 1.0
