"""Unit tests for src.model."""

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification

from src.model import (
    cross_validate_model,
    evaluate_classifier,
    split_data,
    train_logistic_regression,
    train_random_forest,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _classification_df(n_samples: int = 100, n_features: int = 5, random_state: int = 0):
    X_arr, y_arr = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=3,
        n_redundant=0,
        n_classes=2,
        random_state=random_state,
    )
    feat_cols = [f"f{i}" for i in range(n_features)]
    df = pd.DataFrame(X_arr, columns=feat_cols)
    df["label"] = y_arr
    return df, feat_cols


# ---------------------------------------------------------------------------
# split_data
# ---------------------------------------------------------------------------

class TestSplitData:
    def test_returns_four_objects(self):
        df, feat_cols = _classification_df()
        result = split_data(df, "label", feat_cols)
        assert len(result) == 4

    def test_sizes(self):
        df, feat_cols = _classification_df(n_samples=100)
        X_train, X_test, y_train, y_test = split_data(df, "label", feat_cols, test_size=0.2)
        assert len(X_test) == 20
        assert len(X_train) == 80

    def test_no_target_col_in_features(self):
        df, feat_cols = _classification_df()
        X_train, X_test, _, _ = split_data(df, "label", feat_cols)
        assert "label" not in X_train.columns


# ---------------------------------------------------------------------------
# train_random_forest
# ---------------------------------------------------------------------------

class TestTrainRandomForest:
    def test_returns_fitted_classifier(self):
        df, feat_cols = _classification_df()
        X_train, X_test, y_train, y_test = split_data(df, "label", feat_cols)
        clf = train_random_forest(X_train, y_train)
        assert hasattr(clf, "predict")

    def test_predict_shape(self):
        df, feat_cols = _classification_df()
        X_train, X_test, y_train, y_test = split_data(df, "label", feat_cols)
        clf = train_random_forest(X_train, y_train)
        preds = clf.predict(X_test)
        assert preds.shape == (len(X_test),)

    def test_n_estimators_respected(self):
        df, feat_cols = _classification_df()
        X_train, _, y_train, _ = split_data(df, "label", feat_cols)
        clf = train_random_forest(X_train, y_train, n_estimators=10)
        assert clf.n_estimators == 10


# ---------------------------------------------------------------------------
# train_logistic_regression
# ---------------------------------------------------------------------------

class TestTrainLogisticRegression:
    def test_returns_fitted_model(self):
        df, feat_cols = _classification_df()
        X_train, _, y_train, _ = split_data(df, "label", feat_cols)
        clf = train_logistic_regression(X_train, y_train)
        assert hasattr(clf, "coef_")

    def test_predict_shape(self):
        df, feat_cols = _classification_df()
        X_train, X_test, y_train, _ = split_data(df, "label", feat_cols)
        clf = train_logistic_regression(X_train, y_train)
        preds = clf.predict(X_test)
        assert preds.shape == (len(X_test),)


# ---------------------------------------------------------------------------
# evaluate_classifier
# ---------------------------------------------------------------------------

class TestEvaluateClassifier:
    def test_returns_dict_with_expected_keys(self):
        df, feat_cols = _classification_df()
        X_train, X_test, y_train, y_test = split_data(df, "label", feat_cols)
        clf = train_random_forest(X_train, y_train)
        metrics = evaluate_classifier(clf, X_test, y_test)
        for key in ("accuracy", "precision", "recall", "f1", "roc_auc"):
            assert key in metrics

    def test_accuracy_in_range(self):
        df, feat_cols = _classification_df()
        X_train, X_test, y_train, y_test = split_data(df, "label", feat_cols)
        clf = train_random_forest(X_train, y_train)
        metrics = evaluate_classifier(clf, X_test, y_test)
        assert 0.0 <= metrics["accuracy"] <= 1.0

    def test_roc_auc_in_range(self):
        df, feat_cols = _classification_df()
        X_train, X_test, y_train, y_test = split_data(df, "label", feat_cols)
        clf = train_random_forest(X_train, y_train)
        metrics = evaluate_classifier(clf, X_test, y_test)
        if not np.isnan(metrics["roc_auc"]):
            assert 0.0 <= metrics["roc_auc"] <= 1.0


# ---------------------------------------------------------------------------
# cross_validate_model
# ---------------------------------------------------------------------------

class TestCrossValidateModel:
    def test_returns_dict_with_expected_keys(self):
        df, feat_cols = _classification_df(n_samples=50)
        X = df[feat_cols]
        y = df["label"]
        clf = train_random_forest(X, y, n_estimators=5)
        result = cross_validate_model(clf, X, y, cv=3)
        for key in ("mean", "std", "scores"):
            assert key in result

    def test_mean_in_range(self):
        df, feat_cols = _classification_df(n_samples=60)
        X = df[feat_cols]
        y = df["label"]
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(n_estimators=5, random_state=0)
        result = cross_validate_model(clf, X, y, cv=3)
        assert 0.0 <= result["mean"] <= 1.0

    def test_scores_length(self):
        df, feat_cols = _classification_df(n_samples=60)
        X = df[feat_cols]
        y = df["label"]
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(n_estimators=5, random_state=0)
        result = cross_validate_model(clf, X, y, cv=3)
        assert len(result["scores"]) == 3
