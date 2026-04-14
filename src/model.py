"""
model.py
========
Modelling utilities for stress / affect prediction on the EmoPairCompete
dataset.

Functions
---------
split_data(df, target_col, feature_cols, test_size, random_state)
    Stratified train/test split.
train_random_forest(X_train, y_train, **kwargs)
    Fit a RandomForestClassifier.
evaluate_classifier(clf, X_test, y_test)
    Return a dict with accuracy, precision, recall, f1, and roc_auc.
train_logistic_regression(X_train, y_train, **kwargs)
    Fit a LogisticRegression model.
cross_validate_model(clf, X, y, cv, scoring)
    Return cross-validated scores.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split


def split_data(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: Iterable[str],
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified train / test split.

    Parameters
    ----------
    df:
        Input DataFrame.
    target_col:
        Name of the target column.
    feature_cols:
        Feature column names.
    test_size:
        Fraction of samples to use for the test set.
    random_state:
        Random seed for reproducibility.

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    feature_cols = list(feature_cols)
    X = df[feature_cols]
    y = df[target_col]
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_estimators: int = 100,
    random_state: int = 42,
    **kwargs: Any,
) -> RandomForestClassifier:
    """Fit a RandomForestClassifier.

    Parameters
    ----------
    X_train:
        Training features.
    y_train:
        Training labels.
    n_estimators:
        Number of trees (default 100).
    random_state:
        Random seed.
    **kwargs:
        Additional keyword arguments forwarded to RandomForestClassifier.

    Returns
    -------
    RandomForestClassifier
        Fitted classifier.
    """
    clf = RandomForestClassifier(
        n_estimators=n_estimators, random_state=random_state, **kwargs
    )
    clf.fit(X_train, y_train)
    return clf


def train_logistic_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    max_iter: int = 1000,
    random_state: int = 42,
    **kwargs: Any,
) -> LogisticRegression:
    """Fit a LogisticRegression model.

    Parameters
    ----------
    X_train:
        Training features.
    y_train:
        Training labels.
    max_iter:
        Maximum number of iterations (default 1000).
    random_state:
        Random seed.
    **kwargs:
        Additional keyword arguments forwarded to LogisticRegression.

    Returns
    -------
    LogisticRegression
        Fitted model.
    """
    clf = LogisticRegression(max_iter=max_iter, random_state=random_state, **kwargs)
    clf.fit(X_train, y_train)
    return clf


def evaluate_classifier(
    clf: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    average: str = "weighted",
) -> Dict[str, float]:
    """Evaluate a fitted classifier and return a metrics dictionary.

    Parameters
    ----------
    clf:
        Fitted scikit-learn classifier.
    X_test:
        Test features.
    y_test:
        True labels.
    average:
        Averaging strategy for precision / recall / f1 (default ``"weighted"``).

    Returns
    -------
    dict
        Keys: ``accuracy``, ``precision``, ``recall``, ``f1``, ``roc_auc``
        (roc_auc uses ``predict_proba`` when available, otherwise NaN).
    """
    y_pred = clf.predict(X_test)

    roc_auc: float = float("nan")
    if hasattr(clf, "predict_proba"):
        y_proba = clf.predict_proba(X_test)
        classes = np.unique(y_test)
        if len(classes) == 2:
            roc_auc = roc_auc_score(y_test, y_proba[:, 1])
        else:
            try:
                roc_auc = roc_auc_score(
                    y_test, y_proba, multi_class="ovr", average=average
                )
            except ValueError:
                pass

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average=average, zero_division=0),
        "recall": recall_score(y_test, y_pred, average=average, zero_division=0),
        "f1": f1_score(y_test, y_pred, average=average, zero_division=0),
        "roc_auc": roc_auc,
    }


def cross_validate_model(
    clf: Any,
    X: pd.DataFrame,
    y: pd.Series,
    cv: int = 5,
    scoring: str = "f1_weighted",
    random_state: int = 42,
) -> Dict[str, Any]:
    """Return cross-validated scores.

    Parameters
    ----------
    clf:
        A scikit-learn estimator (not yet fitted).
    X:
        Feature matrix.
    y:
        Target vector.
    cv:
        Number of folds (default 5).
    scoring:
        Scoring metric string (default ``"f1_weighted"``).
    random_state:
        Random seed for the StratifiedKFold splitter.

    Returns
    -------
    dict
        Keys: ``mean``, ``std``, ``scores`` (array of per-fold scores).
    """
    cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    scores = cross_val_score(clf, X, y, cv=cv_splitter, scoring=scoring)
    return {"mean": float(scores.mean()), "std": float(scores.std()), "scores": scores}
