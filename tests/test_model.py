"""
tests/test_model.py
-------------------
Unit tests for src/model.py.

Uses a minimal synthetic dataset (10 rows) to verify that the pipeline
builds, trains, and predicts without errors. Does NOT require the real
UNSW-NB15 dataset.
"""

import numpy as np
import pandas as pd
import pytest

from src.model import build_classifier, build_pipeline
from src.preprocessing import build_preprocessor, split_features_target


# ── Fixture ───────────────────────────────────────────────────────────────── #


def _make_pipeline_and_data():
    """Return a fitted pipeline and held-out test data on synthetic data."""
    rng = np.random.default_rng(0)
    n = 30
    df = pd.DataFrame(
        {
            "dur": rng.uniform(0, 5, n),
            "spkts": rng.integers(1, 50, n).astype(float),
            "sbytes": rng.integers(50, 5000, n).astype(float),
            "proto": rng.choice(["tcp", "udp", "icmp"], n),
            "service": rng.choice(["http", "ftp", "-"], n),
            "label": ([0, 1] * (n // 2))[:n],
        }
    )

    split = int(0.7 * n)
    train_df = df.iloc[:split].copy()
    test_df = df.iloc[split:].copy()

    X_train, y_train = split_features_target(train_df)
    X_test, y_test = split_features_target(test_df)

    preprocessor = build_preprocessor(X_train)
    pipeline = build_pipeline(preprocessor)
    pipeline.fit(X_train, y_train)

    return pipeline, X_test, y_test


# ── Tests ─────────────────────────────────────────────────────────────────── #


def test_build_classifier_type():
    """build_classifier returns a RandomForestClassifier."""
    from sklearn.ensemble import RandomForestClassifier
    clf = build_classifier()
    assert isinstance(clf, RandomForestClassifier)


def test_classifier_hyperparameters():
    """Verify baseline hyperparameter values."""
    clf = build_classifier()
    assert clf.n_estimators == 150
    assert clf.random_state == 42
    assert clf.n_jobs == -1
    assert clf.class_weight == "balanced"


def test_build_pipeline_has_two_steps():
    """Pipeline must have exactly 'preprocessor' and 'classifier' steps."""
    from sklearn.pipeline import Pipeline
    df = pd.DataFrame(
        {
            "dur": [1.0, 2.0],
            "proto": ["tcp", "udp"],
            "label": [0, 1],
        }
    )
    X, _ = split_features_target(df)
    preprocessor = build_preprocessor(X)
    pipeline = build_pipeline(preprocessor)
    assert isinstance(pipeline, Pipeline)
    assert "preprocessor" in pipeline.named_steps
    assert "classifier" in pipeline.named_steps


def test_pipeline_predict_returns_binary():
    """predict() must return integer labels of 0 or 1 only."""
    pipeline, X_test, _ = _make_pipeline_and_data()
    y_pred = pipeline.predict(X_test)
    assert set(y_pred).issubset({0, 1}), "Predictions must be 0 or 1"


def test_pipeline_predict_proba_shape():
    """predict_proba() must return (n_samples, 2)."""
    pipeline, X_test, _ = _make_pipeline_and_data()
    y_proba = pipeline.predict_proba(X_test)
    assert y_proba.shape == (len(X_test), 2), "Shape must be (n_samples, 2)"


def test_pipeline_proba_sum_to_one():
    """Each row of predict_proba should sum to approximately 1.0."""
    pipeline, X_test, _ = _make_pipeline_and_data()
    y_proba = pipeline.predict_proba(X_test)
    row_sums = y_proba.sum(axis=1)
    np.testing.assert_allclose(row_sums, 1.0, atol=1e-6)


def test_pipeline_predict_count_matches_input():
    """Number of predictions equals number of input rows."""
    pipeline, X_test, _ = _make_pipeline_and_data()
    y_pred = pipeline.predict(X_test)
    assert len(y_pred) == len(X_test)


def test_save_and_load_model(tmp_path):
    """Saved model can be reloaded and makes identical predictions."""
    from src.model import save_model, load_model

    pipeline, X_test, _ = _make_pipeline_and_data()
    model_path = tmp_path / "test_model.joblib"
    save_model(pipeline, model_path)

    assert model_path.exists(), "Model file should be created on disk"

    loaded = load_model(model_path)
    y_pred_original = pipeline.predict(X_test)
    y_pred_loaded = loaded.predict(X_test)
    np.testing.assert_array_equal(
        y_pred_original, y_pred_loaded, err_msg="Loaded model must produce same predictions"
    )


def test_load_model_raises_if_missing():
    """load_model raises FileNotFoundError for a non-existent path."""
    from src.model import load_model
    with pytest.raises(FileNotFoundError):
        load_model("outputs/nonexistent_model.joblib")
