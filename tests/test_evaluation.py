"""
tests/test_evaluation.py
------------------------
Unit tests for src/evaluation.py.

Uses hard-coded or synthetically generated y_true / y_pred / y_proba arrays
— no real dataset required.
"""

import numpy as np
import pytest

from src.evaluation import evaluate, save_metrics, save_predictions


# ── Fixtures ──────────────────────────────────────────────────────────────── #


def _perfect_inputs():
    """Return label arrays where predictions are 100% correct."""
    y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1])
    y_pred = np.array([0, 0, 1, 1, 0, 1, 0, 1])
    y_proba = np.column_stack([1 - y_pred, y_pred]).astype(float)
    return y_true, y_pred, y_proba


def _imperfect_inputs():
    """Return label arrays with some misclassifications."""
    y_true  = np.array([0, 0, 1, 1, 0, 1, 0, 1, 1, 0])
    y_pred  = np.array([0, 1, 1, 0, 0, 1, 1, 1, 0, 0])  # some wrong
    # Simulate blended probabilities.
    y_proba = np.column_stack([
        [0.9, 0.3, 0.1, 0.8, 0.7, 0.05, 0.4, 0.15, 0.6, 0.85],
        [0.1, 0.7, 0.9, 0.2, 0.3, 0.95, 0.6, 0.85, 0.4, 0.15],
    ])
    return y_true, y_pred, y_proba


# ── Tests: evaluate() ─────────────────────────────────────────────────────── #


def test_evaluate_returns_all_required_keys():
    """evaluate() must return a dict containing all required metric keys."""
    y_true, y_pred, y_proba = _imperfect_inputs()
    metrics = evaluate(y_true, y_pred, y_proba)
    required_keys = {
        "accuracy", "precision", "recall", "f1",
        "roc_auc", "confusion_matrix", "classification_report",
    }
    missing = required_keys - set(metrics.keys())
    assert not missing, f"Missing metric keys: {missing}"


def test_evaluate_perfect_predictions():
    """Perfect predictions should yield accuracy, precision, recall, F1 = 1.0."""
    y_true, y_pred, y_proba = _perfect_inputs()
    metrics = evaluate(y_true, y_pred, y_proba)
    assert metrics["accuracy"] == pytest.approx(1.0)
    assert metrics["precision"] == pytest.approx(1.0)
    assert metrics["recall"] == pytest.approx(1.0)
    assert metrics["f1"] == pytest.approx(1.0)
    assert metrics["roc_auc"] == pytest.approx(1.0)


def test_evaluate_scores_between_0_and_1():
    """All scalar metrics must be in [0, 1]."""
    y_true, y_pred, y_proba = _imperfect_inputs()
    metrics = evaluate(y_true, y_pred, y_proba)
    for key in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        assert 0.0 <= metrics[key] <= 1.0, f"{key} out of valid range"


def test_confusion_matrix_shape():
    """Confusion matrix must be 2×2 for binary classification."""
    y_true, y_pred, y_proba = _imperfect_inputs()
    metrics = evaluate(y_true, y_pred, y_proba)
    cm = np.array(metrics["confusion_matrix"])
    assert cm.shape == (2, 2), "Confusion matrix must be (2, 2)"


def test_confusion_matrix_sums_to_total():
    """Confusion matrix cell sum must equal total number of test records."""
    y_true, y_pred, y_proba = _imperfect_inputs()
    metrics = evaluate(y_true, y_pred, y_proba)
    cm = np.array(metrics["confusion_matrix"])
    assert cm.sum() == len(y_true)


def test_classification_report_is_string():
    """classification_report value must be a non-empty string."""
    y_true, y_pred, y_proba = _imperfect_inputs()
    metrics = evaluate(y_true, y_pred, y_proba)
    assert isinstance(metrics["classification_report"], str)
    assert len(metrics["classification_report"]) > 0


# ── Tests: save_metrics() ─────────────────────────────────────────────────── #


def test_save_metrics_creates_json(tmp_path):
    """save_metrics writes a valid metrics.json file."""
    import json
    y_true, y_pred, y_proba = _imperfect_inputs()
    metrics = evaluate(y_true, y_pred, y_proba)
    save_metrics(metrics, tmp_path)

    json_file = tmp_path / "metrics.json"
    assert json_file.exists(), "metrics.json should be created"
    with open(json_file) as fh:
        loaded = json.load(fh)
    assert "accuracy" in loaded


def test_save_metrics_creates_report(tmp_path):
    """save_metrics writes a classification_report.txt file."""
    y_true, y_pred, y_proba = _imperfect_inputs()
    metrics = evaluate(y_true, y_pred, y_proba)
    save_metrics(metrics, tmp_path)
    report_file = tmp_path / "classification_report.txt"
    assert report_file.exists(), "classification_report.txt should be created"


# ── Tests: save_predictions() ─────────────────────────────────────────────── #


def test_save_predictions_creates_csv(tmp_path):
    """save_predictions writes a predictions.csv file with expected columns."""
    import pandas as pd
    y_true, y_pred, y_proba = _imperfect_inputs()
    save_predictions(y_true, y_pred, y_proba, tmp_path)

    csv_file = tmp_path / "predictions.csv"
    assert csv_file.exists(), "predictions.csv should be created"
    df = pd.read_csv(csv_file)
    assert "actual_label" in df.columns
    assert "predicted_label" in df.columns
    assert "attack_probability" in df.columns
    assert len(df) == len(y_true)


# ── Tests: leakage guard ──────────────────────────────────────────────────── #


def test_attack_cat_not_in_features():
    """
    Confirm that 'attack_cat' is in EXCLUDED_COLUMNS and therefore never
    reaches the model as a feature — this is the primary leakage prevention
    contract.
    """
    from src.preprocessing import EXCLUDED_COLUMNS
    assert "attack_cat" in EXCLUDED_COLUMNS, (
        "'attack_cat' must be excluded from model features to prevent leakage"
    )


def test_id_not_in_features():
    """'id' must be listed in EXCLUDED_COLUMNS."""
    from src.preprocessing import EXCLUDED_COLUMNS
    assert "id" in EXCLUDED_COLUMNS, (
        "'id' must be excluded — it is a record identifier, not a network feature"
    )
