"""
src/evaluation.py
-----------------
Computes and saves evaluation metrics for the trained intrusion-detection model.

Metrics computed
----------------
- Accuracy       : Fraction of records classified correctly.
- Precision      : Of all records predicted as attacks, how many really are?
- Recall         : Of all actual attacks, how many did we catch?
- F1-score       : Harmonic mean of Precision and Recall. Best single-number
                   summary when classes are imbalanced.
- ROC-AUC        : Area under the Receiver Operating Characteristic curve.
                   Computed from predicted *probabilities*, not hard labels.
- Confusion Matrix : [[TN, FP], [FN, TP]] — shows all four prediction outcomes.
- Classification Report : Per-class precision/recall/F1 (scikit-learn format).

Class convention
----------------
  Positive class (label=1) = Attack traffic.
  Negative class (label=0) = Normal traffic.

Terminology
-----------
  True  Positive (TP) : Attack correctly predicted as Attack.
  True  Negative (TN) : Normal correctly predicted as Normal.
  False Positive (FP) : Normal incorrectly flagged as Attack (false alarm).
  False Negative (FN) : Attack incorrectly classified as Normal (missed threat).

False Negatives are especially serious in a security context because a real
attack goes undetected. False Positives cause alert fatigue.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Public API                                                                    #
# --------------------------------------------------------------------------- #


def evaluate(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    y_proba: np.ndarray,
) -> dict:
    """Compute the full evaluation metric set.

    Parameters
    ----------
    y_true  : array-like of int
        Ground-truth labels (0 or 1).
    y_pred  : array-like of int
        Predicted labels (0 or 1) from pipeline.predict().
    y_proba : np.ndarray of shape (n_samples, 2)
        Predicted class probabilities from pipeline.predict_proba().
        Column index 1 is the attack probability.

    Returns
    -------
    dict with keys:
        accuracy, precision, recall, f1, roc_auc,
        confusion_matrix (list[list[int]]),
        classification_report (str)
    """
    attack_proba = y_proba[:, 1]  # probability of label=1 (Attack)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(
            precision_score(y_true, y_pred, pos_label=1, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, y_pred, pos_label=1, zero_division=0)
        ),
        "f1": float(
            f1_score(y_true, y_pred, pos_label=1, zero_division=0)
        ),
        "roc_auc": float(roc_auc_score(y_true, attack_proba)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(
            y_true,
            y_pred,
            target_names=["Normal (0)", "Attack (1)"],
            zero_division=0,
        ),
    }

    _log_metrics(metrics)
    return metrics


def save_metrics(metrics: dict, output_dir: str | Path) -> None:
    """Write metrics to JSON and classification report to a text file.

    Parameters
    ----------
    metrics : dict
        Output of evaluate().
    output_dir : path-like
        Directory where files will be written.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save JSON (numerical metrics only — the report is saved separately).
    json_path = output_dir / "metrics.json"
    serializable = {
        k: v for k, v in metrics.items() if k != "classification_report"
    }
    with open(json_path, "w") as fh:
        json.dump(serializable, fh, indent=2)
    logger.info("[evaluation] Metrics saved to: %s", json_path)

    # Save the text classification report.
    report_path = output_dir / "classification_report.txt"
    with open(report_path, "w") as fh:
        fh.write(metrics["classification_report"])
    logger.info("[evaluation] Classification report saved to: %s", report_path)


def save_predictions(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    y_proba: np.ndarray,
    output_dir: str | Path,
) -> None:
    """Save per-record predictions to CSV.

    Columns
    -------
    actual_label        : Ground-truth label (0 / 1).
    predicted_label     : Model prediction (0 / 1).
    predicted_label_name: 'Normal' or 'Attack'.
    attack_probability  : Predicted probability that record is an attack.

    Parameters
    ----------
    y_true, y_pred, y_proba : as in evaluate()
    output_dir : path-like
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(
        {
            "actual_label": np.array(y_true),
            "predicted_label": np.array(y_pred),
            "predicted_label_name": [
                "Attack" if p == 1 else "Normal" for p in y_pred
            ],
            "attack_probability": y_proba[:, 1],
        }
    )

    csv_path = output_dir / "predictions.csv"
    df.to_csv(csv_path, index=False)
    logger.info("[evaluation] Predictions saved to: %s", csv_path)


# --------------------------------------------------------------------------- #
# Internal helpers                                                               #
# --------------------------------------------------------------------------- #


def _log_metrics(metrics: dict) -> None:
    """Print a readable metrics summary to the logger."""
    logger.info(
        "\n"
        "  ╔══════════════════════════════════╗\n"
        "  ║        Evaluation Results        ║\n"
        "  ╠══════════════════════════════════╣\n"
        "  ║  Accuracy  : %.4f              ║\n"
        "  ║  Precision : %.4f              ║\n"
        "  ║  Recall    : %.4f              ║\n"
        "  ║  F1-score  : %.4f              ║\n"
        "  ║  ROC-AUC   : %.4f              ║\n"
        "  ╚══════════════════════════════════╝",
        metrics["accuracy"],
        metrics["precision"],
        metrics["recall"],
        metrics["f1"],
        metrics["roc_auc"],
    )
    logger.info(
        "[evaluation] Confusion Matrix (rows=Actual, cols=Predicted):\n"
        "  Labels: [Normal(0), Attack(1)]\n%s",
        np.array(metrics["confusion_matrix"]),
    )
