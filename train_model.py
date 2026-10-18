"""
train_model.py
--------------
Main entry-point script for the Network Intrusion Detection ML project.

What this script does (in order)
---------------------------------
1.  Loads UNSW-NB15 training and testing CSV files.
2.  Validates the dataset schema and binary labels.
3.  Separates features (X) from the target label (y).
    - Excludes 'id' (record identifier — not a network feature).
    - Excludes 'attack_cat' (leakage — reveals the attack type).
4.  Builds a preprocessing pipeline (median imputation + OneHotEncoding).
5.  Builds a Random Forest classifier pipeline.
6.  Trains the full pipeline on training data ONLY.
7.  Predicts on the held-out testing set.
8.  Evaluates accuracy, precision, recall, F1, ROC-AUC, confusion matrix.
9.  Saves the trained pipeline (model artifact).
10. Saves metrics, predictions, and all five diagnostic plots to outputs/.

Usage
-----
    python train_model.py

Requirements
------------
- Dataset files must be placed in:
    data/UNSW_NB15_training-set.csv
    data/UNSW_NB15_testing-set.csv
  See data/README.md for download instructions.
- Run: pip install -r requirements.txt
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import numpy as np

# ── Logging setup ────────────────────────────────────────────────────────── #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ── Project paths ────────────────────────────────────────────────────────── #
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

TRAIN_CSV = DATA_DIR / "UNSW_NB15_training-set.csv"
TEST_CSV = DATA_DIR / "UNSW_NB15_testing-set.csv"
MODEL_PATH = OUTPUT_DIR / "intrusion_detection_model.joblib"

# ── Source modules ───────────────────────────────────────────────────────── #
from src.data_loader import load_dataset, inspect_schema
from src.evaluation import evaluate, save_metrics, save_predictions
from src.model import build_pipeline, save_model, train_pipeline
from src.preprocessing import build_preprocessor, get_feature_names, split_features_target
from src.visualization import (
    plot_class_distribution,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_metrics_bar,
    plot_roc_curve,
)


# --------------------------------------------------------------------------- #
# Main pipeline                                                                 #
# --------------------------------------------------------------------------- #


def main() -> None:
    """Run the complete training, evaluation, and output pipeline."""
    t_start = time.time()
    logger.info("=" * 60)
    logger.info("  Network Intrusion Detection — Training Pipeline")
    logger.info("=" * 60)

    # ── Phase 1: Load data ──────────────────────────────────────────────── #
    logger.info("\n[Phase 1] Loading datasets …")
    train_df, test_df = load_dataset(TRAIN_CSV, TEST_CSV)
    inspect_schema(train_df, "Training")
    inspect_schema(test_df, "Testing")

    # ── Phase 2: Preprocessing ──────────────────────────────────────────── #
    logger.info("\n[Phase 2] Splitting features and target …")
    X_train, y_train = split_features_target(train_df)
    X_test, y_test = split_features_target(test_df)

    logger.info(
        "X_train: %s | y_train distribution: %s",
        X_train.shape,
        y_train.value_counts().to_dict(),
    )
    logger.info(
        "X_test : %s | y_test  distribution: %s",
        X_test.shape,
        y_test.value_counts().to_dict(),
    )

    # Build and fit preprocessor on training data only.
    logger.info("\n[Phase 2] Building preprocessing pipeline …")
    preprocessor = build_preprocessor(X_train)

    # ── Phase 3: Build & train pipeline ────────────────────────────────── #
    logger.info("\n[Phase 3] Building full sklearn Pipeline …")
    pipeline = build_pipeline(preprocessor)

    logger.info("[Phase 3] Training Random Forest (this may take a minute) …")
    t_train = time.time()
    pipeline = train_pipeline(pipeline, X_train, y_train)
    logger.info("[Phase 3] Training completed in %.1f seconds.", time.time() - t_train)

    # ── Phase 4: Evaluate ───────────────────────────────────────────────── #
    logger.info("\n[Phase 4] Evaluating on held-out test set …")
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)

    metrics = evaluate(y_test, y_pred, y_proba)

    # ── Phase 5: Save outputs ───────────────────────────────────────────── #
    logger.info("\n[Phase 5] Saving model artifact …")
    save_model(pipeline, MODEL_PATH)

    logger.info("[Phase 5] Saving metrics and predictions …")
    save_metrics(metrics, OUTPUT_DIR)
    save_predictions(y_test, y_pred, y_proba, OUTPUT_DIR)

    # ── Phase 6: Visualizations ─────────────────────────────────────────── #
    logger.info("\n[Phase 6] Generating plots …")

    plot_class_distribution(train_df, test_df, OUTPUT_DIR)
    plot_confusion_matrix(metrics["confusion_matrix"], OUTPUT_DIR)
    plot_metrics_bar(metrics, OUTPUT_DIR)
    plot_roc_curve(
        np.array(y_test), y_proba, metrics["roc_auc"], OUTPUT_DIR
    )

    # Feature importances — extract from the fitted classifier step.
    feature_names = get_feature_names(
        pipeline.named_steps["preprocessor"]
    )
    importances = pipeline.named_steps["classifier"].feature_importances_
    plot_feature_importance(importances, feature_names, OUTPUT_DIR, top_n=15)

    # ── Summary ─────────────────────────────────────────────────────────── #
    elapsed = time.time() - t_start
    logger.info("\n" + "=" * 60)
    logger.info("  Training pipeline completed in %.1f seconds.", elapsed)
    logger.info("=" * 60)
    logger.info("  Model    : %s", MODEL_PATH)
    logger.info("  Outputs  : %s", OUTPUT_DIR.resolve())
    logger.info("")
    logger.info("  ── Evaluation Summary ──────────────────────────────────")
    logger.info("  Accuracy  : %.4f", metrics["accuracy"])
    logger.info("  Precision : %.4f", metrics["precision"])
    logger.info("  Recall    : %.4f", metrics["recall"])
    logger.info("  F1-score  : %.4f", metrics["f1"])
    logger.info("  ROC-AUC   : %.4f", metrics["roc_auc"])
    logger.info("=" * 60)
    logger.info("  Run 'streamlit run app.py' to launch the prediction UI.")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
