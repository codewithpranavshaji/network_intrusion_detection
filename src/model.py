"""
src/model.py
------------
Wraps the scikit-learn RandomForestClassifier inside a full sklearn Pipeline
(preprocessor + classifier) and provides save/load utilities.

Why Random Forest?
------------------
- Handles mixed numeric/categorical data well (after encoding).
- Robust to outliers and class imbalance (class_weight='balanced').
- Provides feature importances, which we use for visualization.
- Interpretable for academic presentations.
- State-of-the-art performance on tabular intrusion-detection benchmarks.

The full Pipeline is saved as a single artifact so that prediction on new
records receives the same transformations as training — no manual preprocessing
step is required by the caller.
"""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Model factory                                                                 #
# --------------------------------------------------------------------------- #


def build_classifier() -> RandomForestClassifier:
    """Return a baseline RandomForestClassifier.

    Hyperparameter notes
    --------------------
    n_estimators=150    : 150 decision trees in the forest. More trees →
                          more stable predictions but slower training.
    random_state=42     : Fixed seed for reproducibility across runs.
    n_jobs=-1           : Use all available CPU cores during training.
    class_weight='balanced' : Adjusts class weights inversely proportional
                          to class frequency. Important here because normal
                          traffic often outnumbers attacks, and we do not
                          want the model to simply predict 'normal' for every
                          record.
    """
    return RandomForestClassifier(
        n_estimators=150,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )


def build_pipeline(preprocessor: ColumnTransformer) -> Pipeline:
    """Combine the preprocessor and classifier into a single sklearn Pipeline.

    Parameters
    ----------
    preprocessor : ColumnTransformer
        The (unfitted) preprocessing transformer from preprocessing.py.

    Returns
    -------
    sklearn.pipeline.Pipeline
        A Pipeline with steps: [('preprocessor', …), ('classifier', …)]
    """
    classifier = build_classifier()
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )
    return pipeline


# --------------------------------------------------------------------------- #
# Training                                                                      #
# --------------------------------------------------------------------------- #


def train_pipeline(
    pipeline: Pipeline,
    X_train,
    y_train,
) -> Pipeline:
    """Fit the complete pipeline (preprocessor + classifier) on training data.

    Parameters
    ----------
    pipeline : sklearn.pipeline.Pipeline
        Unfitted pipeline returned by build_pipeline().
    X_train : pd.DataFrame
        Training feature columns (label and excluded columns already removed).
    y_train : pd.Series
        Binary training labels (0 = Normal, 1 = Attack).

    Returns
    -------
    Pipeline (fitted)
    """
    logger.info(
        "[model] Starting training — %d samples, %d features.",
        len(X_train),
        X_train.shape[1],
    )
    pipeline.fit(X_train, y_train)
    logger.info("[model] Training complete.")
    return pipeline


# --------------------------------------------------------------------------- #
# Persistence                                                                   #
# --------------------------------------------------------------------------- #


def save_model(pipeline: Pipeline, path: str | Path) -> None:
    """Persist the fitted pipeline to disk using joblib.

    Parameters
    ----------
    pipeline : Pipeline (fitted)
        The trained full pipeline.
    path : path-like
        Output file path (e.g., 'outputs/intrusion_detection_model.joblib').
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)
    logger.info("[model] Pipeline saved to: %s", path.resolve())


def load_model(path: str | Path) -> Pipeline:
    """Load a persisted pipeline from disk.

    Parameters
    ----------
    path : path-like
        Path to a .joblib file produced by save_model().

    Returns
    -------
    Pipeline (fitted)

    Raises
    ------
    FileNotFoundError
        If the model file does not exist at the given path.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"[model] Model artifact not found: {path.resolve()}\n"
            "  Run 'python train_model.py' first to produce the model."
        )
    logger.info("[model] Loading pipeline from: %s", path.resolve())
    return joblib.load(path)
