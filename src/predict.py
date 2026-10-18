"""
src/predict.py
--------------
Provides a reusable prediction function for the trained intrusion-detection
pipeline. Accepts a DataFrame of network traffic records and returns predicted
labels together with attack probability scores.

Design notes
------------
- This module is used by both train_model.py (batch evaluation) and app.py
  (Streamlit upload interface).
- It validates that the input contains at least the columns the model was
  trained on (after excluding id / attack_cat / label).
- Missing required columns are filled with NaN and imputed by the pipeline.
- Labels, identifiers, and attack-category columns in uploaded files are
  never passed to the classifier as features.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.model import load_model
from src.preprocessing import validate_prediction_input

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Public API                                                                    #
# --------------------------------------------------------------------------- #


def predict_from_df(
    pipeline,
    df: pd.DataFrame,
) -> dict:
    """Run the trained pipeline on a feature DataFrame and return predictions.

    The function automatically:
    - Drops 'label', 'id', 'attack_cat' if they appear in the input.
    - Aligns the remaining columns to the schema the model was trained on.
    - Fills any gap columns with NaN (imputed internally by the pipeline).

    Parameters
    ----------
    pipeline : sklearn.pipeline.Pipeline (fitted)
        The trained full pipeline (preprocessor + RandomForestClassifier).
    df : pd.DataFrame
        Network traffic records. Column names must be normalized (lowercase,
        stripped). Use the same normalization as in data_loader.py.

    Returns
    -------
    dict with keys:
        y_pred          : np.ndarray of int (0 = Normal, 1 = Attack).
        y_proba         : np.ndarray of shape (n_samples, 2).
                          Column 0 = P(Normal), Column 1 = P(Attack).
        attack_scores   : np.ndarray — shorthand for y_proba[:, 1].
        label_names     : list[str] — 'Normal' or 'Attack' per record.

    Raises
    ------
    ValueError
        If the DataFrame is empty or has no usable feature columns.
    """
    if df.empty:
        raise ValueError("[predict] Input DataFrame is empty. Nothing to predict.")

    # Retrieve the feature names the preprocessor was trained on.
    expected_cols = _get_expected_columns(pipeline)
    if not expected_cols:
        raise ValueError(
            "[predict] Could not determine expected feature columns from pipeline."
        )

    # Align the input to the training schema (drops leakage cols, fills gaps).
    X = validate_prediction_input(df.copy(), expected_cols)

    logger.info(
        "[predict] Running prediction on %d records, %d features.",
        len(X),
        X.shape[1],
    )

    y_pred = pipeline.predict(X)
    y_proba = pipeline.predict_proba(X)

    attack_scores = y_proba[:, 1]
    label_names = ["Attack" if p == 1 else "Normal" for p in y_pred]

    attack_count = int(np.sum(y_pred == 1))
    normal_count = int(np.sum(y_pred == 0))
    logger.info(
        "[predict] Results — Normal: %d | Attack: %d (%.1f%% attack rate)",
        normal_count,
        attack_count,
        100 * attack_count / len(y_pred),
    )

    return {
        "y_pred": y_pred,
        "y_proba": y_proba,
        "attack_scores": attack_scores,
        "label_names": label_names,
    }


def predict_from_file(
    model_path: str | Path,
    input_path: str | Path,
) -> pd.DataFrame:
    """Convenience function: load a model and predict from a CSV file.

    Parameters
    ----------
    model_path  : path-like — path to the saved .joblib pipeline artifact.
    input_path  : path-like — path to a CSV of network traffic records.

    Returns
    -------
    pd.DataFrame with columns:
        predicted_label, predicted_label_name, attack_probability
        (plus all original input columns)
    """
    import numpy as np  # local import avoids top-level duplication

    model_path = Path(model_path)
    input_path = Path(input_path)

    if not input_path.exists():
        raise FileNotFoundError(f"[predict] Input file not found: {input_path}")

    pipeline = load_model(model_path)
    df = pd.read_csv(input_path, low_memory=False)
    df.columns = [c.strip().lower() for c in df.columns]
    df.replace([np.inf, -np.inf], np.nan, inplace=True)  # type: ignore[arg-type]

    results = predict_from_df(pipeline, df)

    output_df = df.copy()
    output_df["predicted_label"] = results["y_pred"]
    output_df["predicted_label_name"] = results["label_names"]
    output_df["attack_probability"] = results["attack_scores"]

    return output_df


# --------------------------------------------------------------------------- #
# Internal helpers                                                               #
# --------------------------------------------------------------------------- #


def _get_expected_columns(pipeline) -> list[str]:
    """Extract the list of feature column names the preprocessor was trained on.

    The ColumnTransformer stores the original feature names in its
    ``feature_names_in_`` attribute after fitting. We use this to know what
    columns the incoming DataFrame should have.
    """
    try:
        preprocessor = pipeline.named_steps["preprocessor"]
        return list(preprocessor.feature_names_in_)
    except (AttributeError, KeyError):
        # Fallback: try to get names from inner transformers.
        try:
            preprocessor = pipeline.named_steps["preprocessor"]
            cols: list[str] = []
            for _, _, c in preprocessor.transformers_:
                cols.extend(c)
            return cols
        except Exception:
            return []
