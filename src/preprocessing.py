"""
src/preprocessing.py
--------------------
Builds a reproducible, leakage-free preprocessing pipeline for the UNSW-NB15
dataset and the Random Forest classifier.

Design principles
-----------------
- All transformations are **fitted only on training data**.
- The 'label' target column is never passed to the model.
- 'id' (record identifier) and 'attack_cat' (attack category — directly
  reveals ground truth) are excluded before any transformation.
- Numeric missing values → median imputation.
- Categorical missing values → most-frequent imputation, then OneHotEncoded
  with unknown-category handling so prediction on unseen data never fails.
- Infinite values must be replaced with NaN before this module is called
  (data_loader.py handles this on load).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Constants — columns that must NEVER enter the model                          #
# --------------------------------------------------------------------------- #

LABEL_COLUMN: str = "label"

# 'id'         : record identifier — carries no signal, just row number.
# 'attack_cat' : encodes the attack category name. Passing it to the model
#                would be data leakage — it directly reveals the answer.
EXCLUDED_COLUMNS: list[str] = ["id", "attack_cat"]


# --------------------------------------------------------------------------- #
# Public API                                                                    #
# --------------------------------------------------------------------------- #


def split_features_target(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """Separate feature columns from the target label.

    The following columns are removed before returning X:
    - LABEL_COLUMN   ('label')
    - EXCLUDED_COLUMNS ('id', 'attack_cat') if present

    Parameters
    ----------
    df : pd.DataFrame
        A loaded (and column-normalized) dataset DataFrame.

    Returns
    -------
    X : pd.DataFrame
        Feature columns only.
    y : pd.Series
        Binary target (0 = Normal, 1 = Attack).
    """
    if LABEL_COLUMN not in df.columns:
        raise ValueError(
            f"[preprocessing] Target column '{LABEL_COLUMN}' not found.\n"
            f"  Available columns: {list(df.columns)}"
        )

    drop_cols = [LABEL_COLUMN] + [c for c in EXCLUDED_COLUMNS if c in df.columns]
    excluded_found = [c for c in EXCLUDED_COLUMNS if c in df.columns]
    if excluded_found:
        logger.info(
            "[preprocessing] Excluding columns from model input: %s", excluded_found
        )

    y = df[LABEL_COLUMN].astype(int)
    X = df.drop(columns=drop_cols)

    logger.info(
        "[preprocessing] Features: %d columns | Target: '%s'",
        X.shape[1],
        LABEL_COLUMN,
    )
    return X, y


def build_preprocessor(X_train: pd.DataFrame) -> ColumnTransformer:
    """Build (but do NOT fit) a ColumnTransformer for the given feature set.

    The transformer handles two branches:
    - Numeric  : median imputation (robust to outliers).
    - Categorical : most-frequent imputation → OneHotEncoder
                    (handle_unknown='ignore' prevents failures at prediction time
                    when an unseen category value appears).

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features (used only to detect column data types).

    Returns
    -------
    ColumnTransformer (unfitted)
    """
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X_train.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    logger.info(
        "[preprocessing] Numeric features  : %d  |  Categorical features: %d",
        len(numeric_cols),
        len(categorical_cols),
    )
    if categorical_cols:
        logger.info("[preprocessing] Categorical columns: %s", categorical_cols)

    numeric_transformer = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            )
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    transformers: list[tuple[str, Any, list[str]]] = []
    if numeric_cols:
        transformers.append(("numeric", numeric_transformer, numeric_cols))
    if categorical_cols:
        transformers.append(("categorical", categorical_transformer, categorical_cols))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",  # drop any column not explicitly handled
        verbose_feature_names_out=False,
    )

    return preprocessor


def get_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """Return the output feature names after the ColumnTransformer is fitted.

    This powers the feature-importance plot, which needs readable names for
    the one-hot-encoded columns.

    Parameters
    ----------
    preprocessor : ColumnTransformer
        A **fitted** ColumnTransformer.

    Returns
    -------
    list[str]
        One name per output column (e.g., 'proto_tcp', 'sbytes', …).
    """
    try:
        return list(preprocessor.get_feature_names_out())
    except AttributeError:
        # Fallback for very old scikit-learn versions.
        names: list[str] = []
        for name, trans, cols in preprocessor.transformers_:
            if hasattr(trans, "get_feature_names_out"):
                names.extend(trans.get_feature_names_out(cols))
            else:
                names.extend(cols)
        return names


def validate_prediction_input(
    df: pd.DataFrame,
    expected_columns: list[str],
) -> pd.DataFrame:
    """Align an incoming prediction DataFrame to the expected feature schema.

    - Drops 'label', 'id', 'attack_cat' if present (never feed ground truth
      into the classifier during inference).
    - Adds columns that are missing (filled with NaN so the imputer handles
      them).
    - Reorders columns to match the training schema.
    - Logs a warning for each missing column.

    Parameters
    ----------
    df : pd.DataFrame
        Raw input DataFrame (column names should already be normalized).
    expected_columns : list[str]
        The feature column names the fitted pipeline was trained on.

    Returns
    -------
    pd.DataFrame
        A DataFrame with exactly the columns in `expected_columns`, in order.
    """
    # Remove target / leakage columns if the user uploaded a labelled file.
    leakage_cols = [LABEL_COLUMN] + EXCLUDED_COLUMNS
    found_leakage = [c for c in leakage_cols if c in df.columns]
    if found_leakage:
        logger.info(
            "[preprocessing] Dropping target/leakage columns from input: %s",
            found_leakage,
        )
        df = df.drop(columns=found_leakage)

    missing = [c for c in expected_columns if c not in df.columns]
    if missing:
        logger.warning(
            "[preprocessing] %d expected feature(s) missing from input — "
            "filling with NaN. These will be imputed.\n  Missing: %s",
            len(missing),
            missing,
        )
        for col in missing:
            df[col] = np.nan

    # Select and reorder.
    return df[expected_columns]
