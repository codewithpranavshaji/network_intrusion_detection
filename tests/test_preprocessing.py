"""
tests/test_preprocessing.py
----------------------------
Unit tests for src/preprocessing.py.

All tests use small synthetic DataFrames — the real UNSW-NB15 dataset is NOT
required. Each test validates a specific aspect of the preprocessing contract.
"""

import numpy as np
import pandas as pd
import pytest

from src.preprocessing import (
    EXCLUDED_COLUMNS,
    LABEL_COLUMN,
    build_preprocessor,
    get_feature_names,
    split_features_target,
    validate_prediction_input,
)


# ── Fixtures ──────────────────────────────────────────────────────────────── #


def _make_df(
    n: int = 10,
    include_id: bool = True,
    include_attack_cat: bool = True,
    label_values: list = None,
) -> pd.DataFrame:
    """Create a synthetic DataFrame that mimics UNSW-NB15 columns."""
    rng = np.random.default_rng(42)
    if label_values is None:
        label_values = ([0, 1] * ((n // 2) + 1))[:n]

    df = pd.DataFrame(
        {
            "dur": rng.uniform(0, 10, n),
            "spkts": rng.integers(1, 100, n).astype(float),
            "sbytes": rng.integers(100, 10000, n).astype(float),
            "proto": rng.choice(["tcp", "udp", "icmp"], n),
            "service": rng.choice(["http", "ftp", "-"], n),
            "state": rng.choice(["FIN", "CON", "INT"], n),
            "label": label_values[:n],
        }
    )
    if include_id:
        df.insert(0, "id", range(1, n + 1))
    if include_attack_cat:
        df["attack_cat"] = rng.choice(["Normal", "DoS", "Probe"], n)
    return df


# ── Tests: split_features_target ─────────────────────────────────────────── #


def test_label_removed_from_X():
    """'label' column must not appear in X."""
    df = _make_df()
    X, y = split_features_target(df)
    assert LABEL_COLUMN not in X.columns, "label must not appear in features"


def test_id_excluded_from_X():
    """'id' column must be excluded from X."""
    df = _make_df(include_id=True)
    X, _ = split_features_target(df)
    assert "id" not in X.columns, "'id' must be excluded (record identifier)"


def test_attack_cat_excluded_from_X():
    """'attack_cat' must be excluded — it leaks the attack type."""
    df = _make_df(include_attack_cat=True)
    X, _ = split_features_target(df)
    assert "attack_cat" not in X.columns, (
        "'attack_cat' must be excluded to prevent target leakage"
    )


def test_y_is_binary():
    """Target series should only contain 0 and 1."""
    df = _make_df()
    _, y = split_features_target(df)
    assert set(y.unique()).issubset({0, 1}), "Labels must be binary (0 or 1)"


def test_split_raises_if_label_missing():
    """split_features_target should raise ValueError if 'label' is missing."""
    df = _make_df().drop(columns=["label"])
    with pytest.raises(ValueError, match="label"):
        split_features_target(df)


# ── Tests: build_preprocessor ─────────────────────────────────────────────  #


def test_build_preprocessor_returns_column_transformer():
    """build_preprocessor returns a ColumnTransformer object."""
    from sklearn.compose import ColumnTransformer

    df = _make_df()
    X, _ = split_features_target(df)
    preprocessor = build_preprocessor(X)
    assert isinstance(preprocessor, ColumnTransformer)


def test_preprocessor_fits_and_transforms():
    """ColumnTransformer fits on training data and transforms it."""
    df = _make_df(n=20)
    X, _ = split_features_target(df)
    preprocessor = build_preprocessor(X)
    preprocessor.fit(X)
    transformed = preprocessor.transform(X)
    assert transformed.shape[0] == len(X), "Row count must be preserved"
    assert transformed.shape[1] > 0, "Output must have at least one column"


def test_missing_numeric_imputed_by_median():
    """Median imputation fills NaN in numeric columns."""
    df = _make_df(n=20)
    X, _ = split_features_target(df)
    # Introduce missing values in a numeric column.
    X = X.copy()
    X.loc[0, "dur"] = np.nan
    preprocessor = build_preprocessor(X)
    preprocessor.fit(X)
    result = preprocessor.transform(X)
    assert not np.isnan(result).any(), "No NaN should remain after imputation"


def test_missing_categorical_imputed():
    """Most-frequent imputation fills NaN in categorical columns."""
    df = _make_df(n=20)
    X, _ = split_features_target(df)
    X = X.copy()
    X.loc[0, "proto"] = np.nan
    preprocessor = build_preprocessor(X)
    preprocessor.fit(X)
    result = preprocessor.transform(X)
    assert not np.isnan(result).any(), "No NaN after categorical imputation"


def test_unknown_category_does_not_raise():
    """handle_unknown='ignore' prevents errors for unseen categories at predict time."""
    df_train = _make_df(n=20)
    X_train, _ = split_features_target(df_train)
    preprocessor = build_preprocessor(X_train)
    preprocessor.fit(X_train)

    # Test record has a protocol never seen in training.
    df_test = df_train.copy()
    df_test.loc[0, "proto"] = "UNKNOWN_PROTOCOL_XYZ"
    X_test, _ = split_features_target(df_test)
    result = preprocessor.transform(X_test)  # must not raise
    assert result.shape[0] == len(X_test)


# ── Tests: validate_prediction_input ─────────────────────────────────────── #


def test_validate_drops_leakage_columns():
    """validate_prediction_input must strip label / id / attack_cat."""
    df = _make_df(n=5)
    X, _ = split_features_target(df)
    expected = list(X.columns)
    # Simulate an uploaded labelled file.
    df_upload = _make_df(n=5)
    result = validate_prediction_input(df_upload, expected)
    for col in [LABEL_COLUMN] + EXCLUDED_COLUMNS:
        assert col not in result.columns, f"'{col}' must be stripped from input"


def test_validate_fills_missing_columns():
    """Missing expected columns are filled with NaN rather than raising."""
    df = _make_df(n=5)
    X, _ = split_features_target(df)
    expected = list(X.columns)
    # Remove one expected column to simulate missing feature.
    partial = X.drop(columns=["dur"])
    result = validate_prediction_input(partial, expected)
    assert "dur" in result.columns, "Missing column should be added back"
    assert result["dur"].isna().all(), "Re-added column should be all NaN"
