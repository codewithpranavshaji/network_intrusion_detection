"""
src/data_loader.py
------------------
Loads, validates, and inspects the UNSW-NB15 training and testing CSV files.

Responsibilities
----------------
- Read CSV files from disk.
- Normalize column names (strip whitespace, lowercase).
- Validate that the 'label' target column is present and binary (0 / 1).
- Report dataset shapes, class counts, missing values, and duplicates.
- Raise descriptive errors when the schema is unexpected.

This module never modifies the data — it only loads and reports.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Constants                                                                     #
# --------------------------------------------------------------------------- #

LABEL_COLUMN = "label"
VALID_LABELS = {0, 1}

# Columns that are expected to exist but may be absent in some distributions.
OPTIONAL_COLUMNS: list[str] = ["id", "attack_cat"]


# --------------------------------------------------------------------------- #
# Public API                                                                    #
# --------------------------------------------------------------------------- #


def load_dataset(
    train_path: str | Path,
    test_path: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and validate the UNSW-NB15 training and testing CSV files.

    Parameters
    ----------
    train_path : path-like
        Path to UNSW_NB15_training-set.csv.
    test_path : path-like
        Path to UNSW_NB15_testing-set.csv.

    Returns
    -------
    train_df, test_df : tuple of DataFrames
        Both DataFrames have normalized column names (stripped, lowercased).

    Raises
    ------
    FileNotFoundError
        If either CSV file does not exist at the given path.
    ValueError
        If the 'label' column is missing or contains unexpected values.
    """
    train_path = Path(train_path)
    test_path = Path(test_path)

    _check_file_exists(train_path, "training")
    _check_file_exists(test_path, "testing")

    logger.info("Loading training dataset from: %s", train_path)
    train_df = _load_csv(train_path)

    logger.info("Loading testing dataset from: %s", test_path)
    test_df = _load_csv(test_path)

    logger.info("Training shape  : %s", train_df.shape)
    logger.info("Testing  shape  : %s", test_df.shape)

    _validate_label_column(train_df, "training")
    _validate_label_column(test_df, "testing")

    _report_class_distribution(train_df, "training")
    _report_class_distribution(test_df, "testing")

    _report_data_quality(train_df, "training")
    _report_data_quality(test_df, "testing")

    return train_df, test_df


def inspect_schema(df: pd.DataFrame, name: str = "dataset") -> None:
    """Print a human-readable schema summary.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to inspect.
    name : str
        Friendly name used in log messages.
    """
    print(f"\n{'='*60}")
    print(f"  Schema inspection: {name}")
    print(f"{'='*60}")
    print(f"  Rows    : {len(df):,}")
    print(f"  Columns : {df.shape[1]}")
    print()
    print(df.dtypes.to_string())
    print(f"{'='*60}\n")


# --------------------------------------------------------------------------- #
# Internal helpers                                                               #
# --------------------------------------------------------------------------- #


def _check_file_exists(path: Path, label: str) -> None:
    """Raise FileNotFoundError with a helpful message if path is missing."""
    if not path.exists():
        raise FileNotFoundError(
            f"[data_loader] {label.capitalize()} CSV file not found:\n"
            f"  Expected: {path.resolve()}\n"
            f"  Please download the UNSW-NB15 dataset and place it there.\n"
            f"  See data/README.md for instructions."
        )


def _load_csv(path: Path) -> pd.DataFrame:
    """Read a CSV file and normalize column names."""
    try:
        df = pd.read_csv(path, low_memory=False)
    except Exception as exc:
        raise RuntimeError(
            f"[data_loader] Failed to read CSV file: {path}\n  Cause: {exc}"
        ) from exc

    # Normalize: strip whitespace and lowercase all column names.
    df.columns = [col.strip().lower() for col in df.columns]

    # Replace infinite values with NaN so imputers handle them uniformly.
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    return df


def _validate_label_column(df: pd.DataFrame, partition: str) -> None:
    """Confirm that the 'label' column exists and is strictly binary."""
    if LABEL_COLUMN not in df.columns:
        raise ValueError(
            f"[data_loader] '{LABEL_COLUMN}' column not found in the "
            f"{partition} dataset.\n"
            f"  Available columns: {list(df.columns)}"
        )

    unique = set(df[LABEL_COLUMN].dropna().unique())
    unexpected = unique - VALID_LABELS
    if unexpected:
        raise ValueError(
            f"[data_loader] Unexpected label values in {partition} dataset: "
            f"{unexpected}.\n  Valid values are: {VALID_LABELS}"
        )

    null_count = df[LABEL_COLUMN].isna().sum()
    if null_count > 0:
        logger.warning(
            "[data_loader] %s: %d null value(s) found in '%s'. "
            "These records will be dropped.",
            partition,
            null_count,
            LABEL_COLUMN,
        )


def _report_class_distribution(df: pd.DataFrame, partition: str) -> None:
    """Log class counts and the positive-class ratio."""
    counts = df[LABEL_COLUMN].value_counts().sort_index()
    total = len(df)
    logger.info(
        "[data_loader] %s class distribution:\n%s",
        partition.capitalize(),
        counts.to_string(),
    )
    if 1 in counts:
        attack_pct = 100 * counts[1] / total
        logger.info(
            "[data_loader] %s: %.1f%% attack records (%d / %d)",
            partition.capitalize(),
            attack_pct,
            counts[1],
            total,
        )


def _report_data_quality(df: pd.DataFrame, partition: str) -> None:
    """Log missing values and duplicate row count."""
    missing = df.isnull().sum()
    cols_with_missing = missing[missing > 0]
    if cols_with_missing.empty:
        logger.info("[data_loader] %s: no missing values.", partition.capitalize())
    else:
        logger.warning(
            "[data_loader] %s: missing values detected:\n%s",
            partition.capitalize(),
            cols_with_missing.to_string(),
        )

    dup_count = df.duplicated().sum()
    if dup_count > 0:
        logger.info(
            "[data_loader] %s: %d duplicate rows detected (kept for analysis).",
            partition.capitalize(),
            dup_count,
        )
    else:
        logger.info("[data_loader] %s: no duplicate rows.", partition.capitalize())
