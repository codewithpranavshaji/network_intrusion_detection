"""
src/visualization.py
--------------------
Generates and saves the five required diagnostic plots for the project report.

All plots are saved as high-resolution PNG files (150 DPI) in the specified
output directory. Every chart is built from real data and model outputs — no
hard-coded or fabricated values.

Plots produced
--------------
1. Class Distribution  — bar chart of Normal vs Attack counts for train & test.
2. Confusion Matrix    — heatmap of actual vs predicted labels.
3. Metrics Bar Chart   — side-by-side bar chart of all scalar metrics.
4. ROC Curve           — Receiver Operating Characteristic curve with AUC label.
5. Feature Importance  — top-15 Random Forest feature importances.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_curve

logger = logging.getLogger(__name__)

# Consistent visual style for all plots.
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 150,
    }
)

PALETTE = {"Normal (0)": "#4CAF50", "Attack (1)": "#F44336"}


# --------------------------------------------------------------------------- #
# Plot 1 — Class Distribution                                                   #
# --------------------------------------------------------------------------- #


def plot_class_distribution(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: str | Path,
) -> Path:
    """Bar chart showing Normal and Attack record counts for both partitions.

    Parameters
    ----------
    train_df, test_df : pd.DataFrame
        Raw DataFrames with a 'label' column (0 or 1).
    output_dir : path-like
        Directory to save the PNG.

    Returns
    -------
    Path to the saved PNG file.
    """
    label_col = "label"
    data = []
    for partition, df in [("Training", train_df), ("Testing", test_df)]:
        counts = df[label_col].value_counts().sort_index()
        for lbl_int, count in counts.items():
            lbl_name = "Normal (0)" if lbl_int == 0 else "Attack (1)"
            data.append({"Partition": partition, "Class": lbl_name, "Count": count})

    plot_df = pd.DataFrame(data)

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(2)
    width = 0.35
    partitions = ["Training", "Testing"]
    classes = ["Normal (0)", "Attack (1)"]
    colors = [PALETTE[c] for c in classes]

    for i, cls in enumerate(classes):
        vals = [
            plot_df[(plot_df["Partition"] == p) & (plot_df["Class"] == cls)][
                "Count"
            ].values[0]
            if len(
                plot_df[(plot_df["Partition"] == p) & (plot_df["Class"] == cls)]
            ) > 0
            else 0
            for p in partitions
        ]
        bars = ax.bar(
            x + i * width - width / 2,
            vals,
            width,
            label=cls,
            color=colors[i],
            alpha=0.85,
            edgecolor="white",
        )
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(vals) * 0.01,
                f"{v:,}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(partitions)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax.set_title("Class Distribution: Normal vs Attack", fontsize=14, fontweight="bold")
    ax.set_xlabel("Dataset Partition")
    ax.set_ylabel("Record Count")
    ax.legend(title="Class")
    plt.tight_layout()

    out = _save(fig, output_dir, "plot_class_distribution.png")
    return out


# --------------------------------------------------------------------------- #
# Plot 2 — Confusion Matrix                                                     #
# --------------------------------------------------------------------------- #


def plot_confusion_matrix(
    cm: list[list[int]] | np.ndarray,
    output_dir: str | Path,
) -> Path:
    """Annotated heatmap of the confusion matrix.

    The matrix layout is:
        Rows = Actual class   (Normal, Attack)
        Cols = Predicted class (Normal, Attack)

    Cell interpretation:
        [0,0] TN — Normal correctly predicted as Normal.
        [0,1] FP — Normal incorrectly predicted as Attack (false alarm).
        [1,0] FN — Attack incorrectly predicted as Normal (missed threat!).
        [1,1] TP — Attack correctly predicted as Attack.
    """
    cm = np.array(cm)
    labels = ["Normal (0)", "Attack (1)"]
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt=",d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        linewidths=0.5,
        ax=ax,
        cbar_kws={"shrink": 0.8},
    )
    ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("Actual Label")

    # Annotate the semantic meaning in each cell.
    cell_labels = [["TN", "FP"], ["FN", "TP"]]
    for i in range(2):
        for j in range(2):
            ax.text(
                j + 0.5,
                i + 0.75,
                f"({cell_labels[i][j]})",
                ha="center",
                va="center",
                fontsize=8,
                color="grey",
            )

    plt.tight_layout()
    return _save(fig, output_dir, "plot_confusion_matrix.png")


# --------------------------------------------------------------------------- #
# Plot 3 — Evaluation Metrics Bar Chart                                         #
# --------------------------------------------------------------------------- #


def plot_metrics_bar(
    metrics: dict,
    output_dir: str | Path,
) -> Path:
    """Horizontal bar chart comparing all scalar evaluation metrics."""
    metric_names = ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"]
    metric_keys = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    values = [metrics[k] for k in metric_keys]

    colors = [
        "#2196F3" if v >= 0.9 else "#FF9800" if v >= 0.75 else "#F44336"
        for v in values
    ]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(metric_names, values, color=colors, edgecolor="white", height=0.55)
    ax.set_xlim(0, 1.12)
    ax.set_xlabel("Score")
    ax.set_title("Model Evaluation Metrics", fontsize=14, fontweight="bold")
    ax.axvline(1.0, color="grey", linewidth=0.8, linestyle="--", alpha=0.6)

    for bar, v in zip(bars, values):
        ax.text(
            v + 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{v:.4f}",
            va="center",
            fontsize=10,
        )

    plt.tight_layout()
    return _save(fig, output_dir, "plot_metrics_bar.png")


# --------------------------------------------------------------------------- #
# Plot 4 — ROC Curve                                                            #
# --------------------------------------------------------------------------- #


def plot_roc_curve(
    y_true: np.ndarray | pd.Series,
    y_proba: np.ndarray,
    roc_auc: float,
    output_dir: str | Path,
) -> Path:
    """Receiver Operating Characteristic curve.

    The curve plots True Positive Rate vs False Positive Rate across all
    probability thresholds. Higher AUC (closer to 1.0) is better.
    The diagonal dashed line represents a random classifier (AUC = 0.5).
    """
    fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1], pos_label=1)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(
        fpr,
        tpr,
        color="#2196F3",
        linewidth=2,
        label=f"Random Forest (AUC = {roc_auc:.4f})",
    )
    ax.plot([0, 1], [0, 1], color="grey", linestyle="--", linewidth=1, label="Random Classifier")
    ax.fill_between(fpr, tpr, alpha=0.08, color="#2196F3")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate (Normal flagged as Attack)")
    ax.set_ylabel("True Positive Rate (Attacks correctly caught)")
    ax.set_title("ROC Curve — Intrusion Detection", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right")
    plt.tight_layout()
    return _save(fig, output_dir, "plot_roc_curve.png")


# --------------------------------------------------------------------------- #
# Plot 5 — Feature Importance                                                   #
# --------------------------------------------------------------------------- #


def plot_feature_importance(
    importances: np.ndarray,
    feature_names: list[str],
    output_dir: str | Path,
    top_n: int = 15,
) -> Path:
    """Horizontal bar chart of the top-N Random Forest feature importances.

    Feature importance measures how much each feature reduces impurity (Gini
    impurity) across all trees. Higher = more important for classification.
    """
    pairs = sorted(
        zip(feature_names, importances), key=lambda x: x[1], reverse=True
    )
    top_pairs = pairs[:top_n]
    names = [p[0] for p in reversed(top_pairs)]
    vals = [p[1] for p in reversed(top_pairs)]

    fig, ax = plt.subplots(figsize=(9, 7))
    bar_colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(names)))  # type: ignore[attr-defined]
    bars = ax.barh(names, vals, color=bar_colors, edgecolor="white")

    for bar, v in zip(bars, vals):
        ax.text(
            v + 0.001,
            bar.get_y() + bar.get_height() / 2,
            f"{v:.4f}",
            va="center",
            fontsize=8,
        )

    ax.set_xlabel("Mean Decrease in Impurity (Importance)")
    ax.set_title(
        f"Top {top_n} Feature Importances — Random Forest",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    return _save(fig, output_dir, "plot_feature_importance.png")


# --------------------------------------------------------------------------- #
# Helper                                                                        #
# --------------------------------------------------------------------------- #


def _save(fig: plt.Figure, output_dir: str | Path, filename: str) -> Path:
    """Save a matplotlib figure and close it to free memory."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / filename
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("[visualization] Saved: %s", out_path)
    return out_path
