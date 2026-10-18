"""
demo_visualizations.py
-----------------------
Generates all 5 project plots using realistic synthetic data.
Run this to preview the visualizations before training on the real dataset.

    python demo_visualizations.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import seaborn as sns
from sklearn.metrics import roc_curve

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

rng = np.random.default_rng(42)

# ── Realistic synthetic metrics (typical RF on UNSW-NB15) ──────────────────
METRICS = {
    "accuracy": 0.9743,
    "precision": 0.9681,
    "recall": 0.9802,
    "f1": 0.9741,
    "roc_auc": 0.9921,
}

# ── Confusion matrix (scaled to ~82k test records) ─────────────────────────
#  TN=54210, FP=1390, FN=740, TP=25992   (roughly 43k normal / 27k attack)
CM = np.array([[54210, 1390],
               [740,  25992]])

# ── Class counts ────────────────────────────────────────────────────────────
TRAIN_NORMAL, TRAIN_ATTACK = 109854, 45509
TEST_NORMAL,  TEST_ATTACK  = 55600,  26732

# ── Feature names & importances ─────────────────────────────────────────────
FEATURE_NAMES = [
    "sbytes", "dbytes", "sttl", "dttl", "ct_srv_src",
    "ct_dst_ltm", "ct_src_ltm", "ct_src_dport_ltm", "ct_dst_sport_ltm",
    "ct_dst_src_ltm", "dur", "sinpkt", "dinpkt", "smean", "dmean",
    "ct_state_ttl", "spkts", "dpkts", "proto_tcp", "proto_udp",
]
raw_imp = rng.exponential(scale=[0.12, 0.10, 0.08, 0.07, 0.06,
                                  0.055, 0.05, 0.048, 0.045, 0.042,
                                  0.038, 0.035, 0.030, 0.028, 0.025,
                                  0.022, 0.018, 0.015, 0.010, 0.008])
IMPORTANCES = raw_imp / raw_imp.sum()

# ── Synthetic ROC curve ─────────────────────────────────────────────────────
n = 30000
y_true_roc = rng.integers(0, 2, n)
noise = rng.normal(0, 0.08, n)
y_score = np.clip(y_true_roc * 0.88 + (1 - y_true_roc) * 0.10 + noise, 0, 1)
fpr, tpr, _ = roc_curve(y_true_roc, y_score)


# ────────────────────────────────────────────────────────────────────────────
def plot1_class_dist():
    data = [
        ("Training", "Normal (0)", TRAIN_NORMAL),
        ("Training", "Attack (1)", TRAIN_ATTACK),
        ("Testing",  "Normal (0)", TEST_NORMAL),
        ("Testing",  "Attack (1)", TEST_ATTACK),
    ]
    partitions = ["Training", "Testing"]
    classes = ["Normal (0)", "Attack (1)"]
    colors = {"Normal (0)": "#4CAF50", "Attack (1)": "#F44336"}
    x = np.arange(2)
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, cls in enumerate(classes):
        vals = [d[2] for d in data if d[0] == partitions[j] and d[1] == cls
                for j in range(2)]
        vals = [next(d[2] for d in data if d[0] == p and d[1] == cls) for p in partitions]
        bars = ax.bar(x + i * width - width / 2, vals, width,
                      label=cls, color=colors[cls], alpha=0.85, edgecolor="white")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 800,
                    f"{v:,}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(partitions)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax.set_title("Class Distribution: Normal vs Attack", fontsize=14, fontweight="bold")
    ax.set_xlabel("Dataset Partition")
    ax.set_ylabel("Record Count")
    ax.legend(title="Class")
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "plot_class_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("✅  plot_class_distribution.png")


def plot2_confusion_matrix():
    labels = ["Normal (0)", "Attack (1)"]
    cell_labels = [["TN", "FP"], ["FN", "TP"]]
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(CM, annot=True, fmt=",d", cmap="Blues",
                xticklabels=labels, yticklabels=labels,
                linewidths=0.5, ax=ax, cbar_kws={"shrink": 0.8})
    ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("Actual Label")
    for i in range(2):
        for j in range(2):
            ax.text(j + 0.5, i + 0.75, f"({cell_labels[i][j]})",
                    ha="center", va="center", fontsize=8, color="grey")
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "plot_confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("✅  plot_confusion_matrix.png")


def plot3_metrics_bar():
    names  = ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"]
    keys   = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    values = [METRICS[k] for k in keys]
    colors = ["#2196F3" if v >= 0.9 else "#FF9800" for v in values]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(names, values, color=colors, edgecolor="white", height=0.55)
    ax.set_xlim(0, 1.12)
    ax.set_xlabel("Score")
    ax.set_title("Model Evaluation Metrics", fontsize=14, fontweight="bold")
    ax.axvline(1.0, color="grey", linewidth=0.8, linestyle="--", alpha=0.6)
    for bar, v in zip(bars, values):
        ax.text(v + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{v:.4f}", va="center", fontsize=10)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "plot_metrics_bar.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("✅  plot_metrics_bar.png")


def plot4_roc_curve():
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color="#2196F3", linewidth=2,
            label=f"Random Forest (AUC = {METRICS['roc_auc']:.4f})")
    ax.plot([0, 1], [0, 1], color="grey", linestyle="--", linewidth=1,
            label="Random Classifier")
    ax.fill_between(fpr, tpr, alpha=0.08, color="#2196F3")
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.05])
    ax.set_xlabel("False Positive Rate (Normal flagged as Attack)")
    ax.set_ylabel("True Positive Rate (Attacks correctly caught)")
    ax.set_title("ROC Curve — Intrusion Detection", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right")
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "plot_roc_curve.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("✅  plot_roc_curve.png")


def plot5_feature_importance():
    pairs = sorted(zip(FEATURE_NAMES, IMPORTANCES), key=lambda x: x[1], reverse=True)
    top = pairs[:15]
    names_r = [p[0] for p in reversed(top)]
    vals_r  = [p[1] for p in reversed(top)]
    colors  = plt.cm.Blues(np.linspace(0.4, 0.9, len(names_r)))

    fig, ax = plt.subplots(figsize=(9, 7))
    bars = ax.barh(names_r, vals_r, color=colors, edgecolor="white")
    for bar, v in zip(bars, vals_r):
        ax.text(v + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{v:.4f}", va="center", fontsize=8)
    ax.set_xlabel("Mean Decrease in Impurity (Importance)")
    ax.set_title("Top 15 Feature Importances — Random Forest",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "plot_feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("✅  plot_feature_importance.png")


if __name__ == "__main__":
    print("\n  Generating demo visualizations …\n")
    plot1_class_dist()
    plot2_confusion_matrix()
    plot3_metrics_bar()
    plot4_roc_curve()
    plot5_feature_importance()
    # Also write a demo metrics.json
    with open(OUTPUT_DIR / "metrics.json", "w") as f:
        json.dump(METRICS, f, indent=2)
    print("\n  All plots saved to outputs/")
    print("  Note: these use synthetic data. Run python train_model.py for real results.\n")
