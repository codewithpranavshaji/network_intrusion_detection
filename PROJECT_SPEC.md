# Network Intrusion Detection Using Machine Learning — Project Specification

## 1. Problem Statement

Network intrusion detection is the task of automatically identifying malicious
activity in network traffic. This project builds a binary classifier that labels
each network traffic record as either **Normal (0)** or **Attack (1)**, using
features derived from individual network connections.

The system is designed for **offline, dataset-based evaluation** — it is NOT a
live packet-capture or network-scanning system.

---

## 2. Dataset

| Property | Value |
|---|---|
| Name | UNSW-NB15 |
| Source | https://research.unsw.edu.au/projects/unsw-nb15-dataset |
| Training records | ~175,341 |
| Testing records | ~82,332 |
| Target column | `label` (0 = Normal, 1 = Attack) |
| Feature types | Numeric + Categorical (mixed) |

### Excluded columns

| Column | Reason |
|---|---|
| `id` | Record identifier — no predictive signal |
| `attack_cat` | Directly encodes the attack category (data leakage) |

---

## 3. Algorithm

**Random Forest Classifier** (`sklearn.ensemble.RandomForestClassifier`)

| Hyperparameter | Value | Rationale |
|---|---|---|
| `n_estimators` | 150 | Enough trees for stability without excessive training time |
| `random_state` | 42 | Reproducibility |
| `n_jobs` | -1 | Use all available CPU cores |
| `class_weight` | `balanced` | Compensates for class imbalance (normal >> attack) |

---

## 4. Preprocessing Pipeline

Built with `sklearn.compose.ColumnTransformer`:

| Step | Numeric columns | Categorical columns |
|---|---|---|
| Missing values | `SimpleImputer(strategy='median')` | `SimpleImputer(strategy='most_frequent')` |
| Encoding | Pass-through | `OneHotEncoder(handle_unknown='ignore')` |

**Critical rule:** All transformations are **fitted on training data only**.
The same fitted transformer is applied to the testing data without re-fitting.

---

## 5. Evaluation Metrics

| Metric | Description |
|---|---|
| Accuracy | Overall fraction of correct predictions |
| Precision | Of predicted attacks, fraction that are real attacks |
| Recall | Of all real attacks, fraction that were detected |
| F1-score | Harmonic mean of Precision and Recall |
| ROC-AUC | Area under the ROC curve (computed from probabilities) |

**Positive class = 1 (Attack)**

---

## 6. Output Files

All outputs are written to `outputs/`:

| File | Description |
|---|---|
| `intrusion_detection_model.joblib` | Trained pipeline artifact |
| `metrics.json` | Scalar evaluation metrics |
| `classification_report.txt` | Per-class precision / recall / F1 |
| `predictions.csv` | Per-record predictions with attack probability |
| `plot_class_distribution.png` | Class balance visualization |
| `plot_confusion_matrix.png` | Confusion matrix heatmap |
| `plot_metrics_bar.png` | Evaluation metrics bar chart |
| `plot_roc_curve.png` | ROC curve with AUC |
| `plot_feature_importance.png` | Top-15 feature importances |

---

## 7. Security Constraints

- Dataset CSV contents are treated as untrusted data (no code execution from CSV).
- The model artifact is only loaded from the path created by this project.
- No live packet capture, external network access, or offensive functionality.
- No secrets or credentials are stored in any source file.
- No external APIs are called during training or prediction.
