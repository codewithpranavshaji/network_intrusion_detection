# Network Intrusion Detection Using Machine Learning

## Academic Report

**Student Project — Binary Classification with Random Forest**  
Dataset: UNSW-NB15 | Algorithm: Random Forest Classifier | Language: Python 3

---

## Abstract

Network intrusion detection is a critical discipline in cybersecurity concerned
with automatically identifying unauthorized or malicious activity within computer
networks. This project implements a supervised machine-learning-based intrusion
detection system (IDS) using the UNSW-NB15 benchmark dataset and a Random Forest
classifier. The system classifies individual network connection records as either
normal (label = 0) or attack (label = 1). A complete preprocessing pipeline
handles missing values, categorical encoding, and target-leakage prevention.
The trained model is evaluated using accuracy, precision, recall, F1-score, and
ROC-AUC. All five required diagnostic visualizations are generated. An optional
Streamlit interface provides an interactive prediction experience.

**Keywords:** Network Intrusion Detection, Machine Learning, Random Forest,
UNSW-NB15, Binary Classification, Cybersecurity, Supervised Learning.

---

## 1. Introduction

The rapid growth of internet-connected devices has made network security a
pressing concern. Traditional rule-based intrusion detection systems rely on
manually maintained signatures that cannot adapt to new attack patterns. Machine
learning offers a data-driven alternative: by learning statistical patterns from
labelled network traffic, an ML model can generalize to previously unseen attack
variants.

This project demonstrates a complete ML pipeline — from raw dataset loading to
trained model deployment — suitable as an academic proof-of-concept for
network intrusion detection.

---

## 2. Problem Statement

**Classification task:** Given a set of features derived from a single network
connection record, predict whether that connection represents:
- **Label 0** — Normal (benign) network traffic.
- **Label 1** — Attack (malicious) network traffic.

**Constraints:**
- This is an offline system: it classifies records from a static dataset.
- It does NOT capture live packets, scan networks, or perform offensive actions.
- High recall (catching real attacks) is prioritized over precision, because a
  missed attack (False Negative) is more dangerous than a False Positive.

---

## 3. Objectives

1. Load and validate the official UNSW-NB15 training and testing CSV files.
2. Build a leakage-free preprocessing pipeline (fit only on training data).
3. Train a Random Forest classifier on training data only.
4. Evaluate the model on the held-out testing set.
5. Generate five required diagnostic visualizations.
6. Save the trained pipeline and evaluation artifacts.
7. Provide an optional Streamlit interface for interactive prediction.
8. Deliver a complete, reproducible, beginner-friendly academic project.

---

## 4. Dataset Description

The **UNSW-NB15** dataset was created at the University of New South Wales
(UNSW) Canberra in 2015. It uses the IXIA PerfectStorm tool to generate a
realistic mix of normal network traffic and nine attack categories (Fuzzers,
Analysis, Backdoors, DoS, Exploits, Generic, Reconnaissance, Shellcode, Worms).

| Property | Value |
|---|---|
| Official source | https://research.unsw.edu.au/projects/unsw-nb15-dataset |
| Training records | ~175,341 |
| Testing records | ~82,332 |
| Number of features | ~49 (varies by distribution) |
| Target column | `label` (0 = Normal, 1 = Attack) |
| Attack categories | 9 (DoS, Probe, Backdoor, etc.) |

### Key Features

| Feature | Type | Description |
|---|---|---|
| `dur` | Numeric | Duration of the connection |
| `proto` | Categorical | Protocol (tcp, udp, icmp …) |
| `service` | Categorical | Network service (http, ftp …) |
| `state` | Categorical | Connection state (FIN, CON …) |
| `sbytes` | Numeric | Source-to-destination bytes |
| `dbytes` | Numeric | Destination-to-source bytes |
| `sttl` | Numeric | Source IP time-to-live |
| `dttl` | Numeric | Destination IP time-to-live |
| `spkts` | Numeric | Source-to-destination packet count |
| `dpkts` | Numeric | Destination-to-source packet count |

### Excluded Columns

| Column | Reason |
|---|---|
| `id` | Record identifier — not a network measurement |
| `attack_cat` | Encodes the attack category name — direct target leakage |

---

## 5. Methodology

### 5.1 Data Preprocessing

The preprocessing pipeline follows a strict no-leakage discipline:
all transformation parameters (imputation medians, category vocabularies)
are computed from training data only and then applied unchanged to test data.

**Steps:**
1. Load CSV files and normalize column names (lowercase, strip whitespace).
2. Replace infinite values with NaN.
3. Separate features (X) and target label (y).
4. Exclude `id` and `attack_cat` from X.
5. For numeric features: fill missing values with the training median.
6. For categorical features: fill missing values with the training mode,
   then one-hot encode with `handle_unknown='ignore'`.
7. Fit everything as a scikit-learn `Pipeline` + `ColumnTransformer`.

### 5.2 Random Forest Algorithm

A **Random Forest** is an ensemble of decision trees. Each tree is trained on
a different random bootstrap sample of the training data, and each node split
considers only a random subset of features. Final predictions are made by
**majority voting** across all trees.

**Why Random Forest for this task?**
- Handles mixed numeric/categorical data naturally.
- Robust to outliers and noisy features.
- Provides built-in feature importances.
- `class_weight='balanced'` compensates for class imbalance.
- Strong benchmark performance on tabular security datasets.

**Hyperparameters:**

| Parameter | Value | Effect |
|---|---|---|
| `n_estimators` | 150 | Number of decision trees |
| `random_state` | 42 | Ensures reproducible results |
| `n_jobs` | -1 | Parallel training on all CPU cores |
| `class_weight` | `balanced` | Up-weights the minority (attack) class |

### 5.3 Evaluation Strategy

- Train exclusively on `UNSW_NB15_training-set.csv`.
- Evaluate exclusively on `UNSW_NB15_testing-set.csv`.
- No hyperparameter tuning using test data.
- All metrics computed with Attack (1) as the positive class.

---

## 6. System Architecture

```
data/
  UNSW_NB15_training-set.csv  ──► data_loader.py ──► preprocessing.py
  UNSW_NB15_testing-set.csv

preprocessing.py  ──► build ColumnTransformer (fit on train only)
                       │
                       ▼
model.py  ──► build RandomForestClassifier
               │
               ▼
           sklearn Pipeline (preprocessor + classifier)
               │
               ├──► train on X_train, y_train
               │
               └──► predict on X_test
                       │
                       ├──► evaluation.py  (metrics, predictions.csv)
                       └──► visualization.py (5 plots)
                                │
                                ▼
                           outputs/
                             intrusion_detection_model.joblib
                             metrics.json
                             classification_report.txt
                             predictions.csv
                             plot_*.png
```

---

## 7. Evaluation Metrics

### Definitions

| Term | Meaning |
|---|---|
| True Positive (TP) | Attack correctly classified as Attack |
| True Negative (TN) | Normal correctly classified as Normal |
| False Positive (FP) | Normal incorrectly flagged as Attack (false alarm) |
| False Negative (FN) | Attack incorrectly classified as Normal (**missed threat**) |

| Metric | Formula | What it measures |
|---|---|---|
| Accuracy | (TP+TN) / total | Overall correct classification rate |
| Precision | TP / (TP+FP) | How many predicted attacks are real attacks |
| Recall | TP / (TP+FN) | How many real attacks are caught |
| F1-score | 2·P·R / (P+R) | Balance between Precision and Recall |
| ROC-AUC | Area under ROC | Ranking ability across all thresholds |

> **False Negatives** are especially dangerous in security: a real attack is
> classified as normal and goes undetected. This is why high Recall is critical.
>
> **False Positives** waste analyst time (alert fatigue). Both must be minimized.

---

## 8. Results and Graphs

### Evaluation Metrics

| Metric | Value |
|---|---|
| Accuracy | 0.8929091969100714 |
| Precision | 0.851423401920967 |
| Recall | 0.975778699373511 |
| F1-score | 0.909369378629799 |
| ROC-AUC | 0.9810753325018304 |

### Generated Plots

After running `python train_model.py`, the following plots are saved to `outputs/`:

| Plot | File |
|---|---|
| Class Distribution | `plot_class_distribution.png` |
| Confusion Matrix | `plot_confusion_matrix.png` |
| Evaluation Metrics | `plot_metrics_bar.png` |
| ROC Curve | `plot_roc_curve.png` |
| Feature Importance | `plot_feature_importance.png` |

---

## 9. Discussion and Limitations

### Limitations

1. **Offline evaluation only.** The model is evaluated on static CSV files.
   Real-world intrusion detection requires streaming pipelines and much lower
   latency.

2. **Dataset bias.** UNSW-NB15 was generated in a controlled lab environment.
   Attack patterns may not match real-world network traffic.

3. **Class imbalance.** Normal traffic records outnumber attack records.
   `class_weight='balanced'` mitigates this but does not eliminate it.

4. **No hyperparameter optimization.** The baseline configuration is
   reasonable but not optimal. A proper grid search or Bayesian optimization
   could improve performance.

5. **Feature engineering.** UNSW-NB15 provides pre-engineered features.
   Additional feature engineering (e.g., flow statistics, time aggregates)
   could improve detection of subtle attacks.

6. **Accuracy is not sufficient alone.** In highly imbalanced datasets,
   a model that always predicts "Normal" can achieve high accuracy while
   detecting zero attacks. Recall and F1 are more informative.

---

## 10. Conclusion

This project demonstrates that a Random Forest classifier trained on the
UNSW-NB15 dataset can effectively distinguish normal from attack network
traffic. All required academic deliverables — preprocessing pipeline, model
training, evaluation metrics, visualizations, and documentation — have been
implemented in a clean, modular, and reproducible Python codebase.

The system should be understood as an educational proof-of-concept for
academic submission, not a production-ready network security tool.

---

## 11. Future Scope

- Implement multi-class classification across all nine attack categories.
- Explore deep learning approaches (LSTM, CNN for sequence modeling).
- Add real-time streaming prediction using Kafka or Flink.
- Investigate SHAP values for deeper model interpretability.
- Perform systematic cross-validation and hyperparameter tuning.
- Evaluate model drift over time with newer network traffic datasets.

---

## 12. References

1. Moustafa, N., & Slay, J. (2015). UNSW-NB15: A comprehensive data set for
   network intrusion detection systems. In *2015 Military Communications and
   Information Systems Conference (MilCIS)* (pp. 1–6). IEEE.
   https://doi.org/10.1109/MilCIS.2015.7348942

2. Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python.
   *Journal of Machine Learning Research*, 12, 2825–2830.
   https://scikit-learn.org/

3. Breiman, L. (2001). Random Forests. *Machine Learning*, 45(1), 5–32.
   https://doi.org/10.1023/A:1010933404324

4. Tavallaee, M., Bagheri, E., Lu, W., & Ghorbani, A. A. (2009). A detailed
   analysis of the KDD CUP 99 data set. In *2009 IEEE Symposium on
   Computational Intelligence for Security and Defense Applications* (CISDA).

5. UNSW Canberra Cyber. (n.d.). UNSW-NB15 Dataset.
   Retrieved from https://research.unsw.edu.au/projects/unsw-nb15-dataset
