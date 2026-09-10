# PharmaGuard — Model Card

**Generated**: 2026-09-10 23:44:08

---

## Model Overview

| Field | Value |
|---|---|
| Model Type | XGBoost (`XGBClassifier`) |
| Objective | `binary:logistic` |
| Task | Binary classification — Serious vs. Not Serious ADR |
| Imbalance Strategy | `scale_pos_weight = 0.38` |
| Framework | XGBoost ≥ 2.0, scikit-learn ≥ 1.3 |
| Serialisation | XGBoost native JSON (`.json`) |

---

## Dataset

| Split | Rows | Positive Rate |
|---|---|---|
| Train | — | ~72.6% |
| Validation | — | ~72.6% |
| Test | — | ~72.6% |

**Source**: FDA FAERS (Adverse Event Reporting System)
**Quarters**: ['2026q1', '2012q4', '2013q1', '2013q2', '2013q3', '2013q4', '2014q1', '2014q2', '2014q3', '2014q4', '2015q1', '2015q2', '2015q3', '2015q4', '2016q1', '2016q2', '2016q3', '2016q4', '2017q1', '2017q2', '2017q3', '2017q4', '2018q1', '2018q2', '2018q3', '2018q4', '2019q1', '2019q2', '2019q3', '2019q4', '2020q1', '2020q2', '2020q3', '2020q4', '2021q1', '2021q2', '2021q3', '2021q4', '2022q1', '2022q2', '2022q3', '2022q4', '2023q1', '2023q2', '2023q3', '2023q4', '2024q1', '2024q2', '2024q3', '2024q4', '2025q1', '2025q2', '2025q3', '2025q4', '2026q2']

---

## Features

### Categorical (OneHotEncoded)
- `sex`
- `route`
- `age_group`
- `drug_name_normalized`

### Numerical (Passthrough)
- `polypharmacy_count`

---

## Hyperparameters

| Parameter | Value |
|---|---|
| `n_estimators` | 500 |
| `max_depth` | 6 |
| `learning_rate` | 0.1 |
| `scale_pos_weight` | 0.38 |
| `eval_metric` | aucpr |
| `early_stopping_rounds` | 30 |
| `random_state` | 42 |

---

## Decision Threshold

| Field | Value |
|---|---|
| Tuning Strategy | `f1` |
| Optimal Threshold | **0.27** |
| Min Precision (if applicable) | 0.3 |

---

## Performance Metrics

| Metric | Train | Validation | Test |
|---|---|---|---|
| Precision | 0.8253 | 0.8248 | 0.8256 |
| Recall | 0.9394 | 0.9390 | 0.9389 |
| F1 Score | 0.8786 | 0.8782 | 0.8786 |
| Accuracy | 0.8115 | 0.8108 | 0.8115 |
| Specificity | 0.4723 | 0.4707 | 0.4735 |
| ROC-AUC | 0.8601 | 0.8577 | 0.8593 |
| PR-AUC | 0.9424 | 0.9412 | 0.9419 |

### Test Set Confusion Matrix

|  | Predicted Not Serious | Predicted Serious |
|---|---|---|
| **Actual Not Serious** | 33113 | 36814 |
| **Actual Serious** | 11337 | 174218 |

---

## Intended Use

- **Primary use**: Flag patient-drug combinations likely to cause serious
  adverse events, supporting pharmacovigilance review.
- **Users**: Data scientists, pharmacovigilance analysts, regulatory reviewers.
- **Not for**: Direct clinical decision-making without human review.

---

## Limitations & Risks

1. **Voluntary reporting bias**: FAERS data is voluntarily reported and does
   not represent the true incidence of adverse events.
2. **Temporal drift**: The model is trained on a single quarter; drug
   landscapes change over time and may degrade performance.
3. **Feature sparsity**: Many FAERS fields (age, route, sex) have high
   missing-value rates, reducing discriminative power.
4. **No causal claims**: A high prediction score does not prove the drug
   *caused* the adverse event.
5. **Class imbalance**: The positive class (serious) is the majority (~73%),
   which may bias the model toward over-predicting seriousness.

---

## Reproducibility

- Config file: `config.yaml`
- Preprocessor: `model_artifacts/preprocessor.pkl`
- Model: `model_artifacts/xgb_model.json`
- Random state: 42
