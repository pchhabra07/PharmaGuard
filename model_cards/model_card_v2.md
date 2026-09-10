# PharmaGuard — Model Card

**Generated**: 2026-05-30 01:27:47

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
**Quarters**: ['2026q1']

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
| Optimal Threshold | **0.24** |
| Min Precision (if applicable) | 0.3 |

---

## Performance Metrics

| Metric | Train | Validation | Test |
|---|---|---|---|
| Precision | 0.8143 | 0.8134 | 0.8145 |
| Recall | 0.9550 | 0.9545 | 0.9543 |
| F1 Score | 0.8790 | 0.8783 | 0.8789 |
| Accuracy | 0.8091 | 0.8080 | 0.8090 |
| Specificity | 0.4220 | 0.4191 | 0.4235 |
| ROC-AUC | 0.8604 | 0.8579 | 0.8596 |
| PR-AUC | 0.9424 | 0.9411 | 0.9419 |

### Test Set Confusion Matrix

|  | Predicted Not Serious | Predicted Serious |
|---|---|---|
| **Actual Not Serious** | 29613 | 40314 |
| **Actual Serious** | 8488 | 177067 |

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
