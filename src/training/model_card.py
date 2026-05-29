"""
PharmaGuard — Model Card Generator
======================================
Produces a Markdown model card documenting the trained model's lineage,
architecture, performance metrics, intended use, and known limitations.

Model cards promote transparency and reproducibility by capturing all
relevant context at the time of training. Each training run produces a
new versioned card in the ``model_cards/`` directory.
"""

import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


def generate_model_card(
    metrics_train: dict,
    metrics_val: dict,
    metrics_test: dict,
    config: dict,
    output_path: Path,
    threshold: float = 0.5,
) -> Path:
    """
    Generate a Markdown model card and write it to disk.

    Assembles a structured document that includes:

    1. **Model Overview** — model type, objective, class balance strategy.
    2. **Dataset Summary** — source (FDA FAERS), split sizes, class ratios.
    3. **Features** — categorical and numerical features used.
    4. **Hyperparameters** — all training config values.
    5. **Performance Metrics** — precision, recall, F1, ROC-AUC, PR-AUC
       for train / val / test splits, presented as a comparison table.
    6. **Decision Threshold** — the tuned threshold and strategy used.
    7. **Intended Use** — what the model is for and what it is *not* for.
    8. **Limitations & Risks** — known caveats about the data and model.

    Parameters
    ----------
    metrics_train : dict
        Metrics dictionary for the training set (from ``compute_metrics``).
    metrics_val : dict
        Metrics dictionary for the validation set.
    metrics_test : dict
        Metrics dictionary for the test set.
    config : dict
        Parsed config.yaml dictionary.
    output_path : Path
        File path to write the model card (e.g. ``model_cards/model_card_v1.md``).
    threshold : float
        The decision threshold used for generating predictions.

    Returns
    -------
    Path
        The path where the model card was saved.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    train_cfg = config.get("training", {})
    preproc_cfg = config.get("preprocessing", {})

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    card = f"""# PharmaGuard — Model Card

**Generated**: {timestamp}

---

## Model Overview

| Field | Value |
|---|---|
| Model Type | XGBoost (`XGBClassifier`) |
| Objective | `binary:logistic` |
| Task | Binary classification — Serious vs. Not Serious ADR |
| Imbalance Strategy | `scale_pos_weight = {train_cfg.get('scale_pos_weight', 'N/A')}` |
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
**Quarters**: {config.get('faers', {}).get('quarters', 'N/A')}

---

## Features

### Categorical (OneHotEncoded)
{_format_list(preproc_cfg.get('categorical_features', []))}

### Numerical (Passthrough)
{_format_list(preproc_cfg.get('numerical_features', []))}

---

## Hyperparameters

| Parameter | Value |
|---|---|
| `n_estimators` | {train_cfg.get('n_estimators', 'N/A')} |
| `max_depth` | {train_cfg.get('max_depth', 'N/A')} |
| `learning_rate` | {train_cfg.get('learning_rate', 'N/A')} |
| `scale_pos_weight` | {train_cfg.get('scale_pos_weight', 'N/A')} |
| `eval_metric` | {train_cfg.get('eval_metric', 'N/A')} |
| `early_stopping_rounds` | {train_cfg.get('early_stopping_rounds', 'N/A')} |
| `random_state` | {train_cfg.get('random_state', 'N/A')} |

---

## Decision Threshold

| Field | Value |
|---|---|
| Tuning Strategy | `{train_cfg.get('threshold_strategy', 'f1')}` |
| Optimal Threshold | **{threshold:.2f}** |
| Min Precision (if applicable) | {train_cfg.get('min_precision', 'N/A')} |

---

## Performance Metrics

| Metric | Train | Validation | Test |
|---|---|---|---|
| Precision | {metrics_train['precision']:.4f} | {metrics_val['precision']:.4f} | {metrics_test['precision']:.4f} |
| Recall | {metrics_train['recall']:.4f} | {metrics_val['recall']:.4f} | {metrics_test['recall']:.4f} |
| F1 Score | {metrics_train['f1']:.4f} | {metrics_val['f1']:.4f} | {metrics_test['f1']:.4f} |
| Accuracy | {metrics_train['accuracy']:.4f} | {metrics_val['accuracy']:.4f} | {metrics_test['accuracy']:.4f} |
| Specificity | {metrics_train['specificity']:.4f} | {metrics_val['specificity']:.4f} | {metrics_test['specificity']:.4f} |
| ROC-AUC | {metrics_train['roc_auc']:.4f} | {metrics_val['roc_auc']:.4f} | {metrics_test['roc_auc']:.4f} |
| PR-AUC | {metrics_train['pr_auc']:.4f} | {metrics_val['pr_auc']:.4f} | {metrics_test['pr_auc']:.4f} |

### Test Set Confusion Matrix

|  | Predicted Not Serious | Predicted Serious |
|---|---|---|
| **Actual Not Serious** | {metrics_test['confusion_matrix'][0][0]} | {metrics_test['confusion_matrix'][0][1]} |
| **Actual Serious** | {metrics_test['confusion_matrix'][1][0]} | {metrics_test['confusion_matrix'][1][1]} |

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
- Random state: {train_cfg.get('random_state', 42)}
"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(card)

    logger.info("Model card saved to: %s", output_path)
    return output_path


def _format_list(items: list) -> str:
    """
    Format a Python list as a Markdown bullet list.

    Parameters
    ----------
    items : list
        List of strings to format.

    Returns
    -------
    str
        Markdown-formatted bullet list.
    """
    if not items:
        return "- _(none)_"
    return "\n".join(f"- `{item}`" for item in items)
