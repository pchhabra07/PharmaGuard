# PharmaGuard 🛡️

**End-to-End MLOps Pipeline for Adverse Drug Event Detection**

PharmaGuard is a comprehensive **Machine Learning and MLOps** project designed to detect serious Adverse Drug Reactions (ADRs) from real-world patient medication reports using the publicly available **FDA FAERS** (Adverse Event Reporting System) dataset. 

Beyond core model training, this project emphasizes robust **MLOps practices**, featuring **automated retraining pipelines**, comprehensive **deployment capabilities**, and a full **CI/CD pipeline** to ensure continuous integration, testing, and seamless delivery of the ML models to production environments.

---

## 🎯 The Problem

When patients take medications, some experience unexpected harmful side effects (Adverse Drug Reactions). PharmaGuard predicts whether a given drug-patient combination is likely to result in a **serious adverse outcome** — hospitalization, disability, or death.

- **Task**: Binary classification (serious vs. not serious)
- **Challenge**: Heavy class imbalance — serious events are ~73% of records
- **Approach**: XGBoost with `scale_pos_weight` for native imbalance handling

---

## 📁 Project Structure

```
PharmaGuard/
├── data/
│   ├── raw/                  # FAERS quarterly ZIP downloads
│   ├── processed/            # Cleaned and merged DataFrames
│   ├── splits/               # Train / Val / Test CSVs
│   └── eda/
│       ├── *.png             # Phase 1 EDA visualizations
│       └── training/         # Phase 3 evaluation plots
├── src/
│   ├── ingestion/            # Phase 1: download, extract, join, validate, EDA
│   ├── preprocessing/        # Phase 2: normalization, features, pipeline, split
│   └── training/             # Phase 3: train, evaluate, threshold, model card
├── model_artifacts/          # Saved model (.json) and pipeline (.pkl)
├── model_cards/              # Model card per registered version
├── tests/                    # pytest unit and integration tests
├── docker/                   # Dockerfile, docker-compose.yml
├── .github/workflows/        # GitHub Actions CI/CD & Cron workflows
├── config.yaml               # Centralized configuration
├── requirements.txt          # Python dependencies
├── run_ingestion.py          # Phase 1 runner script
├── run_preprocessing.py      # Phase 2 runner script
└── run_training.py           # Phase 3 runner script
```

---

## 🚀 Quick Start

### 1. Clone & Setup

```bash
git clone https://github.com/pchhabra07/PharmaGuard.git
cd PharmaGuard

# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Phase 1 — Data Ingestion

Downloads FAERS data, extracts tables, joins them, validates, and runs EDA:

```bash
python run_ingestion.py
```

> ⏱️ This downloads ~61 MB from the FDA and processes the data. Takes 3-5 minutes.

### 3. Phase 2 — Preprocessing Pipeline

Normalizes drug names, engineers features, builds the sklearn pipeline, computes class weights, and splits data:

```bash
python run_preprocessing.py
```

### 4. Phase 3 — Model Training & Evaluation

Trains an XGBoost classifier with early stopping, tunes the decision threshold, evaluates on all splits, generates diagnostic plots, and writes a model card:

```bash
python run_training.py
```

> ⏱️ Training on ~1.2M rows with 396 features. Takes 5-15 minutes depending on hardware.

---

## 📊 Dataset — FDA FAERS

The FDA Adverse Event Reporting System (FAERS) contains millions of voluntary reports submitted by healthcare professionals, patients, and pharmaceutical manufacturers.

- **Source**: https://fis.fda.gov/extensions/FPD-QDE-FAERS/FPD-QDE-FAERS.html
- **Update Frequency**: Quarterly (January, April, July, October)
- **Tables Used**: DEMO, DRUG, REAC, OUTC, INDI

---

## 🔧 Configuration

All parameters are centralized in `config.yaml`:

| Parameter | Description |
|---|---|
| `faers.quarters` | Which FAERS quarters to download |
| `preprocessing.age_bins` | Age bucketing boundaries |
| `preprocessing.drug_fuzzy_threshold` | RapidFuzz matching threshold |
| `splitting.train_ratio` | Train/val/test split proportions |
| `training.scale_pos_weight` | Auto-computed class imbalance weight |
| `training.n_estimators` | Maximum XGBoost boosting rounds |
| `training.max_depth` | Maximum tree depth |
| `training.learning_rate` | Boosting learning rate (eta) |
| `training.eval_metric` | Early stopping metric (`aucpr`) |
| `training.early_stopping_rounds` | Patience for early stopping |
| `training.threshold_strategy` | Threshold tuning strategy (`f1` / `recall_at_precision` / `youden`) |
| `training.threshold` | Tuned decision threshold (auto-updated) |

---

## 🧬 Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.10+ |
| Data | Pandas, NumPy |
| Feature Engineering | RapidFuzz, Pandas |
| Preprocessing | Scikit-learn Pipeline, ColumnTransformer |
| ML Model | XGBoost |
| Visualization | Matplotlib, Seaborn |
| Config | PyYAML |
| Serialization | XGBoost native JSON, Joblib |

---

## 📋 Development Phases

- [x] **Phase 1**: Project Setup & Data Ingestion
- [x] **Phase 2**: Preprocessing Pipeline
- [x] **Phase 3**: Model Training & Evaluation
- [ ] **Phase 4**: Experiment Tracking (MLflow)
- [ ] **Phase 5**: REST API & Docker
- [ ] **Phase 6**: CI/CD (GitHub Actions)
- [ ] **Phase 7**: Cloud Deployment & Auto-Retraining
- [ ] **Phase 8**: SHAP Explainability (Future)
- [ ] **Phase 9**: Drift Detection (Future)

---

## 📜 License

This project is for educational purposes.
