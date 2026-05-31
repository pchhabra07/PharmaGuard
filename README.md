# PharmaGuard 🛡️

[![CI — Tests & Lint](https://github.com/pchhabra07/PharmaGuard/actions/workflows/ci.yml/badge.svg)](https://github.com/pchhabra07/PharmaGuard/actions/workflows/ci.yml)
[![Docker — Build Check](https://github.com/pchhabra07/PharmaGuard/actions/workflows/docker-build.yml/badge.svg)](https://github.com/pchhabra07/PharmaGuard/actions/workflows/docker-build.yml)

**End-to-End MLOps Pipeline for Adverse Drug Event Detection**

PharmaGuard is a comprehensive **Machine Learning and MLOps** project designed to detect serious Adverse Drug Reactions (ADRs) from real-world patient medication reports using the publicly available **FDA FAERS** (Adverse Event Reporting System) dataset. 

Beyond core model training, this project emphasizes robust **MLOps practices**, featuring **experiment tracking with MLflow**, a **FastAPI prediction API** with a public dashboard, **Docker containerization**, and a full **CI/CD pipeline** to ensure continuous integration, testing, and seamless delivery of the ML models to production environments.

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
│   ├── training/             # Phase 3: train, evaluate, threshold, model card
│   ├── tracking/             # Phase 4: MLflow experiment tracker, metrics store
│   ├── api/                  # Phase 5: FastAPI app, routes, schemas, dashboard
│   └── utils/                # Phase 7: new-data detection script
├── model_artifacts/          # Saved model (.json) and pipeline (.pkl), versioned
├── model_cards/              # Model card per registered version
├── metrics/                  # Metrics history JSON (dashboard data source)
├── docker/
│   ├── Dockerfile            # Multi-stage Docker build
│   └── docker-compose.yml    # Local development compose
├── tests/                    # pytest unit and integration tests
├── .github/
│   ├── workflows/            # GitHub Actions CI/CD, Docker, Health & Data checks
│   └── ISSUE_TEMPLATE/       # Auto-generated issue template for new data alerts
├── .dockerignore             # Docker build context exclusions
├── config.yaml               # Centralized configuration
├── requirements.txt          # Python dependencies
├── run_ingestion.py          # Phase 1 runner script
├── run_preprocessing.py      # Phase 2 runner script
├── run_training.py           # Phase 3 runner script (+ MLflow tracking)
├── run_api.py                # Phase 5 API server runner
└── run_retrain.py            # Phase 7 one-command retrain & deploy
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

Trains an XGBoost classifier with early stopping, tunes the decision threshold, evaluates on all splits, generates diagnostic plots, writes a model card, and logs everything to MLflow:

```bash
python run_training.py
```

> ⏱️ Training on ~1.2M rows with 396 features. Takes 5-15 minutes depending on hardware.

### 5. Phase 4 — Experiment Tracking (MLflow)

MLflow tracking is **automatically integrated** into `run_training.py`. Every training run logs hyperparameters, metrics, and artifacts to the local `mlruns/` directory.

To view the MLflow UI:

```bash
mlflow ui --backend-store-uri mlruns
```

Then open [http://localhost:5000](http://localhost:5000) in your browser.

### 6. Phase 5 — REST API & Dashboard

Start the FastAPI prediction server:

```bash
python run_api.py
```

Then open:
- **Dashboard**: [http://localhost:8000](http://localhost:8000) — Public metrics showcase
- **Swagger Docs**: [http://localhost:8000/docs](http://localhost:8000/docs) — Interactive API docs
- **ReDoc**: [http://localhost:8000/redoc](http://localhost:8000/redoc) — Alternative docs

#### API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Public dashboard with model metrics |
| `GET` | `/health` | Health check (model load status) |
| `GET` | `/model/info` | Model metadata and configuration |
| `GET` | `/model/history` | Historical training runs (JSON) |
| `POST` | `/predict` | Single ADR severity prediction |
| `POST` | `/predict/batch` | Batch predictions (up to 1000) |
| `GET` | `/docs` | Swagger UI documentation |

#### Example API Call

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"sex": "F", "age_group": "45-64", "route": "Oral", "drug_name_normalized": "ASPIRIN", "polypharmacy_count": 3}'
```

### 7. Docker

Build and run the API in a Docker container:

```bash
# Build the image
docker build -f docker/Dockerfile -t pharmaguard-api .

# Run the container
docker run -p 8000:8000 pharmaguard-api

# Or use Docker Compose
docker compose -f docker/docker-compose.yml up --build
```

### 8. Phase 7 — Automated New-Data Detection & Retraining

PharmaGuard includes an automated **monthly GitHub Actions workflow** that checks whether new quarterly data has been released on the FDA FAERS portal. The FDA publishes new adverse event data every quarter (January, April, July, October).

#### How it works

1. **Monthly Check**: On the 1st of every month, a GitHub Actions cron job runs `src/utils/check_new_data.py`.
2. **Auto-Issue Creation**: If new quarters are detected on the FDA server that are not in our `config.yaml`, the workflow **automatically creates a GitHub Issue** titled `🚨 [Data Update] New FDA FAERS quarter(s) available`. You receive an email notification from GitHub.
3. **One-Command Retrain**: When you receive the issue, run the retraining script on your local machine:

```bash
python run_retrain.py
```

This single command automatically:
- Detects all new quarters available on the FDA server
- Updates `config.yaml` with the new quarters
- Runs data ingestion (`run_ingestion.py`)
- Runs preprocessing (`run_preprocessing.py`)
- Trains a new model version (`run_training.py`) — auto-versioned (v3, v4, v5, ...)
- Stages, commits, and pushes updated model artifacts to GitHub
- Triggers the CI/CD pipeline automatically

> ⚠️ Training requires several GB of RAM and runs on your local machine. Free cloud servers (Render/GitHub Actions) do not have enough memory for XGBoost training on the full dataset.

#### Manual trigger

You can also trigger the data check manually from the GitHub **Actions** tab → **Scheduled — New Data Check** → **Run workflow**.

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
| `mlflow.enabled` | Enable/disable MLflow experiment tracking |
| `mlflow.experiment_name` | MLflow experiment name |
| `mlflow.tracking_uri` | MLflow tracking backend (default: `mlruns`) |
| `api.host` | API server host (default: `0.0.0.0`) |
| `api.port` | API server port (default: `8000`) |
| `api.model_version` | Model version identifier |

---

## 🧬 Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.11+ |
| Data | Pandas, NumPy |
| Feature Engineering | RapidFuzz, Pandas |
| Preprocessing | Scikit-learn Pipeline, ColumnTransformer |
| ML Model | XGBoost |
| Experiment Tracking | MLflow |
| API Framework | FastAPI, Uvicorn, Pydantic |
| Visualization | Matplotlib, Seaborn |
| Config | PyYAML |
| Serialization | XGBoost native JSON, Joblib |
| Containerization | Docker, Docker Compose |
| Deployment | Render (free tier) |

---

## 📋 Development Phases

- [x] **Phase 1**: Project Setup & Data Ingestion
- [x] **Phase 2**: Preprocessing Pipeline
- [x] **Phase 3**: Model Training & Evaluation
- [x] **Phase 4**: Experiment Tracking (MLflow)
- [x] **Phase 5**: REST API & Docker
- [x] **Phase 6**: CI/CD (GitHub Actions)
- [x] **Phase 7**: Cloud Deployment & Auto-Retraining
- [ ] **Phase 8**: SHAP Explainability (Future)
- [ ] **Phase 9**: Drift Detection (Future)

---

## 📜 License

This project is for educational purposes.
