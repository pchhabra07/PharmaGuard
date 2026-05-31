## 🚨 New FDA FAERS Data Detected

PharmaGuard's automated monthly check has detected **new quarterly data** on the FDA FAERS portal that your model has not been trained on yet.

### What to do

Run the one-command retraining script on your local machine:

```bash
python run_retrain.py
```

This script will automatically:
1. ✅ Detect all new quarters available on the FDA server
2. ✅ Update `config.yaml` with the new quarters
3. ✅ Run data ingestion (`run_ingestion.py`)
4. ✅ Run preprocessing (`run_preprocessing.py`)
5. ✅ Run model training (`run_training.py`) with auto-versioning (v3, v4, ...)
6. ✅ Stage, commit, and push updated model artifacts to GitHub
7. ✅ Trigger automatic redeployment via CI/CD pipeline

### Requirements
- Your local machine (training requires several GB of RAM)
- Virtual environment activated with all dependencies installed

### After running
- The CI pipeline will run automatically on the new commit
- Your deployed API will pick up the new model on next deploy
- Close this issue once the retraining is complete
