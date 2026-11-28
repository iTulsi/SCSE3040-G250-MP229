# SCSE3040 MLOps Mini Project — G250

**Problem:** MP229 — Parcel Delivery Days (Guwahati)  
**Faculty:** Dr. Upasna Joshi  
**Batch:** 76

## Members

| Enrolment number | Member |
| --- | --- |
| S24CSEU0084 | Tulsi Sanskrati Tomar |
| S24CSEU1977 | Abhay Verma Raj |
| S24CSEU1536 | Harshiv Batra |
| S24CSEU2429 | Satyajeet |
| S24CSEU1988 | Harsh Khandelwal |

## Goal

Predict `delivery_days` for parcels using an end-to-end MLOps pipeline.

## Setup

Use Python 3.12 and run commands from the repository root.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Training requires the original MP229 `train.csv` and `later.csv` in
`data/raw/`. These files are tracked through DVC and are not committed
directly. Obtain the original files from the group dataset package.

## Data inspection and tests

```bash
python -m src.mp229.inspect_data
python -m pytest -q
```

Training-only corrections remove duplicate copies, convert Rural weights
from grams to kilograms, mark transfer sentinel values as missing, and
correct the identified target typos. Median imputation is fitted within
the model pipeline. The post-delivery tracking column is excluded.

## Training and tracking

```bash
python -m src.mp229.train
```

Model settings are read from `params.yaml`. Three random forest settings
are compared on a chronological validation split. MLflow records
experiments in `mlflow.db` and registers the original and retrained models.

Retraining uses cleaned training data plus the first 210 date-sorted later
records. Both models are evaluated on the remaining 90 records.
Version 2 achieved MAE 0.780564 days versus 2.107378 days for version 1,
meeting the required 15% improvement. Version 2 received the `champion`
alias. The selected pipeline is saved to `models/model.joblib`.

Each training execution creates additional runs and registry versions.

## Holdout predictions

```bash
python -m src.mp229.predict_holdout
```

This command loads the MLflow champion and requires its local MLflow
artifacts, which are excluded from Git. A fresh clone must regenerate
those artifacts through training before using this command.

The committed `predictions/holdout_predictions.csv` contains 250 records
with columns `record_id,prediction`. Its predictions were verified against
`models/model.joblib`. Holdout MAE is unknown because labels are withheld.

## Findings

Data findings, monitoring results and registry details are recorded in
`reports/findings.yaml`. Recalculate them if the model or data changes.
