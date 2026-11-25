from pathlib import Path

import pandas as pd
from joblib import dump
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.mp229.data_checks import clean_records
from src.parcel.drift import psi_report
from src.parcel.evaluate import mae, mae_by_destination
from src.parcel.promote import should_promote, split_by_date

NUMERIC = [
    "distance_km", "weight_kg", "hub_backlog",
    "pin_code_tier", "declared_value_rs", "transfers",
]
CATEGORICAL = ["service", "destination"]
FEATURES = NUMERIC + CATEGORICAL


def main() -> None:
    data = clean_records(pd.read_csv("data/raw/train.csv"))
    data["date"] = pd.to_datetime(data["date"], errors="raise")

    # Validate on newer dates than those used for fitting.
    cutoff = data["date"].sort_values().iloc[int(len(data) * 0.8)]
    train = data[data["date"] < cutoff]
    validation = data[data["date"] >= cutoff]

    baseline = DummyRegressor(strategy="median")
    baseline.fit(train[FEATURES], train["delivery_days"])

    preprocessor = ColumnTransformer([
        ("numeric", "passthrough", NUMERIC),
        ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ])
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(
            n_estimators=100, min_samples_leaf=2, random_state=42
        )),
    ])
    model.fit(train[FEATURES], train["delivery_days"])

    print(f"Train: {len(train)} rows; validation: {len(validation)} rows")
    print(f"Median baseline MAE: {mae(baseline, validation, FEATURES):.3f}")
    print(f"Random forest MAE:   {mae(model, validation, FEATURES):.3f}")
    later = clean_records(pd.read_csv("data/raw/later.csv"))
    model.fit(data[FEATURES], data["delivery_days"])

    print(f"Later MAE: {mae(model, later, FEATURES):.3f}")
    print("Later MAE by destination:", mae_by_destination(model, later, FEATURES))
    print("Train-to-later PSI:", psi_report(data, later))

    older, newer = split_by_date(later)
    candidate = clone(model)
    retraining = pd.concat([data, older], ignore_index=True)
    candidate.fit(retraining[FEATURES], retraining["delivery_days"])

    v1_mae = mae(model, newer, FEATURES)
    v2_mae = mae(candidate, newer, FEATURES)
    print(f"v1 evaluation MAE: {v1_mae:.3f}")
    print(f"v2 evaluation MAE: {v2_mae:.3f}")
    print("v2 by destination:", mae_by_destination(candidate, newer, FEATURES))
    promote = should_promote(v1_mae, v2_mae)
    print("Promote v2:", promote)
    model_path = Path("models/delivery_model.joblib")
    model_path.parent.mkdir(exist_ok=True)
    dump(candidate if promote else model, model_path)
    print("Saved selected model:", model_path)


if __name__ == "__main__":
    main()
