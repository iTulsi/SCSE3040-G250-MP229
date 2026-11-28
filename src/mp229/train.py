import hashlib
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from joblib import dump
from mlflow.tracking import MlflowClient
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
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
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("MP229 Parcel Delivery")

    data_hash = hashlib.sha256(
        Path("data/raw/train.csv").read_bytes()
    ).hexdigest()
    data = clean_records(pd.read_csv("data/raw/train.csv"), training_data=True)
    data["date"] = pd.to_datetime(data["date"], errors="raise")

    # Validate on newer dates than those used for fitting.
    cutoff = data["date"].sort_values().iloc[int(len(data) * 0.8)]
    train = data[data["date"] < cutoff]
    validation = data[data["date"] >= cutoff]

    baseline = DummyRegressor(strategy="median")
    baseline.fit(train[FEATURES], train["delivery_days"])

    preprocessor = ColumnTransformer([
        ("numeric", SimpleImputer(strategy="median"), NUMERIC),
        ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ])
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(
            n_estimators=100, min_samples_leaf=2, random_state=42
        )),
    ])
    settings = [
        {"n_estimators": 50, "min_samples_leaf": 1},
        {"n_estimators": 100, "min_samples_leaf": 2},
        {"n_estimators": 150, "min_samples_leaf": 4},
    ]

    for setting in settings:
        trial = clone(model)
        trial.set_params(
            regressor__n_estimators=setting["n_estimators"],
            regressor__min_samples_leaf=setting["min_samples_leaf"],
        )

        with mlflow.start_run():
            trial.fit(train[FEATURES], train["delivery_days"])
            validation_mae = mae(trial, validation, FEATURES)

            mlflow.log_param("data_sha256", data_hash)
            mlflow.log_params(setting)
            mlflow.log_metric("validation_mae", validation_mae)

            print(
                f"Setting: {setting}, "
                f"validation MAE: {validation_mae:.3f}, "
                f"run ID: {mlflow.active_run().info.run_id}"
            )
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
    registered_name = "MP229-Parcel-Delivery"

    with mlflow.start_run(run_name="v1-full-training") as run:
        mlflow.log_param("data_sha256", data_hash)
        mlflow.log_param("training_data", "cleaned_train")
        mlflow.log_metric("mae_check", v1_mae)
        v1_info = mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            serialization_format="cloudpickle",
        )
        v1_run_id = run.info.run_id

    v1_version = mlflow.register_model(v1_info.model_uri, registered_name)

    with mlflow.start_run(run_name="v2-retrained") as run:
        mlflow.log_param("data_sha256", data_hash)
        mlflow.log_param("training_data", "cleaned_train_plus_older_later")
        mlflow.log_metric("mae_check", v2_mae)
        v2_info = mlflow.sklearn.log_model(
            sk_model=candidate,
            name="model",
            serialization_format="cloudpickle",
        )
        v2_run_id = run.info.run_id

    v2_version = mlflow.register_model(v2_info.model_uri, registered_name)

    champion = v2_version if promote else v1_version
    MlflowClient().set_registered_model_alias(
        registered_name, "champion", champion.version
    )

    print("Registered model:", registered_name)
    print("v1 run ID:", v1_run_id, "version:", v1_version.version)
    print("v2 run ID:", v2_run_id, "version:", v2_version.version)
    print("Champion version:", champion.version)
    model_path = Path("models/model.joblib")
    model_path.parent.mkdir(exist_ok=True)
    dump(candidate if promote else model, model_path)
    print("Saved selected model:", model_path)


if __name__ == "__main__":
    main()
