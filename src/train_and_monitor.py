from pathlib import Path
import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
import yaml

from src.clean_data import (
    CATEGORICAL_COLS,
    FEATURE_COLS,
    NUMERIC_COLS,
    TARGET_COL,
    clean_train_df,
)

EXPERIMENT_NAME = "mp229-parcel-delivery"
REGISTERED_MODEL_NAME = "mp229-delivery-days"
TRUSTED_TYPES = ["sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor"]


def compute_psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    edges = np.quantile(expected, np.linspace(0.0, 1.0, bins + 1))
    edges[0] = -np.inf
    edges[-1] = np.inf
    edges = np.unique(edges)
    exp_p = np.clip(np.histogram(expected, bins=edges)[0] / len(expected), 1e-6, None)
    act_p = np.clip(np.histogram(actual, bins=edges)[0] / len(actual), 1e-6, None)
    return float(np.sum((act_p - exp_p) * np.log(act_p / exp_p)))


def build_pipeline() -> Pipeline:
    pre = ColumnTransformer(
        transformers=[
            ("num", "passthrough", NUMERIC_COLS),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLS),
        ]
    )
    model = HistGradientBoostingRegressor(random_state=42)
    return Pipeline([("pre", pre), ("model", model)])


def main() -> None:
    train_clean, stats = clean_train_df(pd.read_csv("data/train.csv"))
    later = pd.read_csv("data/later.csv")
    holdout = pd.read_csv("data/holdout_features.csv")

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(EXPERIMENT_NAME)

    v1 = build_pipeline()
    v1.fit(train_clean[FEATURE_COLS], train_clean[TARGET_COL])

    with mlflow.start_run(run_name="v1_cleaned_train") as run_v1:
        v1_run_id = run_v1.info.run_id
        v1_later_preds = v1.predict(later[FEATURE_COLS])
        v1_later_mae = float(mean_absolute_error(later[TARGET_COL], v1_later_preds))
        mlflow.log_metric("later_mae", v1_later_mae)
        mlflow.sklearn.log_model(
            v1,
            artifact_path="model",
            registered_model_name=REGISTERED_MODEL_NAME,
            skops_trusted_types=TRUSTED_TYPES,
        )

    psi_map = {
        col: round(compute_psi(train_clean[col].to_numpy(), later[col].to_numpy(), bins=10), 4)
        for col in NUMERIC_COLS
    }
    most_drifted = max(psi_map, key=psi_map.get)

    mae_by_group = {}
    for grp in ["Metro", "Tier-2", "Rural"]:
        mask = later["destination"] == grp
        mae_by_group[grp] = round(
            float(mean_absolute_error(later.loc[mask, TARGET_COL], v1_later_preds[mask])), 4
        )
    worst_group = max(mae_by_group, key=mae_by_group.get)

    later_sorted = later.sort_values("date").reset_index(drop=True)
    split_idx = int(np.floor(0.7 * len(later_sorted)))
    later_first70 = later_sorted.iloc[:split_idx]
    later_last30 = later_sorted.iloc[split_idx:]

    mae_v1_check = float(mean_absolute_error(later_last30[TARGET_COL], v1.predict(later_last30[FEATURE_COLS])))

    v2_df = pd.concat(
        [train_clean[FEATURE_COLS + [TARGET_COL]], later_first70[FEATURE_COLS + [TARGET_COL]]],
        ignore_index=True,
    )
    v2 = build_pipeline()
    v2.fit(v2_df[FEATURE_COLS], v2_df[TARGET_COL])

    with mlflow.start_run(run_name="v2_retrained_70_later"):
        mae_v2_check = float(mean_absolute_error(later_last30[TARGET_COL], v2.predict(later_last30[FEATURE_COLS])))
        mlflow.log_metric("check_mae", mae_v2_check)
        mlflow.sklearn.log_model(
            v2,
            artifact_path="model",
            registered_model_name=REGISTERED_MODEL_NAME,
            skops_trusted_types=TRUSTED_TYPES,
        )

    promote = mae_v2_check <= 0.85 * mae_v1_check
    decision = "promote" if promote else "keep"
    champion = v2 if promote else v1
    champion_version = "2" if promote else "1"

    client = mlflow.tracking.MlflowClient()
    client.set_registered_model_alias(REGISTERED_MODEL_NAME, "champion", champion_version)

    Path("models").mkdir(parents=True, exist_ok=True)
    joblib.dump(champion, Path("models/champion_model.joblib"))

    Path("predictions").mkdir(parents=True, exist_ok=True)
    holdout_preds = pd.DataFrame(
        {
            "record_id": holdout["record_id"],
            "prediction": np.round(champion.predict(holdout[FEATURE_COLS]), 4),
        }
    )
    holdout_preds.to_csv("predictions/holdout_predictions.csv", index=False)

    findings_path = Path("findings.yaml")
    findings = yaml.safe_load(findings_path.read_text()) if findings_path.exists() else {}
    findings["group_id"] = "G250"
    findings["problem_id"] = "MP229"
    findings["data_issues"] = {
        "leaking_column": {"present": True, "column": "days_from_tracking_page"},
        "unit_mixup": {"present": True, "column": "weight_kg", "group_value": "Rural", "factor": 1000},
        "sentinel_value": {
            "present": True,
            "column": "transfers",
            "value": stats["sentinel_value"],
            "count": stats["sentinel_count"],
        },
        "duplicate_rows": {"present": True, "count": stats["duplicate_extra_count"]},
        "stuck_sensor": {"present": False, "column": None, "start_date": None, "end_date": None},
        "target_typos": {"present": True, "count": stats["target_typo_count"]},
        "spelling_variants": {"present": False, "column": None, "true_levels": None},
    }
    findings["tracking"] = {
        "experiment_name": EXPERIMENT_NAME,
        "best_run_id": v1_run_id,
        "registered_model_name": REGISTERED_MODEL_NAME,
        "champion_version": champion_version,
    }
    findings["monitoring"] = {
        "psi": psi_map,
        "most_drifted_feature": most_drifted,
        "mae_by_group": mae_by_group,
        "worst_served_group": worst_group,
        "mae_v1_on_check": round(mae_v1_check, 4),
        "mae_v2_on_check": round(mae_v2_check, 4),
        "decision": decision,
    }
    findings_path.write_text(yaml.safe_dump(findings, sort_keys=False))
    print("Done! findings.yaml and predictions/holdout_predictions.csv ready.")


if __name__ == "__main__":
    main()
