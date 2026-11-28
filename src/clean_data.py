from pathlib import Path
import pandas as pd

FEATURE_COLS = [
    "distance_km",
    "weight_kg",
    "hub_backlog",
    "pin_code_tier",
    "declared_value_rs",
    "transfers",
    "service",
    "destination",
]
NUMERIC_COLS = [
    "distance_km",
    "weight_kg",
    "hub_backlog",
    "pin_code_tier",
    "declared_value_rs",
    "transfers",
]
CATEGORICAL_COLS = ["service", "destination"]
TARGET_COL = "delivery_days"


def clean_train_df(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    raw_rows = len(df)
    dup_extra = int(df.duplicated().sum())
    out = df.drop_duplicates().copy()

    rural_mask = out["destination"] == "Rural"
    out.loc[rural_mask, "weight_kg"] = out.loc[rural_mask, "weight_kg"] / 1000.0

    sentinel_val = 9999
    sentinel_mask = out["transfers"] == sentinel_val
    sentinel_count = int(sentinel_mask.sum())
    valid_median = out.loc[~sentinel_mask, "transfers"].median()
    out.loc[sentinel_mask, "transfers"] = valid_median

    typo_mask = out[TARGET_COL] > 20.0
    typo_count = int(typo_mask.sum())
    out.loc[typo_mask, TARGET_COL] = out.loc[typo_mask, TARGET_COL] / 10.0

    stats = {
        "raw_rows": raw_rows,
        "cleaned_rows": len(out),
        "duplicate_extra_count": dup_extra,
        "sentinel_value": sentinel_val,
        "sentinel_count": sentinel_count,
        "target_typo_count": typo_count,
    }
    return out, stats


def main() -> None:
    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv("data/train.csv")
    cleaned, stats = clean_train_df(df)
    cleaned.to_csv(out_dir / "train_clean.csv", index=False)
    print("Cleaned stats:", stats)


if __name__ == "__main__":
    main()
