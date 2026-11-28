import pandas as pd


def columns_missing_at_prediction(
    train_csv: str, holdout_csv: str, target: str
) -> list[str]:
    train_columns = set(pd.read_csv(train_csv, nrows=0).columns)
    holdout_columns = set(pd.read_csv(holdout_csv, nrows=0).columns)

    return sorted(train_columns - holdout_columns - {target})

def clean_records(df: pd.DataFrame, *, training_data: bool = False) -> pd.DataFrame:
    """Remove duplicates and correct known issues in the training data."""
    cleaned = df.drop_duplicates().copy()

    if cleaned["record_id"].duplicated().any():
        raise ValueError("Conflicting rows share a record_id")

    if training_data:
        rural = cleaned["destination"] == "Rural"
        cleaned.loc[rural, "weight_kg"] = cleaned.loc[rural, "weight_kg"] / 1000

        cleaned["transfers"] = cleaned["transfers"].replace(9999, float("nan"))

        target_typo = cleaned["delivery_days"] > 10
        cleaned.loc[target_typo, "delivery_days"] = (
            cleaned.loc[target_typo, "delivery_days"] / 10
        )

    return cleaned


if __name__ == "__main__":
    print(
        columns_missing_at_prediction(
            "data/raw/train.csv",
            "data/raw/holdout_features.csv",
            "delivery_days",
        )
 )
