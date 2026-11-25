import pandas as pd


def columns_missing_at_prediction(
    train_csv: str, holdout_csv: str, target: str
) -> list[str]:
    train_columns = set(pd.read_csv(train_csv, nrows=0).columns)
    holdout_columns = set(pd.read_csv(holdout_csv, nrows=0).columns)

    return sorted(train_columns - holdout_columns - {target})


def clean_records(df: pd.DataFrame) -> pd.DataFrame:
    """Remove exact copies; reject IDs that still have conflicting records."""
    cleaned = df.drop_duplicates().copy()

    if cleaned["record_id"].duplicated().any():
        raise ValueError("Conflicting rows share a record_id")

    return cleaned


if __name__ == "__main__":
    print(
        columns_missing_at_prediction(
            "data/raw/train.csv",
            "data/raw/holdout_features.csv",
            "delivery_days",
        )
 )
