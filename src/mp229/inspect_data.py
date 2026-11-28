import pandas as pd

from src.mp229.data_checks import columns_missing_at_prediction
from src.mp229.train import NUMERIC


def main() -> None:
    raw = pd.read_csv("data/raw/train.csv")
    data = raw.drop_duplicates().copy()
    data["date"] = pd.to_datetime(data["date"], errors="raise")
    ordered = data.sort_values("date", kind="mergesort")

    print("Extra duplicate rows:", raw.duplicated().sum())
    print("Missing at prediction:", columns_missing_at_prediction(
        "data/raw/train.csv",
        "data/raw/holdout_features.csv",
        "delivery_days",
    ))

    print("\nWeight by destination:")
    print(data.groupby("destination")["weight_kg"].agg(["min", "median", "max"]))

    print("\nTransfers value counts:")
    print(data["transfers"].value_counts().sort_index())

    print("\nSuspected target typos:")
    suspect = data.loc[data["delivery_days"] > 10, [
        "record_id", "delivery_days", "days_from_tracking_page",
    ]].copy()
    suspect["corrected_days"] = suspect["delivery_days"] / 10
    print(suspect.to_string(index=False))
    print("Suspected target typo count:", len(suspect))

    later = pd.read_csv("data/raw/later.csv")
    for column in ["service", "destination"]:
        print(f"\n{column} training categories:", sorted(data[column].unique()))
        print(f"{column} later categories:", sorted(later[column].unique()))

    print("\nLongest consecutive equal-value runs in date order:")
    for column in NUMERIC:
        runs = ordered[column].ne(ordered[column].shift()).cumsum()
        summary = ordered.groupby(runs).agg(
            value=(column, "first"),
            rows=(column, "size"),
            start_date=("date", "min"),
            end_date=("date", "max"),
        )
        print(f"\n{column}:")
        print(summary.nlargest(1, "rows").to_string(index=False))


if __name__ == "__main__":
    main()
