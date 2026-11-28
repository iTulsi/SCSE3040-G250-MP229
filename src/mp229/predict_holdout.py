from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd

from src.mp229.train import FEATURES


def main() -> None:
    mlflow.set_tracking_uri("sqlite:///mlflow.db")

    holdout = pd.read_csv("data/raw/holdout_features.csv")
    model = mlflow.sklearn.load_model("models:/MP229-Parcel-Delivery@champion")

    predictions = pd.DataFrame(
        {
            "record_id": holdout["record_id"],
            "prediction": model.predict(holdout[FEATURES]),
        }
    )

    output_path = Path("predictions/holdout_predictions.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_path, index=False)

    print(f"Saved {len(predictions)} predictions to {output_path}")


if __name__ == "__main__":
    main()
