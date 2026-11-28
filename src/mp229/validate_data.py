import hashlib
from pathlib import Path

import pandas as pd
import yaml


def main() -> None:
    for filename in ["train.csv", "later.csv"]:
        path = Path("data/raw") / filename
        expected = yaml.safe_load(Path(f"{path}.dvc").read_text())["outs"][0]
        content = path.read_bytes()

        if hashlib.md5(content).hexdigest() != expected["md5"]:
            raise ValueError(f"{filename}: DVC hash mismatch")
        if len(content) != expected["size"]:
            raise ValueError(f"{filename}: DVC size mismatch")
        print(f"PASS: {filename} matches DVC")

    findings = yaml.safe_load(Path("reports/findings.yaml").read_text())
    if (findings["group_id"], findings["problem_id"]) != ("G250", "MP229"):
        raise ValueError("Incorrect group or problem ID")

    issues = findings["data_issues"]
    required = {
        "leaking_column", "unit_mixup", "sentinel_value",
        "duplicate_rows", "stuck_sensor", "target_typos",
        "spelling_variants",
    }
    if set(issues) != required:
        raise ValueError("Expected all seven data-issue entries")

    for name, details in issues.items():
        if type(details["present"]) is not bool:
            raise ValueError(f"{name}: present must be true or false")
        for key, value in details.items():
            if key == "present":
                continue
            if details["present"] and value is None:
                raise ValueError(f"{name}.{key}: missing detail")
            if not details["present"] and value is not None:
                raise ValueError(f"{name}.{key}: expected null")

    raw = pd.read_csv("data/raw/train.csv")
    data = raw.drop_duplicates()
    sentinel = issues["sentinel_value"]
    checks = {
        "duplicate count": issues["duplicate_rows"]["count"]
        == int(raw.duplicated().sum()),
        "sentinel count": sentinel["count"]
        == int(data[sentinel["column"]].eq(sentinel["value"]).sum()),
        "suspected target typo count": issues["target_typos"]["count"]
        == int(data["delivery_days"].gt(10).sum()),
    }
    holdout = pd.read_csv("data/raw/holdout_features.csv", nrows=0)
    leak = issues["leaking_column"]["column"]
    checks["prediction-time exclusion"] = (
        leak in raw.columns and leak not in holdout.columns
    )

    for name, passed in checks.items():
        if not passed:
            raise ValueError(f"Findings mismatch: {name}")
        print(f"PASS: {name}")

    print("PASS: findings structure and dataset integrity")


if __name__ == "__main__":
    main()
