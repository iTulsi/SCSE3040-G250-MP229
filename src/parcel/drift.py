import numpy as np
import pandas as pd

NUMERIC_COLUMNS = [
    "distance_km",
    "weight_kg",
    "hub_backlog",
    "pin_code_tier",
    "declared_value_rs",
    "transfers",
]


def psi(expected, actual, bins=10):
    """PSI of one column. Bin edges come from the training values."""
    expected = pd.Series(expected).dropna().to_numpy(dtype=float)
    actual = pd.Series(actual).dropna().to_numpy(dtype=float)

    # columns with only a few distinct values end up with fewer bins
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf

    expected_share = np.histogram(expected, edges)[0] / len(expected)
    actual_share = np.histogram(actual, edges)[0] / len(actual)

    # avoid log(0) for empty bins
    expected_share = np.clip(expected_share, 1e-4, None)
    actual_share = np.clip(actual_share, 1e-4, None)

    diff = actual_share - expected_share
    return float(np.sum(diff * np.log(actual_share / expected_share)))


def psi_report(train, later):
    return {col: round(psi(train[col], later[col]), 4) for col in NUMERIC_COLUMNS}
