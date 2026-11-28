import pandas as pd
import pytest

from src.mp229.data_checks import clean_records


def test_training_corrections_preserve_original():
    original = pd.DataFrame({
        "record_id": ["A", "B"],
        "destination": ["Rural", "Metro"],
        "weight_kg": [12000.0, 8.0],
        "transfers": [9999, 2],
        "delivery_days": [42.0, 5.0],
    })
    before = original.copy(deep=True)

    result = clean_records(original, training_data=True)

    assert result["weight_kg"].tolist() == [12.0, 8.0]
    assert pd.isna(result.loc[0, "transfers"])
    assert result.loc[1, "transfers"] == 2
    assert result["delivery_days"].tolist() == [4.2, 5.0]
    pd.testing.assert_frame_equal(original, before)


def test_later_records_are_unchanged():
    original = pd.DataFrame({
        "record_id": ["A"],
        "destination": ["Rural"],
        "weight_kg": [14.0],
        "transfers": [3],
        "delivery_days": [12.0],
    })

    result = clean_records(original)

    pd.testing.assert_frame_equal(result, original)


def test_exact_duplicates_are_removed():
    original = pd.DataFrame({
        "record_id": ["A", "A", "B"],
        "weight_kg": [2.0, 2.0, 3.0],
    })

    result = clean_records(original)

    assert result["record_id"].tolist() == ["A", "B"]
    assert len(original) == 3


def test_conflicting_record_ids_are_rejected():
    original = pd.DataFrame({
        "record_id": ["A", "A"],
        "weight_kg": [2.0, 3.0],
    })

    with pytest.raises(ValueError, match="Conflicting rows share a record_id"):
        clean_records(original)
