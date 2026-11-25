def split_by_date(later, share=0.7):
    """Older rows are used for retraining, newer rows for the check."""
    ordered = later.sort_values("date", kind="mergesort").reset_index(drop=True)
    cut = int(len(ordered) * share)
    cutoff_date = ordered.iloc[cut]["date"]
    return (
        ordered[ordered["date"] < cutoff_date],
        ordered[ordered["date"] >= cutoff_date],
    )


def should_promote(mae_v1, mae_v2):
    # client rule: v2 must be at least 15% better than v1
    return mae_v2 <= 0.85 * mae_v1
