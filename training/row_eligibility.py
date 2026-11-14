"""Select complete supervised rows without altering feature/target alignment."""

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype, is_bool_dtype

from data_pipeline.contracts import prepare_daily_observations
from features.schema import FEATURE_COLUMNS, FEATURE_WARMUP_ROWS


def select_model_rows(data):
    """Return (eligible, excluded) from a complete single-asset feature history.

    Call after generating targets and before date-window selection. Earlier
    history supplies indicator warm-up even when training starts much later.
    Missing/undefined numbers are excluded, never filled using future values.
    This checks completeness, not label availability at a retraining cutoff.
    """
    required = FEATURE_COLUMNS + ["target_return", "target_entry_date", "target_exit_date"]
    missing = [column for column in required if column not in data]
    if missing:
        raise ValueError(f"Missing model columns: {missing}")
    result = prepare_daily_observations(data)
    for column in FEATURE_COLUMNS + ["target_return"]:
        if not is_numeric_dtype(result[column]) or is_bool_dtype(result[column]):
            raise ValueError(f"Expected numeric model column: {column}")
    for column in ("target_entry_date", "target_exit_date"):
        result[column] = pd.to_datetime(result[column], utc=True, errors="raise")

    warmup = pd.Series(np.arange(len(result)) < FEATURE_WARMUP_ROWS, index=result.index)
    finite_features = np.isfinite(result[FEATURE_COLUMNS].to_numpy(dtype=float, na_value=np.nan)).all(axis=1)
    complete_target = (
        np.isfinite(result["target_return"].to_numpy(dtype=float, na_value=np.nan))
        & result["target_entry_date"].notna()
        & result["target_exit_date"].notna()
    )
    known_dates = result["target_entry_date"].notna() & result["target_exit_date"].notna()
    ordered_target = ((result["date"] < result["target_entry_date"])
                      & (result["target_entry_date"] < result["target_exit_date"]))
    if (known_dates & ~ordered_target).any():
        raise ValueError("Target dates must follow feature date < entry date < exit date")

    reason = pd.Series("", index=result.index)
    for mask, label in ((warmup, "warmup"), (~finite_features, "invalid_features"),
                        (~complete_target, "incomplete_target")):
        reason.loc[mask] += label + ";"
    eligible = result.loc[reason.eq("")].copy()
    excluded = result.loc[reason.ne("")].copy()
    excluded["exclusion_reason"] = reason.loc[excluded.index].str.rstrip(";")
    return eligible, excluded
