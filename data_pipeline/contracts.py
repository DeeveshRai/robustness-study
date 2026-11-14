"""Shared contract for one asset's daily observations."""

import pandas as pd


def require_daily_frequency(frequency):
    """The feature windows and annualisation currently assume daily bars."""
    if frequency != "daily":
        raise ValueError("Only frequency='daily' is supported by the feature pipeline")
    return frequency


def prepare_daily_observations(data, asset=None):
    """Copy data, parse UTC dates, and validate one ordered asset history.

    Sorting and duplicate resolution belong to cleaning. Reject them here so
    rolling calculations cannot silently run on an invalid history.
    """
    result = data.copy()
    result["date"] = pd.to_datetime(result["date"], utc=True, errors="raise")
    if result.empty or result["date"].isna().any():
        raise ValueError("Observations must contain non-missing dates")
    if asset is not None:
        if not isinstance(asset, str) or not asset.strip():
            raise ValueError("Asset must be a non-empty symbol")
        asset = asset.strip().upper()
        if "asset" in result and not result["asset"].eq(asset).all():
            raise ValueError("Existing asset values do not match the requested asset")
        result["asset"] = asset
    if "asset" not in result or result["asset"].isna().any():
        raise ValueError("An asset symbol is required")
    if result["asset"].nunique() != 1:
        raise ValueError("Process each asset separately before combining datasets")
    if not isinstance(result["asset"].iloc[0], str) or not result["asset"].iloc[0].strip():
        raise ValueError("Asset must be a non-empty symbol")
    if result["date"].dt.normalize().duplicated().any():
        raise ValueError("Expected one observation per UTC date for this asset")
    if not result["date"].is_monotonic_increasing:
        raise ValueError("Observations must be sorted by date before feature generation")
    return result.reset_index(drop=True)
