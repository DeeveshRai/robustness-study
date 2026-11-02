import pandas as pd


def type_coersion(df):
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")

    PRICE_COLS = ["open", "high", "low", "close", "adjClose", "adjHigh", "adjLow", "adjOpen", "divCash", "splitFactor"]
    COUNT_COLS = ["volume", "adjVolume"]

    for col in PRICE_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in COUNT_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce", downcast="integer")

    return df
