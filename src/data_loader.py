"""
data_loader.py
--------------
Loads and validates the raw EUR/USD daily CSV file.
Expected columns: date, open, high, low, close  (volume optional).
"""

import pandas as pd
from src.config import RAW_DATA_PATH


def load_price_data(filepath: str = str(RAW_DATA_PATH)) -> pd.DataFrame:
    """
    Load the raw OHLC CSV and return a clean, sorted DataFrame.

    Returns
    -------
    pd.DataFrame  with columns: date, open, high, low, close
                  index: DatetimeIndex
    """
    print(f"[data_loader] Loading data from: {filepath}")
    df = pd.read_csv(filepath)

    # ── Normalise column names 
    df.columns = [c.strip().lower() for c in df.columns]

    # ── Locate and parse the date column 
    date_col_candidates = ["date", "time", "datetime", "timestamp"]
    date_col = None
    for c in date_col_candidates:
        if c in df.columns:
            date_col = c
            break
    if date_col is None:
        raise ValueError(
            "Could not find a date column. Expected one of: "
            + str(date_col_candidates)
        )

    df[date_col] = pd.to_datetime(df[date_col])
    df = df.rename(columns={date_col: "date"})
    df = df.set_index("date").sort_index()

    # ── Require OHLC ──────
    required = ["open", "high", "low", "close"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # ── Keep only useful columns (drop volume if present but don't require it) ─
    keep = [c for c in required + ["volume"] if c in df.columns]
    df = df[keep].copy()

    # ── Cast to float ───────
    for col in keep:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # ── Drop rows where OHLC are all NaN ───────
    df.dropna(subset=required, how="all", inplace=True)

    # ── Basic sanity checks ────────
    invalid_high = (df["high"] < df["low"]).sum()
    if invalid_high > 0:
        print(f"[data_loader] WARNING: {invalid_high} rows where high < low — dropping them.")
        df = df[df["high"] >= df["low"]]

    print(f"[data_loader] Loaded {len(df)} rows  |  {df.index[0].date()} → {df.index[-1].date()}")
    return df
