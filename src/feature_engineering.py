"""
feature_engineering.py
-----------------------
Build price-based features from raw OHLC data.
All features at time t use ONLY information available at t (no future leakage).
"""

import numpy as np
import pandas as pd


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's RSI — no lookahead bias."""
    delta = series.diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)
    # Use exponential moving average (Wilder smoothing)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs  = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add technical / price-based features to the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Must have columns: open, high, low, close  (DatetimeIndex)

    Returns
    -------
    pd.DataFrame with original OHLC columns plus feature columns.
    Rows with NaN features (warm-up period) are dropped.
    """
    feat = df.copy()
    c = feat["close"]

    # ── Return features ───────────────────────────────────────────────────────
    feat["daily_return"]  = c.pct_change(1)
    feat["return_3d"]     = c.pct_change(3)
    feat["return_5d"]     = c.pct_change(5)
    feat["return_10d"]    = c.pct_change(10)

    # ── Rolling volatility ────────────────────────────────────────────────────
    feat["volatility_5d"]  = feat["daily_return"].rolling(5).std()
    feat["volatility_10d"] = feat["daily_return"].rolling(10).std()

    # ── Moving averages ───────────────────────────────────────────────────────
    feat["ma5"]  = c.rolling(5).mean()
    feat["ma10"] = c.rolling(10).mean()
    feat["ma20"] = c.rolling(20).mean()

    # ── Close minus MA (mean-reversion signals) ───────────────────────────────
    feat["close_minus_ma5"]  = c - feat["ma5"]
    feat["close_minus_ma10"] = c - feat["ma10"]
    feat["close_minus_ma20"] = c - feat["ma20"]

    # ── Intraday range features ───────────────────────────────────────────────
    feat["high_low_range"]   = feat["high"] - feat["low"]
    feat["close_open_range"] = feat["close"] - feat["open"]

    # ── RSI ───────────────────────────────────────────────────────────────────
    feat["rsi_14"] = compute_rsi(c, period=14)

    # ── Drop warm-up rows that have NaN features ──────────────────────────────
    feature_cols = [
        "daily_return", "return_3d", "return_5d", "return_10d",
        "volatility_5d", "volatility_10d",
        "ma5", "ma10", "ma20",
        "close_minus_ma5", "close_minus_ma10", "close_minus_ma20",
        "high_low_range", "close_open_range",
        "rsi_14",
    ]
    before = len(feat)
    feat.dropna(subset=feature_cols, inplace=True)
    dropped = before - len(feat)
    if dropped:
        print(f"[feature_engineering] Dropped {dropped} warm-up rows (NaN features).")

    print(f"[feature_engineering] Feature matrix shape: {feat.shape}")
    return feat


# Convenience list for downstream modules
FEATURE_COLS = [
    "daily_return", "return_3d", "return_5d", "return_10d",
    "volatility_5d", "volatility_10d",
    "ma5", "ma10", "ma20",
    "close_minus_ma5", "close_minus_ma10", "close_minus_ma20",
    "high_low_range", "close_open_range",
    "rsi_14",
]
