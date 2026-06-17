from pathlib import Path

import pandas as pd
import yfinance as yf
import requests_cache 
import requests


from src.config import (
    DOWNLOAD_END_DATE,
    DOWNLOAD_START_DATE,
    RAW_DATA_PATH,
    YAHOO_TICKER,
    ensure_directories,
)


def clean_yahoo_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean Yahoo Finance EUR/USD data into the format required by the V1 pipeline:

    date, open, high, low, close

    Volume is ignored for V1 because forex volume from free data sources is often
    unavailable, inconsistent, or feed-specific.
    """

    if df is None or df.empty:
        raise ValueError(
            "Yahoo Finance returned an empty dataset. "
            "The ticker may have failed, the date range may be invalid, "
            "or Yahoo may simply be having a moment."
        )

    # yfinance can sometimes return MultiIndex columns.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.reset_index()

    df.columns = [
        str(col).strip().lower().replace(" ", "_")
        for col in df.columns
    ]

    rename_map = {
        "date": "date",
        "datetime": "date",
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close",
    }

    df = df.rename(columns=rename_map)

    required_columns = ["date", "open", "high", "low", "close"]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Downloaded Yahoo data is missing required columns: {missing_columns}. "
            f"Available columns: {list(df.columns)}"
        )

    df = df[required_columns].copy()

    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    for col in ["open", "high", "low", "close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=required_columns)
    df = df.sort_values("date").reset_index(drop=True)

    if df.empty:
        raise ValueError(
            "Downloaded data became empty after cleaning. "
            "That means the raw data was unusable."
        )

    # Save dates in simple CSV-friendly format.
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")

    return df


def download_eurusd_daily_data(
    output_path: Path = RAW_DATA_PATH,
    start_date: str = DOWNLOAD_START_DATE,
    end_date: str | None = DOWNLOAD_END_DATE,
) -> pd.DataFrame:
    """
    Download EUR/USD daily OHLC data from Yahoo Finance and save it as CSV.

    Output:
    data/raw/EURUSD_DAILY.csv
    """

    ensure_directories()

    print("=" * 60)
    print("Downloading EUR/USD daily data from Yahoo Finance")
    print("=" * 60)
    print(f"Ticker: {YAHOO_TICKER}")
    print(f"Start date: {start_date}")
    print(f"End date: {end_date if end_date else 'latest available'}")
    print("")

    # Provide a custom session with standard browser headers
    session = requests.Session()
    session.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/91.0.4472.124 Safari/537.36"
        )
    })
    requests_cache.install_cache(
        "yahoo_cache",
        backend="sqlite",
        expire_after=3600,  # Cache expires after 1 hour
        session=session,
    )

    df = None

    try:
        eurusd = yf.Ticker(YAHOO_TICKER)
        df = eurusd.history(
            start=start_date,
            end=end_date,
            interval="1d",
            auto_adjust=False
        )
        """
        df = yf.download(
            tickers=YAHOO_TICKER,
            start=start_date,
            end=end_date,
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=False,
            multi_level_index=False,
        )
        """
    except Exception as exc:
        raise RuntimeError(
            "Failed to download EUR/USD data from Yahoo Finance. "
            "Check your internet connection and yfinance installation."
        ) from exc

    if df is None or df.empty:
        raise RuntimeError("Failed to clean downloaded EUR/USD data.")

    clean_df = clean_yahoo_data(df) 

    output_path.parent.mkdir(parents=True, exist_ok=True)

    clean_df.to_csv(output_path, index=False)

    print("Download complete.")
    print(f"Rows saved: {len(clean_df)}")
    print(f"First date: {clean_df['date'].iloc[0]}")
    print(f"Last date: {clean_df['date'].iloc[-1]}")
    print(f"Saved to: {output_path}")
    print("")

    return clean_df


def ensure_eurusd_data_exists(
    raw_data_path: Path = RAW_DATA_PATH,
    force_download: bool = False,
) -> None:
    """
    Ensure that data/raw/EURUSD_DAILY.csv exists before the pipeline loads data.

    If the file exists:
        do nothing unless force_download=True.

    If the file does not exist:
        download it from Yahoo Finance.
    """

    ensure_directories()

    if raw_data_path.exists() and not force_download:
        print(f"Raw data already exists: {raw_data_path}")
        print("Skipping download.")
        print("")
        return

    if force_download:
        print("Force download enabled. Existing raw data will be replaced.")
    else:
        print("Raw data file not found. Downloading now.")

    download_eurusd_daily_data(output_path=raw_data_path)


if __name__ == "__main__":
    download_eurusd_daily_data()