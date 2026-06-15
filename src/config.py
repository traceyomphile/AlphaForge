"""
config.py
---------
Central configuration for the AI Forex Trading Research System - Version 1.
All paths, hyperparameters, and constants live here.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models_saved"
REPORTS_DIR = PROJECT_ROOT / "reports"

RAW_DATA_PATH = DATA_RAW_DIR / "EURUSD_DAILY.csv"
FEATURES_PATH = DATA_PROCESSED_DIR / "EURUSD_features.csv"
PREDICTIONS_PATH = DATA_PROCESSED_DIR / "EURUSD_predictions.csv"

MODEL_REPORT_PATH = REPORTS_DIR / "model_report.txt"
BACKTEST_REPORT_PATH = REPORTS_DIR / "backtest_report.txt"
EQUITY_CURVE_PATH = REPORTS_DIR / "equity_curve.png"

BEST_MODEL_PATH = MODELS_DIR / "best_model.pkl"
SCALER_PATH = MODELS_DIR / "scaler.pkl"

PAIR = "EUR/USD"
TIMEFRAME = "Daily"

LABEL_BUY = 1
LABEL_HOLD = 0
LABEL_SELL = -1

LABEL_TO_SIGNAL = {
    LABEL_BUY: "BUY",
    LABEL_HOLD: "HOLD",
    LABEL_SELL: "SELL",
}

SIGNAL_TO_LABEL = {
    "BUY": LABEL_BUY,
    "HOLD": LABEL_HOLD,
    "SELL": LABEL_SELL,
}

LABEL_ORDER = [LABEL_SELL, LABEL_HOLD, LABEL_BUY]
LABEL_NAMES = ["SELL", "HOLD", "BUY"]

DEFAULT_LABEL_THRESHOLD = 0.001
DEFAULT_TRANSACTION_COST = 0.0001

TRAIN_SIZE = 0.70
VAL_SIZE = 0.15
# TEST_RATIO  = remaining 0.15 (implicit)

RANDOM_STATE = 42

FEATURE_COLUMNS = [
    "daily_return",
    "return_3d",
    "return_5d",
    "return_10d",
    "volatility_5d",
    "volatility_10d",
    "ma_5",
    "ma_10",
    "ma_20",
    "close_minus_ma_5",
    "close_minus_ma_10",
    "close_minus_ma_20",
    "high_low_range",
    "close_open_range",
    "rsi_14",
]

# ── Backtesting 
TRANSACTION_COST = 0.0001   # ~1 pip spread estimate

# ── Random seed 
RANDOM_SEED = 42

# ── Trading pair / timeframe (for display) 
PAIR      = "EUR/USD"
TIMEFRAME = "Daily"

YAHOO_TICKER = "EURUSD=X"

AUTO_DOWNLOAD_DATA = True
FORCE_DOWNLOAD_DATA = False

DOWNLOAD_START_DATE = "2003-01-01"
DOWNLOAD_END_DATE = None

def ensure_directories() -> None:
    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)