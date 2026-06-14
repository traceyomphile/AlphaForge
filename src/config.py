from pathlib import Path

# Root project directory
BASE_DIR = Path(__file__).resolve().parents[1]

# Data paths
RAW_DATA_PATH = BASE_DIR / "data" / "raw" / "EURUSD_DAILY.csv"
PROCESSED_DATA_PATH = BASE_DIR / "data" / "processed" / "EURUSD_features.csv"

# Output paths
MODELS_DIR = BASE_DIR / "models_saved"
REPORTS_DIR = BASE_DIR / "reports"

BEST_MODEL_PATH = MODELS_DIR / "best_model.pkl"
SCALER_PATH = MODELS_DIR / "scaler.pkl"

MODEL_REPORT_PATH = REPORTS_DIR / "model_report.txt"
BACKTEST_REPORT_PATH = REPORTS_DIR / "backtest_report.txt"
EQUITY_CURVE_PATH = REPORTS_DIR / "equity_curve.png"

# Labeling
THRESHOLD = 0.001  # 0.1%

BUY_LABEL = 1
HOLD_LABEL = 0
SELL_LABEL = -1

LABEL_NAMES = {
    SELL_LABEL: "SELL",
    HOLD_LABEL: "HOLD",
    BUY_LABEL: "BUY",
}

# Backtest
TRANSACTION_COST = 0.0001

# Time-series split
TRAIN_SIZE = 0.70
VALIDATION_SIZE = 0.15
TEST_SIZE = 0.15

# Reproducibility
RANDOM_STATE = 42

# Features used by the model
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