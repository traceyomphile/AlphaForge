"""
config.py
---------
Central configuration for the AI Forex Trading Research System - Version 1.
All paths, hyperparameters, and constants live here.
"""

import os

# ── Project root 
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── Data paths 
RAW_DATA_PATH       = os.path.join(BASE_DIR, "data", "raw",       "EURUSD_DAILY.csv")
PROCESSED_FEATURES  = os.path.join(BASE_DIR, "data", "processed", "EURUSD_features.csv")
PROCESSED_PREDS     = os.path.join(BASE_DIR, "data", "processed", "EURUSD_predictions.csv")

# ── Model / scaler paths 
MODEL_PATH  = os.path.join(BASE_DIR, "models_saved", "best_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models_saved", "scaler.pkl")

# ── Report paths 
MODEL_REPORT_PATH   = os.path.join(BASE_DIR, "reports", "model_report.txt")
BACKTEST_REPORT_PATH= os.path.join(BASE_DIR, "reports", "backtest_report.txt")
EQUITY_CURVE_PATH   = os.path.join(BASE_DIR, "reports", "equity_curve.png")

# ── Labeling 
LABEL_THRESHOLD = 0.001   # 0.1% move → BUY or SELL; else HOLD
LABEL_MAP = {"BUY": 1, "HOLD": 0, "SELL": -1}
LABEL_NAMES = {1: "BUY", 0: "HOLD", -1: "SELL"}

# ── Train / val / test split ratios 
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
# TEST_RATIO  = remaining 0.15 (implicit)

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