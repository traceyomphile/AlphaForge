import subprocess
import sys
from pathlib import Path

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]

FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "EURUSD_features.csv"
PREDICTIONS_PATH = PROJECT_ROOT / "data" / "processed" / "EURUSD_predictions.csv"
MODEL_REPORT_PATH = PROJECT_ROOT / "reports" / "model_report.txt"
BACKTEST_REPORT_PATH = PROJECT_ROOT / "reports" / "backtest_report.txt"
EQUITY_CURVE_PATH = PROJECT_ROOT / "reports" / "equity_curve.png"
MAIN_SCRIPT_PATH = PROJECT_ROOT / "src" / "main.py"


st.set_page_config(
    page_title="AI Forex Trading Research System - Version 1",
    layout="wide",
)


def file_exists(path: Path) -> bool:
    return path.exists() and path.is_file()


def read_text_file(path: Path) -> str:
    if not file_exists(path):
        return ""

    return path.read_text(encoding="utf-8")


def parse_key_value_metrics(report_text: str) -> dict:
    """
    Parse simple report lines like:

    total_return: 0.123
    test_accuracy: 0.55
    """

    metrics = {}

    for line in report_text.splitlines():
        if ":" not in line:
            continue

        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()

        if not key or not value:
            continue

        try:
            metrics[key] = float(value)
        except ValueError:
            metrics[key] = value

    return metrics


def format_percent(value):
    try:
        return f"{float(value) * 100:.2f}%"
    except Exception:
        return "N/A"


def format_number(value):
    try:
        return f"{float(value):.4f}"
    except Exception:
        return "N/A"


def run_pipeline_from_dashboard():
    """
    Run the backend pipeline from Streamlit.

    This assumes Streamlit is launched from the project root or that this file
    exists inside frontend/.
    """

    result = subprocess.run(
        [sys.executable, str(MAIN_SCRIPT_PATH)],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
    )

    return result


st.title("AI Forex Trading Research System - Version 1")

st.sidebar.header("Settings")
st.sidebar.write("Selected pair: **EUR/USD only**")
st.sidebar.write("Timeframe: **Daily only**")

st.sidebar.warning(
    "Research only. Not financial advice. "
    "This dashboard does not place trades."
)

if st.sidebar.button("Run / Refresh Pipeline"):
    with st.spinner("Running backend pipeline..."):
        result = run_pipeline_from_dashboard()

    if result.returncode == 0:
        st.sidebar.success("Pipeline completed successfully.")
        st.sidebar.code(result.stdout)
    else:
        st.sidebar.error("Pipeline failed.")
        st.sidebar.code(result.stderr or result.stdout)


required_files = [
    FEATURES_PATH,
    PREDICTIONS_PATH,
    MODEL_REPORT_PATH,
    BACKTEST_REPORT_PATH,
    EQUITY_CURVE_PATH,
]

missing_files = [path for path in required_files if not file_exists(path)]

if missing_files:
    st.error("Some required output files do not exist yet.")

    st.write("Run this command first:")

    st.code("python src/main.py", language="bash")

    st.write("Or use the sidebar button to run the pipeline.")

    with st.expander("Missing files"):
        for path in missing_files:
            st.write(f"- `{path.relative_to(PROJECT_ROOT)}`")

    st.stop()


predictions_df = pd.read_csv(PREDICTIONS_PATH)
features_df = pd.read_csv(FEATURES_PATH)

if "date" in predictions_df.columns:
    predictions_df["date"] = pd.to_datetime(predictions_df["date"], errors="coerce")

if "date" in features_df.columns:
    features_df["date"] = pd.to_datetime(features_df["date"], errors="coerce")


st.header("Latest Signal")

if predictions_df.empty:
    st.warning("Predictions file exists but is empty. That is not useful, obviously.")
else:
    latest = predictions_df.sort_index().iloc[-1]

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric("Date", str(latest.get("date", "N/A"))[:10])
    col2.metric("Pair", latest.get("pair", "EUR/USD"))
    col3.metric("Signal", latest.get("signal", "N/A"))

    confidence = latest.get("confidence", None)

    if pd.notna(confidence):
        col4.metric("Confidence", format_percent(confidence))
    else:
        col4.metric("Confidence", "N/A")

    close_price = latest.get("close", None)

    if pd.notna(close_price):
        col5.metric("Latest Close", format_number(close_price))
    else:
        col5.metric("Latest Close", "N/A")

    next_return = latest.get("next_return", None)

    if pd.notna(next_return):
        st.info(f"Next-day return in historical data: {format_percent(next_return)}")
    else:
        st.info("Next-day return unavailable for the latest row, because tomorrow has not happened yet. Tragic.")


st.header("Model Performance")

model_report_text = read_text_file(MODEL_REPORT_PATH)
model_metrics = parse_key_value_metrics(model_report_text)

m1, m2, m3, m4, m5 = st.columns(5)

m1.metric("Best Model", str(model_metrics.get("best_model", "N/A")))
m2.metric("Accuracy", format_percent(model_metrics.get("test_accuracy", None)))
m3.metric("Precision", format_percent(model_metrics.get("test_macro_precision", None)))
m4.metric("Recall", format_percent(model_metrics.get("test_macro_recall", None)))
m5.metric("F1-score", format_percent(model_metrics.get("test_macro_f1", None)))

with st.expander("Full model report"):
    st.text(model_report_text)


st.header("Backtest Performance")

backtest_report_text = read_text_file(BACKTEST_REPORT_PATH)
backtest_metrics = parse_key_value_metrics(backtest_report_text)

b1, b2, b3, b4, b5 = st.columns(5)

b1.metric("Total Return", format_percent(backtest_metrics.get("total_return", None)))
b2.metric("Max Drawdown", format_percent(backtest_metrics.get("max_drawdown", None)))
b3.metric("Win Rate", format_percent(backtest_metrics.get("win_rate", None)))
b4.metric("Trades", str(backtest_metrics.get("number_of_trades", "N/A")))
b5.metric("Sharpe", format_number(backtest_metrics.get("sharpe_ratio", None)))

b6, b7 = st.columns(2)

b6.metric("Profit Factor", str(backtest_metrics.get("profit_factor", "N/A")))
b7.metric("Annualized Return", format_percent(backtest_metrics.get("annualized_return", None)))

with st.expander("Full backtest report"):
    st.text(backtest_report_text)


st.header("Equity Curve")

if file_exists(EQUITY_CURVE_PATH):
    st.image(str(EQUITY_CURVE_PATH), caption="Cumulative return / equity curve")
else:
    st.warning("Equity curve image not found.")


st.header("Data Preview")

tab1, tab2 = st.tabs(["Processed Features", "Predictions"])

with tab1:
    st.subheader("Latest processed feature rows")
    st.dataframe(features_df.tail(20), use_container_width=True)

with tab2:
    st.subheader("Latest prediction rows")
    st.dataframe(predictions_df.tail(20), use_container_width=True)


st.warning(
    "This is not financial advice. Backtest performance does not guarantee live trading performance. "
    "Forex trading is risky. This frontend only displays research outputs and does not place trades."
)