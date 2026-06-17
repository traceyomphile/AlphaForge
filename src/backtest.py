import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import DEFAULT_TRANSACTION_COST, EQUITY_CURVE_PATH
from src.metrics import (
    calculate_annualized_return,
    calculate_max_drawdown,
    calculate_profit_factor,
    calculate_sharpe_ratio,
)


def run_backtest(
    predictions_df: pd.DataFrame,
    transaction_cost: float = DEFAULT_TRANSACTION_COST,
) -> tuple[pd.DataFrame, dict]:
    """
    Simple next-day backtest using model predictions.

    Trading rules:
    - BUY  => long EUR/USD for next day
    - SELL => short EUR/USD for next day
    - HOLD => flat

    Return logic:
    - BUY return  = next_return - transaction_cost
    - SELL return = -next_return - transaction_cost
    - HOLD return = 0
    """

    df = predictions_df.copy()

    df = df.dropna(subset=["next_return", "prediction"]).copy()
    df["prediction"] = df["prediction"].astype(int)

    df["strategy_return"] = 0.0

    buy_mask = df["prediction"] == 1
    sell_mask = df["prediction"] == -1

    df.loc[buy_mask, "strategy_return"] = (
        df.loc[buy_mask, "next_return"] - transaction_cost
    )

    df.loc[sell_mask, "strategy_return"] = (
        -df.loc[sell_mask, "next_return"] - transaction_cost
    )

    df["equity_curve"] = (1 + df["strategy_return"]).cumprod()

    total_return = float(df["equity_curve"].iloc[-1] - 1) if not df.empty else 0.0
    number_of_days = len(df)

    active_trades = df[df["prediction"] != 0].copy()
    number_of_trades = len(active_trades)

    if number_of_trades > 0:
        win_rate = float((active_trades["strategy_return"] > 0).mean())
        average_trade_return = float(active_trades["strategy_return"].mean())
    else:
        win_rate = 0.0
        average_trade_return = 0.0

    max_drawdown = calculate_max_drawdown(df["equity_curve"]) if not df.empty else 0.0
    sharpe_ratio = calculate_sharpe_ratio(df["strategy_return"])
    profit_factor = calculate_profit_factor(active_trades["strategy_return"])
    annualized_return = calculate_annualized_return(total_return, number_of_days)

    metrics = {
        "total_return": total_return,
        "annualized_return": annualized_return,
        "max_drawdown": max_drawdown,
        "win_rate": win_rate,
        "average_trade_return": average_trade_return,
        "number_of_trades": int(number_of_trades),
        "profit_factor": profit_factor,
        "sharpe_ratio": sharpe_ratio,
    }

    return df, metrics


def save_equity_curve_chart(
    backtest_df: pd.DataFrame,
    output_path=EQUITY_CURVE_PATH,
) -> None:
    """
    Save cumulative return / equity curve chart.
    """

    if backtest_df.empty:
        return

    plt.figure(figsize=(12, 6))
    plt.plot(backtest_df.index, backtest_df["equity_curve"])
    plt.title("EUR/USD Strategy Equity Curve - Version 1")
    plt.xlabel("Date")
    plt.ylabel("Equity Curve")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def format_backtest_report(metrics: dict) -> str:
    """
    Create readable backtest report.
    """

    def fmt_percent(value):
        return f"{value * 100:.2f}%"

    profit_factor = metrics["profit_factor"]
    if np.isinf(profit_factor):
        profit_factor_text = "inf"
    else:
        profit_factor_text = f"{profit_factor:.6f}"

    lines = []

    lines.append("AI Forex Trading Research System - Version 1")
    lines.append("Backtest Report")
    lines.append("=" * 60)
    lines.append("")
    lines.append(f"total_return: {metrics['total_return']:.6f}")
    lines.append(f"annualized_return: {metrics['annualized_return']:.6f}")
    lines.append(f"max_drawdown: {metrics['max_drawdown']:.6f}")
    lines.append(f"win_rate: {metrics['win_rate']:.6f}")
    lines.append(f"average_trade_return: {metrics['average_trade_return']:.6f}")
    lines.append(f"number_of_trades: {metrics['number_of_trades']}")
    lines.append(f"profit_factor: {profit_factor_text}")
    lines.append(f"sharpe_ratio: {metrics['sharpe_ratio']:.6f}")
    lines.append("")
    lines.append("Readable Summary")
    lines.append("-" * 60)
    lines.append(f"Total return: {fmt_percent(metrics['total_return'])}")
    lines.append(f"Annualized return: {fmt_percent(metrics['annualized_return'])}")
    lines.append(f"Max drawdown: {fmt_percent(metrics['max_drawdown'])}")
    lines.append(f"Win rate: {fmt_percent(metrics['win_rate'])}")
    lines.append(f"Average trade return: {fmt_percent(metrics['average_trade_return'])}")
    lines.append(f"Number of trades: {metrics['number_of_trades']}")
    lines.append(f"Profit factor: {profit_factor_text}")
    lines.append(f"Sharpe ratio: {metrics['sharpe_ratio']:.4f}")
    lines.append("")
    lines.append("Warning:")
    lines.append(
        "This backtest is simplified. It ignores many real-world issues such as slippage, "
        "liquidity, broker execution, overnight costs, spread changes, and psychological damage."
    )

    return "\n".join(lines)

if __name__ == "__main__":
    print("This is the backtest module. It provides functions to run a simple backtest on model predictions and generate reports.")