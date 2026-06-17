import numpy as np
import pandas as pd


def calculate_max_drawdown(equity_curve: pd.Series) -> float:
    """
    Max drawdown from an equity curve.

    Returns a negative number, e.g. -0.12 means -12%.
    """

    running_max = equity_curve.cummax()
    drawdown = equity_curve / running_max - 1
    return float(drawdown.min())


def calculate_sharpe_ratio(returns: pd.Series, periods_per_year: int = 252) -> float:
    """
    Annualized Sharpe ratio using daily returns.
    Risk-free rate assumed to be 0 for Version 1.
    """

    returns = returns.dropna()

    if returns.empty:
        return 0.0

    std = returns.std()

    if std == 0 or np.isnan(std):
        return 0.0

    return float((returns.mean() / std) * np.sqrt(periods_per_year))


def calculate_profit_factor(returns: pd.Series) -> float:
    """
    Profit factor = gross profit / absolute gross loss.
    """

    returns = returns.dropna()

    gross_profit = returns[returns > 0].sum()
    gross_loss = returns[returns < 0].sum()

    if gross_loss == 0:
        return float("inf") if gross_profit > 0 else 0.0

    return float(gross_profit / abs(gross_loss))


def calculate_annualized_return(
    total_return: float,
    number_of_days: int,
    periods_per_year: int = 252,
) -> float:
    """
    Annualized return from total return.
    """

    if number_of_days <= 0:
        return 0.0

    return float((1 + total_return) ** (periods_per_year / number_of_days) - 1)