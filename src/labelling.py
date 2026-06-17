import pandas as pd

from src.config import DEFAULT_LABEL_THRESHOLD, LABEL_BUY, LABEL_HOLD, LABEL_SELL


def create_labels(
    df: pd.DataFrame,
    threshold: float = DEFAULT_LABEL_THRESHOLD,
    drop_unlabeled: bool = False,
) -> pd.DataFrame:
    """
    Create target labels using next-day return.

    next_return = close[t+1] / close[t] - 1

    BUY  =  1 if next_return > threshold
    SELL = -1 if next_return < -threshold
    HOLD =  0 otherwise

    The final row will have no next_return because future data is unavailable.
    """

    df = df.copy()

    df["next_return"] = df["close"].shift(-1) / df["close"] - 1

    df["target"] = LABEL_HOLD
    df.loc[df["next_return"] > threshold, "target"] = LABEL_BUY
    df.loc[df["next_return"] < -threshold, "target"] = LABEL_SELL

    df.loc[df["next_return"].isna(), "target"] = pd.NA

    if drop_unlabeled:
        df = df.dropna(subset=["target"]).copy()
        df["target"] = df["target"].astype(int)

    return df