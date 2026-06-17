""" 
feature engineering 

create features that describe what the market has been doing."""

import pandas as pd
import numpy as np

def create_features(df):
    """
    create technical and price based features.

    parameters
    df: pd.DataFrame
        must contain columns: open, high, low, close
    
    returns
    pd.DataFrame
    """
    df = df.copy()
    #--return: tells how much the price has changed since the last day.
    df["return"] = df["close"].pct_change()
    df["return_3d"] = df["close"].pct_change(3)
    df["return_5d"] = df["close"].pct_change(5)
    df["return_10d"] = df["close"].pct_change(10)

    #--volatility: measures how wild the price movement is.
    df["volatility_5"]= (df["return"].rolling(window=5).std())
    df["volatility_10"]= (df["return"].rolling(window=10).std())
    
    #--moving averages: avarage closing price over certain number of days.
    df["ma_5"] = df["close"].rolling(window=5).mean()
    df["ma_10"] = df["close"].rolling(window=10).mean()
    df["ma_20"] = df["close"].rolling(window=20).mean()
    
    #distance from moving averages: how far the current price is from the moving average.
    df["close_minus_ma_5"] = df["close"] - df["ma_5"]
    df["close_minus_ma_10"] = df["close"] - df["ma_10"]
    df["close_minus_ma_20"] = df["close"] - df["ma_20"]
    
    #--measures daily movement
    df["high_low_range"] = df["high"] - df["low"]
    #--measures direction of the day.
    df["close_open_range"] = df["close"] - df["open"]
    
    #--relative strength index: measures the speed and change of price movements.
    delta =df["close"].diff()
    gain=delta.clip(lower=0)
    loss=-1*delta.clip(upper=0)
    avg_gain=gain.rolling(window=14).mean()
    avg_loss=loss.rolling(window=14).mean()
    rsi=avg_gain/avg_loss
    df["rsi_14"]=100-(100/(1+rsi))

    df.dropna(inplace=True)
    return df