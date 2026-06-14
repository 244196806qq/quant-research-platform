import pandas as pd
import numpy as np


def ma(df: pd.DataFrame, short: int = 10, mid: int = 50, long: int = 200) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out[f"MA{short}"] = df["Adj Close"].rolling(short).mean()
    out[f"MA{mid}"] = df["Adj Close"].rolling(mid).mean()
    out[f"MA{long}"] = df["Adj Close"].rolling(long).mean()
    return out


def zscore(df: pd.DataFrame, window: int = 50) -> pd.Series:
    mean = df["Adj Close"].rolling(window).mean()
    std = df["Adj Close"].rolling(window).std()
    return (df["Adj Close"] - mean) / std


def momentum(df: pd.DataFrame, window: int = 50) -> pd.Series:
    ma = df["Adj Close"].rolling(window).mean()
    return df["Adj Close"] / ma


def rsi(df: pd.DataFrame, window: int = 14) -> pd.Series:
    delta = df["Adj Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def rolling_std(df: pd.DataFrame, window: int = 20) -> pd.Series:
    return df["Market_Return"].rolling(window).std()


def atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    high_low = df["High"] - df["Low"]
    high_prev_close = (df["High"] - df["Close"].shift(1)).abs()
    low_prev_close = (df["Low"] - df["Close"].shift(1)).abs()
    tr = pd.concat([high_low, high_prev_close, low_prev_close], axis=1).max(axis=1)
    return tr.rolling(window).mean()


def ewma_vol(df: pd.DataFrame, span: int = 20) -> pd.Series:
    return df["Market_Return"].ewm(span=span).std()


def annual_std(df: pd.DataFrame, window: int = 20, periods: int = 252) -> pd.Series:
    return df["Market_Return"].rolling(window).std() * (periods ** 0.5)

def strategy_drawdown(df: pd.DataFrame) -> pd.Series:
    rolling_peak = df["Strategy_Equity"].cummax()


INDICATOR_MAP = {
    "MA": ma,
    "Z-Score": zscore,
    "Momentum": momentum,
    "RSI": rsi,
    "Rolling Std": rolling_std,
    "Avg True Range": atr,
    "EWMA": ewma_vol,
    "Annual Std": annual_std,
}


def calculate_indicator(df: pd.DataFrame, indicator: str, **kwargs):
    try:
        fn = INDICATOR_MAP[indicator]
    except KeyError:
        raise ValueError(f"Unknown indicator: {indicator}")
    return fn(df, **kwargs)
