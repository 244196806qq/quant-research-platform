import pandas as pd
from indicators import calculate_indicator
import numpy as np


def _make_series(index, value=0.0, name="Signal"):
    return pd.Series(value, index=index, name=name, dtype="float64")


def _summarize_signal(signal: pd.Series) -> dict:
    return {
        "min": float(signal.min()),
        "max": float(signal.max()),
        "mean": float(signal.mean()),
        "has_inf": not np.isfinite(signal).all(),
        "has_nan": signal.isna().any(),
    }


def determine_strategy(df_asset: pd.DataFrame, strategy: str, params: dict = None, holdings: dict = None, indicators: dict = None):
    """
    Unified strategy interface: returns (signal, strategy_return, diagnostics)
    """
    try:
        fn = STRATEGY_REGISTRY[strategy]
    except KeyError:
        raise ValueError(f"Unknown strategy: {strategy}")

    return fn(df_asset, params or {}, holdings or {}, indicators)


def ma_strategy(df: pd.DataFrame, params: dict, holdings: dict, indicators: dict = None):
    ma_df = calculate_indicator(df, "MA")
    signal = _make_series(df.index, 1.0)
    short = ma_df.columns[0]
    mid = ma_df.columns[1]
    long = ma_df.columns[2]
    signal.loc[ma_df[short] < ma_df[mid]] = 0.75
    signal.loc[ma_df[short] < ma_df[long]] = 0.5
    signal.loc[ma_df[mid] < ma_df[long]] = 0.25
    strategy_return = signal.shift(1).fillna(0) * df["Market_Return"].fillna(0)
    diag = _summarize_signal(signal)
    diag.update({"trade_count": int((signal.shift(1).fillna(0) != signal.fillna(0)).sum())})
    return signal, strategy_return, diag


def zscore_strategy(df: pd.DataFrame, params: dict, holdings: dict, indicators: dict = None):
    s = calculate_indicator(df, "Z-Score")
    signal = _make_series(df.index, 1.0)
    signal.loc[s > params.get("zscore_high", 2.0)] = 0.0
    signal.loc[s < params.get("zscore_low", -2.0)] = 2.0
    strategy_return = signal.shift(1).fillna(0) * df["Market_Return"].fillna(0)
    diag = _summarize_signal(signal)
    diag.update({"signal_dist": s.describe().to_dict()})
    return signal, strategy_return, diag


def momentum_strategy(df: pd.DataFrame, params: dict, holdings: dict, indicators: dict = None):
    m = calculate_indicator(df, "Momentum")
    signal = _make_series(df.index, 0.2)
    signal.loc[m > 1.0] = 1.0
    strategy_return = signal.shift(1).fillna(0) * df["Market_Return"].fillna(0)
    diag = _summarize_signal(signal)
    return signal, strategy_return, diag


def rsi_strategy(df: pd.DataFrame, params: dict, holdings: dict, indicators: dict = None):
    r = calculate_indicator(df, "RSI")
    signal = _make_series(df.index, 1.0)
    signal.loc[r < params.get("rsi_lower", 30)] = params.get("rsi_overshoot", 2.0)
    signal.loc[r > params.get("rsi_upper", 70)] = params.get("rsi_cooldown", 0.2)
    strategy_return = signal.shift(1).fillna(0) * df["Market_Return"].fillna(0)
    diag = _summarize_signal(signal)
    return signal, strategy_return, diag


def rolling_std_strategy(df: pd.DataFrame, params: dict, holdings: dict, indicators: dict = None):
    vol = calculate_indicator(df, "Rolling Std")
    signal = _make_series(df.index, 1.0)
    signal.loc[vol > params.get("vol_high", 0.03)] = 0.25
    signal.loc[(vol > params.get("vol_mid", 0.02)) & (vol <= params.get("vol_high", 0.03))] = 0.5
    signal.loc[(vol > params.get("vol_low", 0.01)) & (vol <= params.get("vol_mid", 0.02))] = 0.75
    strategy_return = signal.shift(1).fillna(0) * df["Market_Return"].fillna(0)
    diag = _summarize_signal(signal)
    return signal, strategy_return, diag


def atr_strategy(df: pd.DataFrame, params: dict, holdings: dict, indicators: dict = None):
    atr_s = calculate_indicator(df, "Avg True Range")
    signal = _make_series(df.index, 1.0)
    # Use a simple rule vs its rolling mean
    roll_mean = atr_s.rolling(100).mean()
    signal.loc[atr_s > 1.5 * roll_mean] = 0.5
    strategy_return = signal.shift(1).fillna(0) * df["Market_Return"].fillna(0)
    diag = _summarize_signal(signal)
    return signal, strategy_return, diag


def ewma_strategy(df: pd.DataFrame, params: dict, holdings: dict, indicators: dict = None):
    e = calculate_indicator(df, "EWMA")
    # EWMA returns vol series
    signal = _make_series(df.index, 1.0)
    roll_mean = e.rolling(100).mean()
    signal.loc[e > 1.5 * roll_mean] = 0.5
    strategy_return = signal.shift(1).fillna(0) * df["Market_Return"].fillna(0)
    diag = _summarize_signal(signal)
    return signal, strategy_return, diag


def annual_std_strategy(df: pd.DataFrame, params: dict, holdings: dict, indicators: dict = None):
    a = calculate_indicator(df, "Annual Std")
    signal = _make_series(df.index, 1.0)
    signal.loc[a > params.get("annual_vol_threshold", 0.30)] = 0.5
    strategy_return = signal.shift(1).fillna(0) * df["Market_Return"].fillna(0)
    diag = _summarize_signal(signal)
    return signal, strategy_return, diag


def combined_strategy(df: pd.DataFrame, params: dict, holdings: dict, indicators: dict):
    if indicators is None:
        raise ValueError("Combined strategy requires an indicators dictionary.")

    trend_sig, _, _ = determine_strategy(df, indicators["trend"], params=params, holdings=holdings)
    vol_sig, _, _ = determine_strategy(df, indicators["volatility"], params=params, holdings=holdings)
    timing_sig, _, _ = determine_strategy(df, indicators["timing"], params=params, holdings=holdings)

    signal = trend_sig * vol_sig * timing_sig
    strategy_return = signal.shift(1).fillna(0) * df["Market_Return"].fillna(0)
    diag = _summarize_signal(signal)
    return signal, strategy_return, diag

def learning_strategy(df: pd.DataFrame, params: dict, holdings: dict, indicators: dict):
    # Placeholder for a strategy that could use machine learning predictions
    # For now, it just returns a flat signal and zero return
    signal = _make_series(df.index, 1.0)
    strategy_return = _make_series(df.index, 0.0, name="Strategy_Return")
    diag = {"note": "Learning strategy not implemented yet."}
    return signal, strategy_return, diag


STRATEGY_REGISTRY = {
    "MA": ma_strategy,
    "Z-Score": zscore_strategy,
    "Momentum": momentum_strategy,
    "RSI": rsi_strategy,
    "Rolling Std": rolling_std_strategy,
    "Avg True Range": atr_strategy,
    "EWMA": ewma_strategy,
    "Annual Std": annual_std_strategy,
    "Combined": combined_strategy,
    "Learning": learning_strategy,
}


# Metadata for UI and automatic parameter handling
STRATEGY_METADATA = {
    "MA": {
        "name": "MA",
        "description": "Trend-following moving average exposure scaling.",
        "category": "Trend",
        "parameters": {"fast_ma": 10, "mid_ma": 50, "slow_ma": 200},
    },
    "Momentum": {
        "name": "Momentum",
        "description": "Momentum exposure based on price vs moving average.",
        "category": "Trend",
        "parameters": {"window": 50},
    },
    "Z-Score": {
        "name": "Z-Score",
        "description": "Mean-reversion based on price z-score.",
        "category": "Mean Reversion",
        "parameters": {"zscore_low": -2.0, "zscore_high": 2.0},
    },
    "RSI": {
        "name": "RSI",
        "description": "Mean-reversion using RSI thresholds.",
        "category": "Mean Reversion",
        "parameters": {"rsi_lower": 30, "rsi_upper": 70, "rsi_overshoot": 2.0, "rsi_cooldown": 0.2},
    },
    "Rolling Std": {
        "name": "Rolling Std",
        "description": "Volatility-based exposure scaling.",
        "category": "Volatility",
        "parameters": {"vol_low": 0.01, "vol_mid": 0.02, "vol_high": 0.03},
    },
    "Avg True Range": {
        "name": "Avg True Range",
        "description": "ATR-based risk adjustment.",
        "category": "Volatility",
        "parameters": {"window": 14},
    },
    "EWMA": {
        "name": "EWMA",
        "description": "EWMA volatility-based exposure scaling.",
        "category": "Volatility",
        "parameters": {"span": 20},
    },
    "Annual Std": {
        "name": "Annual Std",
        "description": "Annualized volatility thresholding.",
        "category": "Volatility",
        "parameters": {"window": 20, "annual_vol_threshold": 0.30},
    },
    "Combined": {
        "name": "Combined",
        "description": "Combination of trend, volatility and timing factors.",
        "category": "Composite",
        "parameters": {},
        "required_indicators": ["trend", "volatility", "timing"],
    },
}


