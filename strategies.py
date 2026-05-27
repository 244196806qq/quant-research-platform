import pandas as pd
from indicators import calculate_indicator


def _signal(index, default_value):
    signal = pd.Series(default_value, index=index, name="Signal", dtype="float64")
    return signal


def determine_strategy(df_asset, strategy, indicators=None):
    if strategy == "Combined":
        return combined_strategy(df_asset, indicators)

    try:
        return STRATEGIES[strategy](df_asset)
    except KeyError:
        raise ValueError(f"Unknown strategy: {strategy}")


def ma_strategy(df_asset):
    calculate_indicator(df_asset, "MA")
    signal = _signal(df_asset.index, 1.0)
    signal.loc[df_asset["MA10"] < df_asset["MA50"]] = 0.75
    signal.loc[df_asset["MA10"] < df_asset["MA200"]] = 0.50
    signal.loc[df_asset["MA50"] < df_asset["MA200"]] = 0.25
    return signal


def zscore_strategy(df_asset):
    calculate_indicator(df_asset, "Z-Score")
    signal = _signal(df_asset.index, 1.0)
    signal.loc[df_asset["Z-Score"] > 2] = 0.0
    signal.loc[df_asset["Z-Score"] < -2] = 2.0
    return signal


def momentum_strategy(df_asset):
    calculate_indicator(df_asset, "Momentum")
    signal = _signal(df_asset.index, 0.20)
    signal.loc[df_asset["MA50"] < df_asset["Adj Close"]] = 1.0
    return signal


def rsi_strategy(df_asset):
    calculate_indicator(df_asset, "RSI")
    signal = _signal(df_asset.index, 1.0)
    signal.loc[df_asset["RSI"] < 30] = 2.0
    signal.loc[df_asset["RSI"] > 70] = 0.2
    return signal


def rolling_std_strategy(df_asset):
    calculate_indicator(df_asset, "Rolling Std")
    signal = _signal(df_asset.index, 1.0)
    signal.loc[df_asset["Volatility"] > 0.03] = 0.25
    signal.loc[(df_asset["Volatility"] > 0.02) & (df_asset["Volatility"] <= 0.03)] = 0.50
    signal.loc[(df_asset["Volatility"] > 0.01) & (df_asset["Volatility"] <= 0.02)] = 0.75
    return signal


def atr_strategy(df_asset):
    calculate_indicator(df_asset, "Avg True Range")
    signal = _signal(df_asset.index, 1.0)
    signal.loc[df_asset["ATR"] > 1.5 * df_asset["ATR Mean"]] = 0.5
    return signal


def ewma_strategy(df_asset):
    calculate_indicator(df_asset, "EWMA")
    signal = _signal(df_asset.index, 1.0)
    signal.loc[df_asset["EWMA_Volatility"] > 1.5 * df_asset["EWMA_Mean"]] = 0.5
    return signal


def annual_std_strategy(df_asset):
    calculate_indicator(df_asset, "Annual Std")
    signal = _signal(df_asset.index, 1.0)
    signal.loc[df_asset["Annual Volatility"] > 0.30] = 0.5
    return signal


STRATEGIES = {
    "MA": ma_strategy,
    "Z-Score": zscore_strategy,
    "Momentum": momentum_strategy,
    "RSI": rsi_strategy,
    "Rolling Std": rolling_std_strategy,
    "Avg True Range": atr_strategy,
    "EWMA": ewma_strategy,
    "Annual Std": annual_std_strategy,
}


def calculate_factor(df_asset, indicator):
    try:
        return STRATEGIES[indicator](df_asset)
    except KeyError:
        raise ValueError(f"Unknown combined strategy indicator: {indicator}")


def combined_strategy(df_asset, indicators):
    if indicators is None:
        raise ValueError("Combined strategy requires an indicators dictionary.")

    trend_factor = calculate_factor(df_asset, indicators["trend"])
    volatility_factor = calculate_factor(df_asset, indicators["volatility"])
    timing_factor = calculate_factor(df_asset, indicators["timing"])

    signal = trend_factor * volatility_factor * timing_factor
    signal.name = "Signal"
    
    return signal


