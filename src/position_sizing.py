import pandas as pd
import numpy as np


def fixed_position_size(signal: pd.Series, fixed_exposure: float) -> pd.Series:
    return signal * fixed_exposure


def signal_scaled_size(signal: pd.Series, scale: float) -> pd.Series:
    return signal * scale


def capped_leverage_size(exposure: pd.Series, min_exp: float = 0.0, max_exp: float = 1.0) -> pd.Series:
    return exposure.clip(lower=min_exp, upper=max_exp)


def realized_volatility(returns: pd.Series, lookback: int = 63, periods: int = 252) -> pd.Series:
    # annualized rolling volatility
    return returns.replace([np.inf, -np.inf], np.nan).rolling(lookback).std() * np.sqrt(periods)


def volatility_target_size(signal: pd.Series, returns: pd.Series, vol_target: float = 0.15, lookback: int = 63, max_leverage: float = 1.5) -> (pd.Series, pd.Series):
    rv = realized_volatility(returns, lookback=lookback).shift(1)  # use yesterday's volatility to avoid lookahead bias
    # avoid divide by zero
    scale = vol_target / rv
    scale = scale.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    # cap scaling to max leverage
    scale = scale.clip(upper=max_leverage)
    exposure = signal * scale
    return exposure, rv

def drawdown_based_size(equity: pd.Series, first_drawdown_target: float = -0.2, second_drawdown_target: float = -0.3, third_drawdown_target: float = -0.4, lookback: int = 63) -> pd.Series:
    rolling_peak = equity.rolling(lookback).max()
    drawdown = (equity - rolling_peak) / rolling_peak
    exposure = pd.Series(1.0, index = equity.index)
    exposure.loc[drawdown < first_drawdown_target] = .75
    exposure.loc[drawdown < second_drawdown_target] = 0.5
    exposure.loc[drawdown < third_drawdown_target] = 0.25
    return exposure

def combined_position_size(*exposure, min_exposure: float = 0.0, max_exposure: float = 1.0) -> pd.Series:
    final = exposure[0].copy()

    for exp in exposure[1:]:
        final *= exp
    
    return final.clip(lower=min_exposure, upper=max_exposure)
