from strategies import determine_strategy
import pandas as pd
import numpy as np


DEPOSIT_FREQUENCY_DAYS = 21


def clean_signal(series: pd.Series, allow_leverage: bool = False) -> pd.Series:
    series = series.replace([np.inf, -np.inf], 0).fillna(0)
    if not allow_leverage:
        series = series.clip(lower=0.0, upper=1.0)
    return series


def clean_returns(series: pd.Series) -> pd.Series:
    return series.replace([np.inf, -np.inf], 0).fillna(0)


def calculate_cagr(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    start = equity.iloc[0]
    end = equity.iloc[-1]
    if start <= 0:
        return 0.0
    days = (equity.index[-1] - equity.index[0]).days
    years = days / 365.25 if days > 0 else 1.0
    return (end / start) ** (1 / years) - 1 if years > 0 else 0.0


def calculate_max_drawdown(equity: pd.Series) -> float:
    rolling_max = equity.cummax()
    drawdown = (equity - rolling_max) / rolling_max
    return float(drawdown.min()) if not drawdown.empty else 0.0


def calculate_annual_volatility(returns: pd.Series, periods: int = 252) -> float:
    returns = returns.replace([np.inf, -np.inf], np.nan).dropna()
    if returns.empty:
        return 0.0
    return float(returns.std() * np.sqrt(periods))


def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0, periods: int = 252) -> float:
    returns = returns.replace([np.inf, -np.inf], np.nan).dropna()
    if returns.empty:
        return 0.0
    excess_returns = returns - risk_free_rate / periods
    volatility = returns.std()
    return float(excess_returns.mean() * np.sqrt(periods) / volatility) if volatility != 0 else 0.0


def count_deposits(num_days, deposit_frequency=DEPOSIT_FREQUENCY_DAYS):
    return (num_days - 1) // deposit_frequency


def calculate_equity_curve(daily_returns, initial_money, deposit, deposit_frequency=DEPOSIT_FREQUENCY_DAYS):
    equity = []
    current_value = initial_money

    for day_number, daily_return in enumerate(daily_returns.fillna(0)):
        current_value *= (1 + daily_return)
        if day_number % deposit_frequency == 0 and day_number != 0:
            current_value += deposit
        equity.append(current_value)

    return pd.Series(equity, index=daily_returns.index, name="Market_Equity")


# ------------------- Position sizing utilities -------------------
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
    rv = realized_volatility(returns, lookback=lookback)
    # avoid divide by zero
    scale = vol_target / rv
    scale = scale.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    # cap scaling to max leverage
    scale = scale.clip(upper=max_leverage)
    exposure = signal * scale
    return exposure, rv


def compute_portfolio_weights(dfs: list, active: list, mode: str = "manual", lookback: int = 63) -> dict:
    """Return a dict of weights for each active ticker based on mode.
    mode: 'manual'|'equal'|'inverse_vol'|'signal_strength'
    """
    if mode == "manual":
        return None
    n = len(active)
    if n == 0:
        return {t: 0.0 for t in active}
    if mode == "equal":
        w = 1.0 / n
        return {t: w for t in active}
    if mode == "inverse_vol":
        vols = []
        for df in dfs:
            rv = realized_volatility(df["Market_Return"], lookback=lookback)
            vols.append(rv.fillna(method="ffill").iloc[-1] if not rv.dropna().empty else np.nan)
        inv = np.array([1.0 / v if v and not np.isnan(v) else 0.0 for v in vols], dtype=float)
        s = inv.sum()
        if s == 0:
            return {t: 0.0 for t in active}
        weights = {t: float(inv[i] / s) for i, t in enumerate(active)}
        return weights
    # signal_strength mode requires signals; handled elsewhere
    return {t: 1.0 / n for t in active}

# ------------------- end sizing utilities -------------------


def calculate_daily_equity(dfs, initial_money, deposit):
    for i in range(len(dfs)):
        dfs[i]["Market_Equity"] = calculate_equity_curve(dfs[i]["Market_Return"], initial_money, deposit)


def calculate_daily_return(dfs):
    for i in range(len(dfs)):
        raw_returns = dfs[i]["Adj Close"].pct_change()
        dfs[i]["Market_Return"] = clean_returns(raw_returns)


def rolling_sharpe(returns: pd.Series, window: int = 63, periods: int = 252) -> pd.Series:
    rolling_mean = returns.rolling(window).mean()
    rolling_std = returns.rolling(window).std()
    return (rolling_mean * np.sqrt(periods) / rolling_std).replace([np.inf, -np.inf], np.nan)


def rolling_volatility(returns: pd.Series, window: int = 63, periods: int = 252) -> pd.Series:
    return returns.rolling(window).std() * np.sqrt(periods)


def rolling_drawdown(equity: pd.Series, window: int = 63) -> pd.Series:
    rolling_max = equity.rolling(window, min_periods=1).max()
    return equity / rolling_max - 1


def simulate_strategy(dfs, holding, symbols, strategy, initial_money, deposit, indicators=None, params=None, diagnostics=None, allow_leverage: bool = False, fee: float = 0.0, slippage: float = 0.0):
    """
    Simulate a single strategy across multiple assets (columns of dfs).
    - `holding` is a dict mapping ticker -> portfolio weight (sums to ~1)
    - `fee` and `slippage` are fractional costs applied to turnover
    Returns a pandas Series (equity) for the combined portfolio using the strategy signals.
    """
    df_strategy = pd.DataFrame(index=dfs[0].index)
    df_strategy["Market_Return"] = 0.0

    # per-strategy diagnostics container
    diagnostics = diagnostics or {}
    diagnostics.setdefault(strategy, [])

    # load sizing and weighting params
    sizing_mode = (params or {}).get("sizing_mode", "fixed")
    vol_target = (params or {}).get("vol_target", 0.15)
    sizing_lookback = (params or {}).get("sizing_lookback", 63)
    max_leverage = (params or {}).get("max_leverage", 1.5)
    min_exposure = (params or {}).get("min_exposure", 0.0)
    max_exposure = (params or {}).get("max_exposure", 1.5)
    weight_mode = (params or {}).get("weight_mode", "manual")
    rebalance = (params or {}).get("rebalance", "daily")
    rebalance_map = {"daily": 1, "weekly": 5, "monthly": 21}
    rebalance_step = rebalance_map.get(rebalance, 1)

    # Precompute raw signals and returns for portfolio-level sizing modes
    raw_signals = []
    asset_returns = []
    per_ticker_diags = []
    for i in range(len(dfs)):
        df = dfs[i]
        sig, strat_ret, diag = determine_strategy(df, strategy, params=params or {}, holdings=holding, indicators=indicators)
        raw_signals.append(sig)
        asset_returns.append(df["Market_Return"].fillna(0))
        per_ticker_diags.append((symbols[i], diag))

    # compute portfolio weights if non-manual
    pw = compute_portfolio_weights(dfs, symbols, mode=weight_mode, lookback=sizing_lookback)
    if pw is None:
        # use provided holding dict
        pw = {s: float(holding.get(s, 0.0)) for s in symbols}
    elif weight_mode == "signal_strength":
        # weight proportional to mean absolute signal
        vals = [abs(sig).mean() for sig in raw_signals]
        s = sum(vals)
        pw = {symbols[i]: float(vals[i] / s) if s > 0 else 0.0 for i in range(len(symbols))}

    # prepare accumulator for total exposure and realized vol tracking
    total_abs_exposure = pd.Series(0.0, index=dfs[0].index)

    for i in range(len(dfs)):
        sym = symbols[i]
        df = dfs[i]
        sig = raw_signals[i]
        returns_series = asset_returns[i]

        # record raw diag
        entry = {
            "ticker": sym,
            "signal_raw_min": float(sig.min()),
            "signal_raw_max": float(sig.max()),
            "signal_raw_has_inf": not np.isfinite(sig).all(),
            "signal_raw_has_nan": sig.isna().any(),
        }

        # Clean signal
        clean_sig = clean_signal(sig, allow_leverage=allow_leverage)
        entry.update({
            "signal_clean_min": float(clean_sig.min()),
            "signal_clean_max": float(clean_sig.max()),
            "signal_clean_clipped": ((sig < 0) | (sig > 1)).any(),
        })

        # sizing model
        if sizing_mode == "fixed":
            exposure = fixed_position_size(clean_sig, (params or {}).get("fixed_exposure", 1.0))
            rv = realized_volatility(returns_series, lookback=sizing_lookback)
        elif sizing_mode == "vol_target":
            exposure, rv = volatility_target_size(clean_sig, returns_series, vol_target=vol_target, lookback=sizing_lookback, max_leverage=max_leverage)
        elif sizing_mode == "signal_scaled":
            exposure = signal_scaled_size(clean_sig, (params or {}).get("scale", 1.0))
            rv = realized_volatility(returns_series, lookback=sizing_lookback)
        else:
            exposure = clean_sig.copy()
            rv = realized_volatility(returns_series, lookback=sizing_lookback)

        # apply exposure caps
        exposure = capped_leverage_size(exposure, min_exp=min_exposure, max_exp=max_exposure)

        # rebalance: only update exposures on rebalance days
        if rebalance_step > 1:
            mask = (np.arange(len(exposure)) % rebalance_step) == 0
            exposure = exposure.where(mask).ffill()

        # store exposure series for diagnostics
        diagnostics.setdefault(strategy + "_exposure", {})[sym] = exposure.copy()
        diagnostics.setdefault(strategy + "_realized_vol", {})[sym] = rv

        # weight by portfolio allocation for this symbol
        alloc = float(pw.get(sym, 0.0))
        pos_weight = exposure * alloc

        # accumulate total absolute exposure for portfolio diagnostics
        total_abs_exposure += pos_weight.abs()

        # compute turnover and trade counts
        delta = pos_weight.diff().abs().fillna(pos_weight.abs())
        trade_count = int((delta > 0).sum())
        turnover = float(delta.sum())

        # average holding duration (simple run-length on pos>0)
        holds = (pos_weight > 0).astype(int)
        durations = []
        current = 0
        for v in holds:
            if v:
                current += 1
            elif current:
                durations.append(current)
                current = 0
        if current:
            durations.append(current)
        avg_hold = float(sum(durations) / len(durations)) if durations else 0.0

        # transaction costs applied when position changes (approx as fraction of portfolio)
        cost_series = delta * (fee + slippage)

        # apply strategy returns scaled by allocated position weight (use shift for execution)
        applied_returns = pos_weight.shift(1).fillna(0) * df["Market_Return"].fillna(0)
        # subtract costs from returns (costs are applied as absolute fractional drags)
        net_returns = applied_returns - cost_series

        # aggregate into portfolio return
        df_strategy["Market_Return"] += net_returns

        # record per-ticker diagnostics
        entry.update({
            "return_min": float(net_returns.min()),
            "return_max": float(net_returns.max()),
            "return_has_inf": not np.isfinite(net_returns).all(),
            "return_has_nan": net_returns.isna().any(),
            "trade_count": trade_count,
            "turnover": turnover,
            "avg_hold_days": avg_hold,
            "exposure_mean": float(exposure.mean()),
        })

        diagnostics[strategy].append(entry)

    # attach portfolio-level exposure diagnostics
    diagnostics[strategy + "_total_exposure"] = total_abs_exposure

    # final equity
    df_strategy["Market_Equity"] = calculate_equity_curve(df_strategy["Market_Return"], initial_money, deposit)

    # attach some rolling diagnostics summary for the strategy
    try:
        returns = df_strategy["Market_Return"].replace([np.inf, -np.inf], np.nan).dropna()
        diagnostics.setdefault(strategy + "_rolling", {})
        diagnostics[strategy + "_rolling"] = {
            "rolling_sharpe": rolling_sharpe(returns),
            "rolling_vol": rolling_volatility(returns),
            "rolling_dd": rolling_drawdown(df_strategy["Market_Equity"]),
        }
    except Exception:
        pass

    return df_strategy["Market_Equity"]


def build_benchmarks(dfs, active, initial_money=10000, deposit=0):
    """Return a dict of benchmark equity series: equal_weight, buy_hold per asset, and SPY if available."""
    buy_hold = pd.DataFrame({ticker: dfs[i]["Market_Equity"] for i, ticker in enumerate(active)})
    # equal weight returns
    equal_returns = buy_hold.pct_change().mean(axis=1).fillna(0)
    equal_equity = calculate_equity_curve(equal_returns, initial_money, deposit)
    benchmarks = {"equal_weight": equal_equity, "buy_hold": buy_hold}
    # SPY if present
    if "SPY" in active:
        idx = active.index("SPY")
        benchmarks["SPY"] = dfs[idx]["Market_Equity"]
    return benchmarks
