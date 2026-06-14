"""Streamlit app entrypoint for the quant backtesting starter project.

This app uses the existing project architecture and wires the current
backtest and strategy logic to an interactive UI.
"""
from pathlib import Path
import datetime

import pandas as pd
import streamlit as st
import numpy as np

from backtest import (
    calculate_annual_volatility,
    calculate_cagr,
    calculate_daily_equity,
    calculate_daily_return,
    calculate_max_drawdown,
    calculate_sharpe_ratio,
    count_deposits,
    simulate_strategy,
)
from data_loader import csv_reader
from data_manager import ensure_ticker_data, ticker_exists
from plotting import (
    buy_and_hold_figure,
    clean_plot_df,
    drawdown_figure,
    equity_curve_figure,
    rolling_volatility_figure,
    signal_figure,
)
import strategies as strategies_mod
import plotly.express as px
import plotly.graph_objects as go


# strategy options are loaded dynamically from the registry
STRATEGY_OPTIONS = list(strategies_mod.STRATEGY_REGISTRY.keys())


st.set_page_config(page_title="Quant Backtester — Streamlit", layout="wide")


def init_state():
    st.session_state.setdefault("tickers", [])
    st.session_state.setdefault("active", [])
    st.session_state.setdefault("holdings", {})
    st.session_state.setdefault("strategies", [])
    # set sensible defaults for combined strategy indicators from metadata
    meta = getattr(strategies_mod, "STRATEGY_METADATA", {})
    def _first_by_category(cat):
        for k, v in meta.items():
            if v.get("category") == cat:
                return k
        # fallback
        return next(iter(strategies_mod.STRATEGY_REGISTRY.keys()))

    st.session_state.setdefault("combined_indicators", {
        "trend": _first_by_category("Trend"),
        "volatility": _first_by_category("Volatility"),
        "timing": _first_by_category("Mean Reversion"),
    })
    st.session_state.setdefault("strategy_params", {})
    st.session_state.setdefault("params", {
        "ma_window": 20,
        "rsi_lower": 30,
        "rsi_upper": 70,
        "zscore_low": -2.0,
        "zscore_high": 2.0,
        "volatility_threshold": 0.02,
        "atr_multiplier": 1.5,
        "ewma_span": 20,
        "initial_money": 10000,
        "deposit_amount": 100,
    })


def _normalize_ticker(ticker: str) -> str:
    return ticker.strip().upper()


def add_ticker_to_state(ticker: str):
    t = _normalize_ticker(ticker)
    if not t or t in st.session_state["tickers"]:
        return t

    st.session_state["tickers"].append(t)
    st.session_state["active"].append(t)

    if t not in st.session_state["holdings"]:
        st.session_state["holdings"][t] = 0.0
    if len(st.session_state["tickers"]) == 1:
        st.session_state["holdings"][t] = 1.0
    return t


def remove_selected(selected):
    for t in selected:
        if t in st.session_state["tickers"]:
            st.session_state["tickers"].remove(t)
        if t in st.session_state["active"]:
            st.session_state["active"].remove(t)
        st.session_state["holdings"].pop(t, None)


def sidebar_ui():
    st.sidebar.title("Controls")

    # --- Stock Controls ---
    with st.sidebar.expander("Stock Controls", expanded=True):
        st.subheader("Add / Remove Tickers")
        c1, c2 = st.columns([3, 1])
        ticker_in = c1.text_input("Ticker (e.g. AAPL)", key="ticker_input")
        if c2.button("Add", key="add_ticker_btn"):
            ticker = _normalize_ticker(ticker_in)
            if not ticker:
                st.warning("Enter a ticker symbol to add.")
            elif ticker in st.session_state["tickers"]:
                st.info(f"{ticker} is already in the list.")
            else:
                local = ticker_exists(ticker)
                try:
                    ensure_ticker_data(ticker)
                    add_ticker_to_state(ticker)
                    if local:
                        st.success(f"Loaded {ticker} from local data.")
                    else:
                        st.success(f"Downloaded and added {ticker}.")
                except Exception as exc:
                    st.error(f"Failed to add {ticker}: {exc}")

        # remove selected row
        r1, r2 = st.columns([3, 1])
        remove_choice = r1.multiselect("Remove selected", options=st.session_state["tickers"], key="remove_choice")
        if r2.button("Remove", key="remove_btn"):
            remove_selected(remove_choice)

        st.markdown("---")
        st.write(f"Tickers: {len(st.session_state['tickers'])} — Active: {len(st.session_state['active'])}")
        local_files = sorted([p.stem for p in Path("data").glob("*.csv")])
        if local_files:
            st.selectbox("Local CSVs", options=local_files, key="local_csvs_box")
        else:
            st.caption("No local CSV files found in /data")

    # --- Portfolio ---
    with st.sidebar.expander("Portfolio", expanded=False):
        st.subheader("Holdings & Weights")
        active = st.session_state.get("active", [])
        if not active:
            st.info("No active tickers. Add tickers in Stock Controls.")
        else:
            cols = st.columns(2)
            for i, ticker in enumerate(active):
                col = cols[i % 2]
                weight = col.number_input(f"{ticker} weight", min_value=0.0, max_value=1.0, value=float(st.session_state["holdings"].get(ticker, 0.0)), step=0.01, key=f"weight_{ticker}")
                st.session_state["holdings"][ticker] = weight

            total_weight = sum(st.session_state["holdings"].get(t, 0.0) for t in active)
            st.metric("Total weight", f"{total_weight:.2f}")
            if not 0.99 <= total_weight <= 1.01:
                st.warning("Weights should total close to 1.00 (100%).")

            if st.button("Normalize weights"):
                if total_weight > 0:
                    for t in active:
                        st.session_state["holdings"][t] = st.session_state["holdings"][t] / total_weight

    # --- Strategy Selection ---
    with st.sidebar.expander("Strategy Selection", expanded=True):
        st.subheader("Choose strategies")
        metadata = getattr(strategies_mod, "STRATEGY_METADATA", {})
        categories = {}
        for name, meta in metadata.items():
            categories.setdefault(meta.get("category", "Other"), []).append(name)

        chosen = set(st.session_state.get("strategies", []))
        for cat, names in categories.items():
            with st.expander(cat, expanded=False):
                cols = st.columns(2)
                for i, name in enumerate(sorted(names)):
                    checked = cols[i % 2].checkbox(name, value=(name in chosen), key=f"chk_{name}")
                    if checked:
                        chosen.add(name)
                    elif name in chosen:
                        chosen.discard(name)

        st.session_state["strategies"] = list(chosen)

        # Combined indicator picker (if combined selected)
        if "Combined" in st.session_state.get("strategies", []):
            st.markdown("---")
            st.caption("Combined strategy components")
            # find available strategy names by category
            def names_by_cat(cat):
                return [k for k, v in metadata.items() if v.get("category") == cat]

            trend_choices = names_by_cat("Trend") or list(strategies_mod.STRATEGY_REGISTRY.keys())
            vol_choices = names_by_cat("Volatility") or trend_choices
            timing_choices = names_by_cat("Mean Reversion") or trend_choices

            trend = st.selectbox("Trend component", options=trend_choices, index=trend_choices.index(st.session_state["combined_indicators"]["trend"]))
            vol = st.selectbox("Volatility component", options=vol_choices, index=vol_choices.index(st.session_state["combined_indicators"]["volatility"]))
            timing = st.selectbox("Timing component", options=timing_choices, index=timing_choices.index(st.session_state["combined_indicators"]["timing"]))
            st.session_state["combined_indicators"] = {"trend": trend, "volatility": vol, "timing": timing}

    # --- Indicator Parameters ---
    with st.sidebar.expander("Indicator Parameters", expanded=False):
        st.subheader("Common indicator params")
        p1, p2 = st.columns(2)
        params = st.session_state.setdefault("params", {})
        params["ma_window"] = p1.slider("MA window", 5, 200, int(params.get("ma_window", 20)))
        params["ewma_span"] = p1.number_input("EWMA span", min_value=1, value=int(params.get("ewma_span", 20)), step=1)
        params["rsi_lower"] = p2.slider("RSI lower", 0, 50, int(params.get("rsi_lower", 30)))
        params["rsi_upper"] = p2.slider("RSI upper", 50, 100, int(params.get("rsi_upper", 70)))
        params["zscore_low"] = p2.number_input("Z low", value=float(params.get("zscore_low", -2.0)))
        params["zscore_high"] = p2.number_input("Z high", value=float(params.get("zscore_high", 2.0)))
        st.session_state["params"] = params

        # Per-strategy parameter expanders
        st.markdown("---")
        st.subheader("Strategy parameters")
        st.session_state.setdefault("strategy_params", {})
        for strat in st.session_state.get("strategies", []):
            meta = metadata.get(strat, {})
            sparams = meta.get("parameters", {})
            if not sparams:
                continue
            with st.expander(f"{strat} parameters", expanded=False):
                cols = st.columns(2)
                st.session_state["strategy_params"].setdefault(strat, {})
                for i, (pname, pval) in enumerate(sparams.items()):
                    col = cols[i % 2]
                    key = f"param_{strat}_{pname}"
                    if isinstance(pval, int):
                        nv = col.number_input(pname, value=int(st.session_state["strategy_params"][strat].get(pname, pval)), step=1, key=key)
                    else:
                        nv = col.number_input(pname, value=float(st.session_state["strategy_params"][strat].get(pname, pval)), step=0.01, key=key)
                    st.session_state["strategy_params"][strat][pname] = nv

    # --- Position Sizing ---
    with st.sidebar.expander("Position Sizing", expanded=False):
        st.subheader("Sizing & Weights")
        params = st.session_state.setdefault("params", {})
        params["weight_mode"] = st.selectbox("Weighting mode", options=["manual", "equal", "inverse_vol", "signal_strength"], index=["manual", "equal", "inverse_vol", "signal_strength"].index(params.get("weight_mode", "manual")))
        params["sizing_mode"] = st.selectbox("Sizing mode", options=["fixed", "vol_target", "signal_scaled", "none"], index=["fixed", "vol_target", "signal_scaled", "none"].index(params.get("sizing_mode", "fixed")))
        if params["sizing_mode"] == "vol_target":
            params["vol_target"] = st.number_input("Target vol (annual)", min_value=0.0, max_value=5.0, value=float(params.get("vol_target", 0.15)), step=0.01)
            params["sizing_lookback"] = st.number_input("Sizing lookback (days)", min_value=1, value=int(params.get("sizing_lookback", 63)), step=1)
        st.session_state["params"] = params

    # --- Backtest Settings ---
    with st.sidebar.expander("Backtest Settings", expanded=False):
        st.subheader("Run settings")
        st.session_state.setdefault("allow_leverage", False)
        allow_lev = st.checkbox("Allow leverage", value=st.session_state["allow_leverage"]) 
        st.session_state["allow_leverage"] = allow_lev
        st.number_input("Initial money", min_value=0.0, value=float(st.session_state["params"]["initial_money"]), step=100.0, key="ui_initial_money")
        st.number_input("Deposit amount", min_value=0.0, value=float(st.session_state["params"]["deposit_amount"]), step=10.0, key="ui_deposit_amount")

        min_date = datetime.date(1900, 1, 1)
        max_date = datetime.date.today()
        st.date_input("Start date", key="start_date", value=st.session_state.get("start_date", min_date), min_value=min_date, max_value=max_date,)
        st.date_input("End date", key="end_date", value=st.session_state.get("end_date", max_date), min_value=min_date, max_value=max_date,)

        with st.expander("Advanced settings", expanded=False):
            params = st.session_state.setdefault("params", {})
            params["max_leverage"] = st.number_input("Max leverage", min_value=0.0, value=float(params.get("max_leverage", 1.5)), step=0.1)
            params["min_exposure"] = st.number_input("Min exposure", min_value=0.0, value=float(params.get("min_exposure", 0.0)), step=0.01)
            params["max_exposure"] = st.number_input("Max exposure", min_value=0.0, value=float(params.get("max_exposure", 1.5)), step=0.01)
            params["rebalance"] = st.selectbox("Rebalance frequency", options=["daily", "weekly", "monthly"], index=["daily", "weekly", "monthly"].index(params.get("rebalance", "daily")))
            params["fee"] = st.number_input("Trading fee (fraction)", min_value=0.0, value=float(params.get("fee", 0.0)), step=0.0001, format="%.6f")
            params["slippage"] = st.number_input("Slippage (fraction)", min_value=0.0, value=float(params.get("slippage", 0.0)), step=0.0001, format="%.6f")
            st.session_state["params"] = params

    # --- Diagnostics / Export ---
    with st.sidebar.expander("Diagnostics / Export", expanded=False):
        st.subheader("Export")
        diag = st.session_state.get("last_diagnostics")
        if diag:
            st.download_button("Download diagnostics (JSON)", data=pd.Series(diag).to_json(), file_name="diagnostics.json")
        else:
            st.caption("Run a backtest to enable export")

    st.sidebar.markdown("---")
    return st.sidebar.button("Run / Update")


def build_metrics(df_strategy: pd.DataFrame, dfs: dict, active: list, final_money: float) -> pd.DataFrame:
    rows = []
    for i in range(len(active)):
        equity = dfs[i]["Market_Equity"].dropna()
        returns = equity.pct_change().dropna()
        rows.append(
            {
                "Strategy": "Buy & Hold " + active[i],
                "Final Equity": round(float(equity.iloc[-1]) if not equity.empty else 0.0, 2),
                "Return (%)": round(float((equity.iloc[-1] - final_money) / final_money * 100) if (not equity.empty and final_money > 0) else 0.0, 2),
                "CAGR (%)": round(calculate_cagr(equity) * 100, 2),
                "Max Drawdown (%)": round(calculate_max_drawdown(equity) * 100, 2),
                "Volatility (%)": round(calculate_annual_volatility(returns) * 100, 2),
                "Sharpe Ratio": round(calculate_sharpe_ratio(returns), 2),
                "Number of Trades": 0,
            }
        )
    for strategy in df_strategy.columns:
        equity = df_strategy[strategy].dropna()
        returns = equity.pct_change().dropna()
        rows.append(
            {
                "Strategy": strategy,
                "Final Equity": round(float(equity.iloc[-1]) if not equity.empty else 0.0, 2),
                "Return (%)": round(float((equity.iloc[-1] - final_money) / final_money * 100) if (not equity.empty and final_money > 0) else 0.0, 2),
                "CAGR (%)": round(calculate_cagr(equity) * 100, 2),
                "Max Drawdown (%)": round(calculate_max_drawdown(equity) * 100, 2),
                "Volatility (%)": round(calculate_annual_volatility(returns) * 100, 2),
                "Sharpe Ratio": round(calculate_sharpe_ratio(returns), 2),
                "Number of Trades": 0,
            }
        )
    return pd.DataFrame(rows)


def summarize_series(series: pd.Series, label: str) -> dict:
    finite_series = series.replace([np.inf, -np.inf], pd.NA)
    return {
        "label": label,
        "min": float(finite_series.min()) if not finite_series.empty else 0.0,
        "max": float(finite_series.max()) if not finite_series.empty else 0.0,
        "has_inf": not np.isfinite(series).all(),
        "has_nan": series.isna().any(),
    }


def main():
    init_state()
    st.title("Quant Backtester — Streamlit")
    st.markdown("A lightweight interactive backtesting UI. Select tickers and strategies in the sidebar, then click **Run / Update**.")

    run = sidebar_ui()

    tabs = st.tabs(["Overview", "Strategy Results", "Buy & Hold", "Metrics", "Data Preview", "Diagnostics"])

    # Overview tab
    with tabs[0]:
        st.header("Overview")
        st.write("Active tickers:" , st.session_state["active"]) 
        st.write("Selected strategies:", st.session_state["strategies"]) 
        st.write("Parameters:", st.session_state["params"]) 

    # Run/update workflow
    if run:
        # Basic validations
        active = st.session_state["active"]
        strategies = st.session_state["strategies"]
        if not active:
            st.warning("Please add at least one active ticker before running.")
        elif not strategies:
            st.warning("Please select at least one strategy before running.")
        else:
            total_weight = sum(st.session_state["holdings"].get(t, 0.0) for t in active)
            if not 0.99 <= total_weight <= 1.01:
                st.warning("Portfolio weights do not sum to ~1.0. Continue anyway if intentional.")

            # Use UI inputs if provided
            initial_money = st.session_state.get("ui_initial_money", st.session_state["params"]["initial_money"]) if st.session_state.get("ui_initial_money") is not None else st.session_state["params"]["initial_money"]
            deposit_amount = st.session_state.get("ui_deposit_amount", st.session_state["params"]["deposit_amount"]) if st.session_state.get("ui_deposit_amount") is not None else st.session_state["params"]["deposit_amount"]

            strategy_diagnostics = {}
            dfs = None
            df_strategy = None
            buy_hold = None

            try:
                with st.spinner("Loading data and running backtest..."):
                    # ensure data exists locally (download if needed)
                    for ticker in active:
                        local_path = Path("data") / f"{ticker}.csv"
                        if local_path.exists():
                            # load later via csv_reader
                            continue
                        try:
                            ensure_ticker_data(ticker)
                            st.success(f"Downloaded and saved data for {ticker}.")
                        except Exception as exc:
                            st.error(f"Failed to download {ticker}: {exc}")
                            raise RuntimeError(f"Missing data for {ticker}") from exc

                    dfs = csv_reader(active, start_date=st.session_state.get("start_date"), end_date=st.session_state.get("end_date"))
                    calculate_daily_return(dfs)
                    calculate_daily_equity(dfs, initial_money, deposit_amount)

                    # basic return diagnostics
                    for ticker, df in zip(active, dfs):
                        stats = summarize_series(df["Market_Return"], f"{ticker} Market_Return")
                        if stats["has_inf"] or stats["has_nan"]:
                            st.warning(f"{ticker}: Market_Return contained inf/NaN and was cleaned.")

                    final_money = initial_money + deposit_amount * count_deposits(len(dfs[0]))

                    df_strategy = pd.DataFrame(index=dfs[0].index)
                    for strategy in strategies:
                        df_strategy[strategy] = simulate_strategy(
                            dfs,
                            st.session_state["holdings"],
                            active,
                            strategy,
                            initial_money,
                            deposit_amount,
                            st.session_state.get("combined_indicators") if strategy == "Combined" else None,
                            params=st.session_state.get("params", {}),
                            diagnostics=strategy_diagnostics,
                            allow_leverage=st.session_state.get("allow_leverage", False),
                        )

                    buy_hold = pd.DataFrame({ticker: dfs[i]["Market_Equity"] for i, ticker in enumerate(active)})

                st.success("Backtest completed successfully.")

            except Exception as exc:
                st.error("Backtest failed: " + str(exc))
                return

            # Prepare cleaned plotting data
            safe_df_strategy = clean_plot_df(df_strategy)
            safe_buy_hold = clean_plot_df(buy_hold)

            # Strategy Results tab
            with tabs[1]:
                st.header("Strategy Results")
                if safe_df_strategy.empty and safe_buy_hold.empty:
                    st.warning("No valid equity data to plot.")
                else:
                    st.plotly_chart(equity_curve_figure(safe_df_strategy, safe_buy_hold), use_container_width=True)
                    if not safe_df_strategy.empty:
                        st.plotly_chart(drawdown_figure(safe_df_strategy), use_container_width=True)

            # Buy & Hold tab
            with tabs[2]:
                st.header("Buy & Hold")
                if safe_buy_hold.empty:
                    st.warning("No buy-and-hold data available.")
                else:
                    st.plotly_chart(buy_and_hold_figure(safe_buy_hold), use_container_width=True)

            # Metrics tab
            with tabs[3]:
                st.header("Metrics")
                metrics_df = build_metrics(safe_df_strategy, dfs, active, final_money)
                # format numbers for display
                display = metrics_df.copy()
                display["Final Equity"] = display["Final Equity"].map(lambda x: f"${x:,.2f}")
                display[["Return (%)", "CAGR (%)", "Max Drawdown (%)", "Volatility (%)"] ] = display[["Return (%)", "CAGR (%)", "Max Drawdown (%)", "Volatility (%)"] ].round(2).astype(str).applymap(lambda v: v + "%")
                sort_by = st.selectbox("Sort metrics by", options=list(metrics_df.columns[1:]), index=1)
                try:
                    sorted_df = metrics_df.sort_values(by=sort_by, ascending=False)
                except Exception:
                    sorted_df = metrics_df
                st.dataframe(sorted_df, use_container_width=True)

            # Data Preview tab
            with tabs[4]:
                st.header("Data Preview")
                for ticker, df in zip(active, dfs):
                    st.subheader(ticker)
                    st.dataframe(df.head())

            # Diagnostics tab
            with tabs[5]:
                st.header("Diagnostics")
                st.subheader("Strategy diagnostics")
                if not strategy_diagnostics:
                    st.info("No diagnostics available. Run a backtest to populate diagnostics.")
                # iterate over registered strategies present in diagnostics
                reg_keys = set(strategies_mod.STRATEGY_REGISTRY.keys())
                found = [k for k in strategy_diagnostics.keys() if k in reg_keys]
                for strategy in sorted(found):
                    entries = strategy_diagnostics.get(strategy, [])
                    rolling = strategy_diagnostics.get(strategy + "_rolling", {})
                    sig_map = strategy_diagnostics.get(strategy + "_signal", {})
                    exp_map = strategy_diagnostics.get(strategy + "_exposure", {})

                    with st.expander(f"{strategy}", expanded=False):
                        if entries:
                            df_entries = pd.DataFrame(entries)
                            agg = {
                                "max_signal": df_entries["signal_clean_max"].max(),
                                "min_signal": df_entries["signal_clean_min"].min(),
                                "avg_exposure": df_entries["exposure_mean"].mean(),
                                "trade_count": int(df_entries["trade_count"].sum()),
                                "turnover": float(df_entries["turnover"].sum()),
                                "avg_hold_days": float(df_entries["avg_hold_days"].mean()),
                                "nan_counts": int(df_entries["signal_raw_has_nan"].sum()),
                                "inf_counts": int(df_entries["signal_raw_has_inf"].sum()),
                            }
                            st.markdown("**Summary**")
                            st.table(pd.DataFrame([agg]))

                            # per-ticker table
                            st.markdown("**Per-ticker diagnostics**")
                            st.dataframe(df_entries, use_container_width=True)
                        else:
                            st.write("No per-ticker diagnostics for this strategy.")

                        # Rolling plots
                        if rolling:
                            rs = rolling.get("rolling_sharpe")
                            rv = rolling.get("rolling_vol")
                            rdd = rolling.get("rolling_dd")
                            if rs is not None:
                                st.subheader("Rolling Sharpe")
                                fig = go.Figure()
                                fig.add_trace(go.Scatter(x=rs.index, y=rs.values, name="Rolling Sharpe"))
                                st.plotly_chart(fig, use_container_width=True)
                            if rv is not None:
                                st.subheader("Rolling Volatility")
                                fig = go.Figure()
                                fig.add_trace(go.Scatter(x=rv.index, y=rv.values, name="Rolling Vol"))
                                st.plotly_chart(fig, use_container_width=True)
                            if rdd is not None:
                                st.subheader("Rolling Drawdown")
                                fig = go.Figure()
                                fig.add_trace(go.Scatter(x=rdd.index, y=rdd.values, name="Rolling DD"))
                                st.plotly_chart(fig, use_container_width=True)

                        # Exposure over time
                        if exp_map:
                            exp_df = pd.DataFrame(exp_map)
                            st.subheader("Exposure Over Time (per ticker)")
                            st.plotly_chart(signal_figure(exp_df, title=f"{strategy} Exposure"), use_container_width=True)

                        # total portfolio exposure
                        total_exp = strategy_diagnostics.get(strategy + "_total_exposure")
                        if total_exp is not None:
                            st.subheader("Total Portfolio Exposure")
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(x=total_exp.index, y=total_exp.values, name="Total Abs Exposure"))
                            st.plotly_chart(fig, use_container_width=True)

                        # Signal distribution
                        if sig_map:
                            all_signals = pd.concat(list(sig_map.values()), axis=0)
                            st.subheader("Signal Distribution")
                            fig = px.histogram(all_signals, nbins=50, title=f"{strategy} Signal Distribution")
                            st.plotly_chart(fig, use_container_width=True)



if __name__ == "__main__":
    main()
