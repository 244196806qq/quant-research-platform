import numpy as np
import pandas as pd
import plotly.graph_objects as go


def clean_plot_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"])
        df = df.set_index("Date")

    df.index = pd.to_datetime(df.index, errors="coerce")
    df = df.loc[~df.index.isna()]

    numeric_cols = df.select_dtypes(include="number").columns
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=numeric_cols, how="all")

    return df


def _layout(title: str) -> dict:
    return {
        "title": {"text": title, "x": 0.01, "xanchor": "left"},
        "legend": {"orientation": "h", "y": -0.2, "x": 0.0},
        "hovermode": "x unified",
        "xaxis": {"title": "Date", "showgrid": False},
        "yaxis": {"showgrid": True, "gridcolor": "rgba(200,200,200,0.2)"},
        "template": "plotly_white",
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
    }


def equity_curve_figure(strategy_equity: pd.DataFrame, buy_hold: pd.DataFrame, title: str = "Equity Curve") -> go.Figure:
    fig = go.Figure()

    for column in buy_hold.columns:
        fig.add_trace(
            go.Scatter(
                x=buy_hold.index,
                y=buy_hold[column],
                name=f"Buy & Hold {column}",
                mode="lines",
                line={"dash": "dash", "width": 2},
            )
        )

    for column in strategy_equity.columns:
        fig.add_trace(
            go.Scatter(
                x=strategy_equity.index,
                y=strategy_equity[column],
                name=column,
                mode="lines",
                line={"width": 3},
            )
        )

    fig.update_layout(_layout(title))
    return fig


def buy_and_hold_figure(buy_hold: pd.DataFrame, title: str = "Buy and Hold Equity") -> go.Figure:
    fig = go.Figure()
    for column in buy_hold.columns:
        fig.add_trace(
            go.Scatter(
                x=buy_hold.index,
                y=buy_hold[column],
                name=column,
                mode="lines",
                line={"dash": "dash", "width": 2},
            )
        )
    fig.update_layout(_layout(title))
    return fig


def drawdown_figure(equity_df: pd.DataFrame, title: str = "Drawdown") -> go.Figure:
    fig = go.Figure()
    for column in equity_df.columns:
        drawdown = equity_df[column] / equity_df[column].cummax() - 1
        fig.add_trace(
            go.Scatter(
                x=equity_df.index,
                y=drawdown,
                name=column,
                mode="lines",
                line={"width": 2},
            )
        )
    layout = _layout(title)
    layout["yaxis"]["title"] = "Drawdown"
    fig.update_layout(layout)
    return fig


def rolling_volatility_figure(returns_df: pd.DataFrame, window: int = 20, title: str = "Rolling Volatility") -> go.Figure:
    fig = go.Figure()
    for column in returns_df.columns:
        rolling_vol = returns_df[column].rolling(window).std() * np.sqrt(252)
        fig.add_trace(
            go.Scatter(
                x=rolling_vol.index,
                y=rolling_vol,
                name=column,
                mode="lines",
                line={"width": 2},
            )
        )
    layout = _layout(title)
    layout["yaxis"]["title"] = "Annualized Volatility"
    fig.update_layout(layout)
    return fig


def signal_figure(signal_df: pd.DataFrame, title: str = "Signals") -> go.Figure:
    fig = go.Figure()
    for column in signal_df.columns:
        fig.add_trace(
            go.Scatter(
                x=signal_df.index,
                y=signal_df[column],
                name=column,
                mode="lines",
                line={"width": 2},
            )
        )
    layout = _layout(title)
    layout["yaxis"]["title"] = "Signal Value"
    fig.update_layout(layout)
    return fig
        
