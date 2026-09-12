from pathlib import Path

import pandas as pd

try:
    from .data_manager import local_filepath, normalize_ticker
except ImportError:
    from data_manager import local_filepath, normalize_ticker


def get_stock_files(symbols) -> list[Path]:
    stock_files = []
    seen = set()

    for symbol in symbols:
        ticker = normalize_ticker(symbol)
        if ticker in seen:
            raise ValueError(f"Ticker requested more than once: {ticker}")
        seen.add(ticker)
        stock_files.append(local_filepath(ticker))

    return stock_files


def load_stock_csv(file: Path) -> pd.DataFrame:
    path = Path(file)
    df = pd.read_csv(path)
    if "Date" not in df.columns:
        raise ValueError(f"Ticker CSV is missing required Date column: {path}")
    if df.empty:
        raise ValueError(f"Ticker CSV contains no market data: {path}")

    df["Date"] = pd.to_datetime(df["Date"])
    return df.sort_values("Date").set_index("Date")


def csv_reader(symbols, start_date=None, end_date=None):
    stock_files = get_stock_files(symbols)
    if not stock_files:
        raise ValueError("At least one ticker is required.")

    dfs = [load_stock_csv(file) for file in stock_files]

    all_dates = dfs[0].index
    for df in dfs[1:]:
        all_dates = all_dates.union(df.index)
    all_dates = all_dates.sort_values()

    for index, df in enumerate(dfs):
        dfs[index] = df.reindex(all_dates).ffill().bfill()

    if start_date is not None:
        start = pd.to_datetime(start_date)
        dfs = [df.loc[df.index >= start] for df in dfs]
    if end_date is not None:
        end = pd.to_datetime(end_date)
        dfs = [df.loc[df.index <= end] for df in dfs]

    return dfs
