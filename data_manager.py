"""Simple data manager for loading local CSVs and downloading missing tickers.

This module follows the project's RULES: always prefer local CSV files
in the `data/` folder and only download missing tickers using reusable logic
from `yFinance.py`.
"""
from pathlib import Path
import pandas as pd

from yFinance import download_ticker_data


DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)


def _normalize_ticker(ticker: str) -> str:
    return ticker.strip().upper()


def local_filepath(ticker: str) -> Path:
    t = _normalize_ticker(ticker)
    return DATA_DIR / f"{t}.csv"


def ticker_exists(ticker: str) -> bool:
    return local_filepath(ticker).exists()


def load_local_data(ticker: str) -> pd.DataFrame:
    """Load a local CSV from `data/TICKER.csv` and prepare a Date index."""
    path = local_filepath(ticker)
    df = pd.read_csv(path)
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").set_index("Date")
    return df


def ensure_ticker_data(
    ticker: str,
    start: str = "1986-08-20",
    end: str = "2026-05-05",
    data_dir: str = "data",
    auto_adjust: bool = False,
) -> Path:
    """Ensure `data/TICKER.csv` exists locally, downloading it if needed."""
    path = local_filepath(ticker)
    if path.exists():
        return path

    downloaded_path = download_ticker_data(
        ticker,
        start=start,
        end=end,
        data_dir=data_dir,
        auto_adjust=auto_adjust,
    )

    if not downloaded_path.exists():
        raise FileNotFoundError(f"Expected downloaded file not found for {ticker}: {downloaded_path}")
    return downloaded_path
