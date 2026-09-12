import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf

try:
    from .data_manager import (
        ambiguous_unclassified_tickers,
        iter_local_data_files,
        missing_metadata_tickers,
    )
except ImportError:
    from data_manager import (
        ambiguous_unclassified_tickers,
        iter_local_data_files,
        missing_metadata_tickers,
    )

TODAY_DATE = date.today()


def download_end_date(today: date) -> date:
    """Return yfinance's exclusive end date after the latest weekday."""
    if today.weekday() == 5:
        latest_market_date = today - timedelta(days=1)
    elif today.weekday() == 6:
        latest_market_date = today - timedelta(days=2)
    else:
        latest_market_date = today
    return latest_market_date + timedelta(days=1)


def update_one_file(file):
    file_path = Path(file)
    symbol = file_path.stem.strip().upper()
    df = pd.read_csv(file_path)
    if df.empty:
        raise ValueError(f"{file_path} contains no market data")
    if "Date" not in df.columns:
        raise ValueError(f"{file_path} is missing required column: Date")

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")

    last_date = df["Date"].iloc[-1].date()

    start_date = last_date + timedelta(days=1)
    end_date = download_end_date(TODAY_DATE)

    if start_date >= end_date:
        print(f"{symbol}: already up to date")
        return

    new_data = yf.download(
        symbol,
        start=start_date,
        end=end_date,
        auto_adjust=False,
        progress=False,
    )

    wanted_columns = ["Date", "Adj Close", "Close", "High", "Low", "Open", "Volume"]

    if new_data is None or new_data.empty:
        print(f"{symbol}: no new market data")
        return

    if isinstance(new_data.columns, pd.MultiIndex):
        new_data.columns = new_data.columns.get_level_values(0)

    new_data = new_data.reset_index()
    missing_columns = [column for column in wanted_columns if column not in new_data.columns]
    if missing_columns:
        raise ValueError(
            f"Downloaded data for {symbol} is missing required columns: {missing_columns}"
        )

    new_data = new_data[wanted_columns]
    new_data.columns.name = None

    combined = pd.concat([df, new_data], ignore_index=True)
    combined["Date"] = pd.to_datetime(combined["Date"])
    combined = combined.drop_duplicates(subset=["Date"], keep="last")
    combined = combined.sort_values("Date")

    combined.to_csv(file_path, index=False)

    print(f"{symbol}: updated from {last_date} to {combined['Date'].iloc[-1].date()}")


def update_all_files():
    missing = missing_metadata_tickers()
    if missing:
        print(f"Metadata tickers missing canonical CSV files: {', '.join(missing)}")

    ambiguous = ambiguous_unclassified_tickers()
    if ambiguous:
        print(f"Ambiguous unclassified ticker files skipped: {', '.join(ambiguous)}")

    for file in iter_local_data_files():
        update_one_file(file)
        time.sleep(0.1)


if __name__ == "__main__":
    update_all_files()
