import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
TODAY_DATE = date.today()
TODAY_WEEKDAY = TODAY_DATE.weekday()


def update_one_file(file):
    file_path = Path(file)
    symbol = file_path.stem
    df = pd.read_csv(file_path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")

    last_date = df["Date"].iloc[-1].date()

    start_date = last_date + timedelta(days=1)
    end_date = (TODAY_DATE) - (timedelta(days=TODAY_WEEKDAY - 4) if TODAY_WEEKDAY > 4 else timedelta(days=0))

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

    if isinstance(new_data.columns, pd.MultiIndex):
        new_data.columns = new_data.columns.get_level_values(0)

    new_data = new_data.reset_index()
    new_data = new_data[wanted_columns]
    new_data.columns.name = None

    if new_data.empty:
        print(f"{symbol}: no new market data")
        return

    combined = pd.concat([df, new_data], ignore_index=True)
    combined["Date"] = pd.to_datetime(combined["Date"])
    combined = combined.drop_duplicates(subset=["Date"], keep="last")
    combined = combined.sort_values("Date")

    combined.to_csv(file_path, index=False)

    print(f"{symbol}: updated from {last_date} to {combined['Date'].iloc[-1].date()}")


def update_all_files():
    files = sorted(DATA_DIR.glob("**/_stocks/*.csv"))
    for file in files:
        update_one_file(file)
        time.sleep(0.1)


if __name__ == "__main__":
    update_all_files()