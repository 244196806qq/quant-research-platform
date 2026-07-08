import sys
import time
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.yFinance import download_ticker_data


def add_manual_ticker():
    ticker_list = pd.read_csv(PROJECT_ROOT / "data" / "ticker_list.txt")
    valid_tickers = []
    failed_tickers = []
    for ticker in ticker_list["Ticker"]:
        try:
            download_ticker_data(ticker)
            valid_tickers.append(ticker)
        except Exception:
            failed_tickers.append(ticker)
        time.sleep(5)
    print(f"Successfully downloaded data for: {valid_tickers}")
    print(f"Failed to download data for: {failed_tickers}")

if __name__ == "__main__":
    add_manual_ticker()