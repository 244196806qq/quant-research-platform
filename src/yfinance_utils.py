import argparse
from pathlib import Path

import pandas as pd
import yfinance as yf


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _normalize_symbol(symbol: str) -> str:
    return symbol.strip().upper()


def _flatten_columns(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        if symbol in df.columns.get_level_values(-1):
            df.columns = df.columns.get_level_values(0)
        else:
            df.columns = [
                " ".join(filter(None, map(str, col))).strip()
                for col in df.columns.to_flat_index()
            ]

    return df


def download_ticker_data(
    symbol: str,
    start: str = "1986-08-20",
    end: str = "2026-06-01",
    data_dir: str = "data",
    auto_adjust: bool = False,
) -> Path:
    symbol = _normalize_symbol(symbol)
    if not symbol:
        raise ValueError("Ticker symbol must not be empty.")

    target_dir = Path(data_dir)
    if not target_dir.is_absolute():
        target_dir = PROJECT_ROOT / target_dir
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / f"{symbol}.csv"

    try:
        df = yf.download(symbol, start=start, end=end, auto_adjust=auto_adjust, progress=False)
    except Exception as exc:
        raise RuntimeError(f"Failed to download {symbol}: {exc}") from exc

    if df is None or df.empty:
        raise ValueError(f"No data downloaded for {symbol}. Check the symbol and date range.")

    df = _flatten_columns(df, symbol)
    df = df.reset_index()

    expected_columns = ["Date", "Adj Close", "Close", "High", "Low", "Open", "Volume"]
    if not set(expected_columns).issubset(set(df.columns)):
        raise ValueError(
            f"Downloaded data for {symbol} is missing required columns. Got: {list(df.columns)}"
        )

    df = df[expected_columns]
    df.to_csv(target_path, index=False, date_format="%Y-%m-%d")
    return target_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download stock history to CSV using yfinance.")
    parser.add_argument("symbol", help="Ticker symbol to download, e.g. AAPL")
    parser.add_argument("--start", default="1986-08-20", help="Start date in YYYY-MM-DD format")
    parser.add_argument("--end", default="2026-05-05", help="End date in YYYY-MM-DD format")
    parser.add_argument("--data-dir", default="data", help="Directory to save CSV data")
    parser.add_argument("--auto-adjust", action="store_true", help="Use auto-adjusted prices")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    path = download_ticker_data(
        args.symbol,
        start=args.start,
        end=args.end,
        data_dir=args.data_dir,
        auto_adjust=args.auto_adjust,
    )
    print(f"Saved {args.symbol.upper()} data to {path}")


if __name__ == "__main__":
    main()
