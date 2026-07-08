import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.update_ticker_data import update_one_file
from src.yFinance import download_ticker_data


DATA_DIR = PROJECT_ROOT / "data"
STRUCTURE_FILE = DATA_DIR / "folder_structure.json"
METADATA_FILE = DATA_DIR / "stock_metadata.json"


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")

    with open(path, "r") as f:
        return json.load(f)


def create_subfolders(structure: dict, current_path: Path):
    for folder_name, subfolders in structure.items():
        folder_path = current_path / folder_name
        folder_path.mkdir(parents=True, exist_ok=True)

        if isinstance(subfolders, dict):
            create_subfolders(subfolders, folder_path)


def get_category_folder_path(stock_info: dict) -> Path:
    path = DATA_DIR

    for level in [
        stock_info.get("sector"),
        stock_info.get("industry"),
        stock_info.get("subindustry"),
    ]:
        if level:
            path /= level

    return path


def get_stock_folder_path(stock_info: dict) -> Path:
    return get_category_folder_path(stock_info) / "_stocks"


def write_stock_lists(stock_metadata: dict, starting_ticker = "NVDA"):
    folder_to_stocks = {}
    at_starting = False

    for ticker, info in stock_metadata.items():
        if not at_starting:
            if starting_ticker == ticker:
                at_starting = True
            else:
                continue
        if at_starting:
            folder_path = get_category_folder_path(info)
            folder_path.mkdir(parents=True, exist_ok=True)

            folder_to_stocks.setdefault(folder_path, [])
            folder_to_stocks[folder_path].append(ticker)

    for folder_path, tickers in folder_to_stocks.items():
        with open(folder_path / "stock_list.txt", "w") as f:
            f.write("\n".join(sorted(tickers)))


def ensure_stock_data_in_folders(stock_metadata: dict, starting_ticker="NVDA"):
    at_starting = False
    for ticker, info in stock_metadata.items():
        if not at_starting:
            if starting_ticker == ticker:
                at_starting = True
            else:
                continue
        if at_starting:
            folder_path = get_stock_folder_path(info)
            file_path = folder_path / f"{ticker}.csv"
            print(file_path)
            if file_path.exists():
                update_one_file(str(file_path))
                time.sleep(0.5)
            else:
                print(f"Downloading {ticker} ...")
                download_ticker_data(
                    ticker,
                    data_dir=str(folder_path),
                    auto_adjust=False,
                )
                time.sleep(1)


def main():
    structure = load_json(STRUCTURE_FILE)
    stock_metadata = load_json(METADATA_FILE)

    create_subfolders(structure, DATA_DIR)
    write_stock_lists(stock_metadata, "PTC")
    ensure_stock_data_in_folders(stock_metadata, "NVDA")

    print("Folders created, stock lists updated, and stock data downloaded/updated.")


if __name__ == "__main__":
    main()