import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_manager import canonical_filepath, ensure_ticker_data, local_filepath  # noqa: E402
from src.update_ticker_data import update_one_file  # noqa: E402


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


def write_stock_lists(stock_metadata: dict):
    folder_to_stocks = {}

    for ticker in stock_metadata:
        folder_path = canonical_filepath(ticker, stock_metadata).parent.parent
        folder_path.mkdir(parents=True, exist_ok=True)

        folder_to_stocks.setdefault(folder_path, [])
        folder_to_stocks[folder_path].append(ticker)

    for folder_path, tickers in folder_to_stocks.items():
        with open(folder_path / "stock_list.txt", "w") as f:
            f.write("\n".join(sorted(tickers)))


def ensure_stock_data_in_folders(stock_metadata: dict, starting_ticker=None):
    at_starting = starting_ticker is None
    for ticker in stock_metadata:
        if not at_starting:
            if starting_ticker == ticker:
                at_starting = True
            else:
                continue
        if at_starting:
            expected_path = canonical_filepath(ticker, stock_metadata)
            print(expected_path)
            try:
                existing_path = local_filepath(ticker)
            except FileNotFoundError:
                print(f"Downloading {ticker} ...")
                ensure_ticker_data(ticker, auto_adjust=False)
                time.sleep(1)
            else:
                update_one_file(existing_path)
                time.sleep(0.5)


def main():
    structure = load_json(STRUCTURE_FILE)
    stock_metadata = load_json(METADATA_FILE)

    create_subfolders(structure, DATA_DIR)
    write_stock_lists(stock_metadata)
    ensure_stock_data_in_folders(stock_metadata)

    print("Folders created, stock lists updated, and stock data downloaded/updated.")


if __name__ == "__main__":
    main()
