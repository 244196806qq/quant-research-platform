import pandas as pd

try:
    from .data_manager import ambiguous_unclassified_tickers, iter_local_data_files
except ImportError:
    from data_manager import ambiguous_unclassified_tickers, iter_local_data_files


WANTED_COLUMNS = ["Date", "Adj Close", "Close", "High", "Low", "Open", "Volume"]


def clean_one_file(file):
    df = pd.read_csv(file)

    # Keep only columns that exist
    df = df[[column for column in WANTED_COLUMNS if column in df.columns]]
    df = df.dropna(how="all")

    df.to_csv(file, index=False)


def clean_all_files():
    ambiguous = ambiguous_unclassified_tickers()
    if ambiguous:
        print(f"Ambiguous unclassified ticker files skipped: {', '.join(ambiguous)}")

    for file in iter_local_data_files():
        clean_one_file(file)


if __name__ == "__main__":
    clean_all_files()
