from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

wanted_columns = ["Date", "Adj Close", "Close", "High", "Low", "Open", "Volume"]

for file in DATA_DIR.glob("*.csv"):
    df = pd.read_csv(file)

    # Keep only columns that exist
    df = df[[col for col in wanted_columns if col in df.columns]]
    df = df.dropna(how="all")

    df.to_csv(file, index=False)
