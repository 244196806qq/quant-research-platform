import yfinance as yf
import pandas as pd

symbol = "JNJ"
try:
    df = yf.download(
        symbol,
        start="1900-01-01",
        end="2026-05-05",
        auto_adjust = False
    )
except Exception as e:
    print(e)

df.to_csv(f"data/{symbol}.csv")

df_new = pd.read_csv(f"data/{symbol}.csv")
df_new.drop([0,1], inplace = True)
df_new.columns = ["Date", "Adj Close", "Close", "High", "Low", "Open", "Volume"]
df_new.to_csv(f"data/{symbol}.csv", index = False)