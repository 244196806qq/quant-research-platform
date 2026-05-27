import pandas as pd


def csv_reader(symbols):
    dfs = [pd.read_csv(f"data/{symbol}.csv") for symbol in symbols if symbol != "Strategy"]

    for i in range(len(dfs)):
        dfs[i]["Date"] = pd.to_datetime(dfs[i]["Date"])
        dfs[i] = dfs[i].sort_values("Date")
        dfs[i] = dfs[i].set_index("Date")

    shared_dates = dfs[0].index
    for df in dfs[1:]:
        shared_dates = shared_dates.intersection(df.index)

    shared_dates = shared_dates.sort_values()

    for i in range(len(dfs)):
        # Keep only exact trading dates shared by every asset. A date range can
        # still leave missing rows when one asset did not trade on a given day.
        dfs[i] = dfs[i].loc[shared_dates]

    return dfs
