import pandas as pd


def csv_reader(symbols, start_date=None, end_date=None):
    dfs = [pd.read_csv(f"data/{symbol}.csv") for symbol in symbols if symbol != "Strategy"]

    for i in range(len(dfs)):
        dfs[i]["Date"] = pd.to_datetime(dfs[i]["Date"])
        dfs[i] = dfs[i].sort_values("Date")
        dfs[i] = dfs[i].set_index("Date")

    # shared_dates = dfs[0].index
    # for df in dfs[1:]:
    #     shared_dates = shared_dates.intersection(df.index)

    # shared_dates = shared_dates.sort_values()

    # for i in range(len(dfs)):
    #     # Keep only exact trading dates shared by every asset. A date range can
    #     # still leave missing rows when one asset did not trade on a given day.
    #     dfs[i] = dfs[i].loc[shared_dates]

    all_dates = dfs[0].index
    for df in dfs[1:]:
        all_dates = all_dates.union(df.index)

    all_dates = all_dates.sort_values()

    for i in range(len(dfs)):
        dfs[i] = dfs[i].reindex(all_dates).ffill().bfill()

    if start_date is not None:
        for i in range(len(dfs)):
            dfs[i] = dfs[i].loc[dfs[i].index >= pd.to_datetime(start_date)]
    if end_date is not None:
        for i in range(len(dfs)):
            dfs[i] = dfs[i].loc[dfs[i].index <= pd.to_datetime(end_date)]
    
    return dfs
