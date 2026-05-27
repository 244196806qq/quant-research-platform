import pandas as pd


def calculate_moving_average(df_asset):
    df_asset["MA10"] = df_asset["Adj Close"].rolling(10).mean()
    df_asset["MA50"] = df_asset["Adj Close"].rolling(50).mean()
    df_asset["MA200"] = df_asset["Adj Close"].rolling(200).mean()


def calculate_z_score(df_asset):
    df_asset["zscore_mean_50"] = df_asset["Adj Close"].rolling(50).mean()
    df_asset["zscore_std_50"] = df_asset["Adj Close"].rolling(50).std()
    df_asset["Z-Score"] = (df_asset["Adj Close"] - df_asset["zscore_mean_50"]) / df_asset["zscore_std_50"]


def calculate_momentum(df_asset):
    df_asset["MA50"] = df_asset["Adj Close"].rolling(50).mean()


def calculate_rsi(df_asset):
    delta = df_asset["Adj Close"].diff()
    gain = delta.clip(lower = 0)
    lose = -delta.clip(upper = 0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = lose.rolling(14).mean()
    rs = avg_gain / avg_loss
    df_asset["RSI"] = 100 - (100 / (1 + rs))


def calculate_rolling_std(df_asset):
    df_asset["Volatility"] = df_asset["Market_Return"].rolling(20).std()


def calculate_ATR(df_asset):
    high_low = df_asset["High"] - df_asset["Low"]
    high_prev_close = (df_asset["High"] - df_asset["Close"].shift(1)).abs()
    low_prev_close = (df_asset["Low"] - df_asset["Close"].shift(1)).abs()
    df_asset["TR"] = pd.concat([high_low, high_prev_close, low_prev_close], axis = 1).max(axis = 1)
    df_asset["ATR"] = df_asset["TR"].rolling(14).mean()
    df_asset["ATR Percent"] = (df_asset["ATR"] / df_asset["Close"]) * 100
    df_asset["ATR Mean"] = df_asset["ATR"].rolling(100).mean()


def calculate_EWMA(df_asset):
    df_asset["EWMA_Volatility"] = (df_asset["Market_Return"].ewm(span = 20).std())
    df_asset["EWMA_Mean"] = (df_asset["EWMA_Volatility"].rolling(100).mean())


def calculate_Annual_STD(df_asset):
    df_asset["Annual Volatility"] = df_asset["Market_Return"].rolling(20).std() * (252 ** 0.5)


def calculate_indicator(df_asset, indicator):
    if indicator == "MA":
        return calculate_moving_average(df_asset)
    elif indicator == "Z-Score":
        return calculate_z_score(df_asset)
    elif indicator == "Momentum":
        return calculate_momentum(df_asset)
    elif indicator == "RSI":
        return calculate_rsi(df_asset)
    elif indicator == "Rolling Std":
        return calculate_rolling_std(df_asset)
    elif indicator == "Avg True Range":
        return calculate_ATR(df_asset)
    elif indicator == "EWMA":
        return calculate_EWMA(df_asset)
    elif indicator == "Annual Std":
        return calculate_Annual_STD(df_asset)
    else:
        raise ValueError(f"Unknown indicator: {indicator}")
