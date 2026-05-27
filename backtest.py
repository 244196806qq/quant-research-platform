from strategies import determine_strategy
import pandas as pd


DEPOSIT_FREQUENCY_DAYS = 21


def count_deposits(num_days, deposit_frequency=DEPOSIT_FREQUENCY_DAYS):
    return (num_days - 1) // deposit_frequency


def calculate_equity_curve(daily_returns, initial_money, deposit, deposit_frequency=DEPOSIT_FREQUENCY_DAYS):
    equity = []
    current_value = initial_money

    for day_number, daily_return in enumerate(daily_returns.fillna(0)):
        current_value *= (1 + daily_return)
        if day_number % deposit_frequency == 0 and day_number != 0:
            current_value += deposit
        equity.append(current_value)

    return pd.Series(equity, index=daily_returns.index, name="Market_Equity")


def calculate_daily_equity(dfs, initial_money, deposit):
    for i in range(len(dfs)):
        dfs[i]["Market_Equity"] = calculate_equity_curve(dfs[i]["Market_Return"], initial_money, deposit)


def calculate_daily_return(dfs):
    for i in range(len(dfs)):
        dfs[i]["Market_Return"] = dfs[i]["Adj Close"].pct_change()


def simulate_strategy(dfs, holding, symbols, strategy, initial_money, deposit, indicators=None):
    df_strategy = pd.DataFrame(index = dfs[0].index)
    df_strategy["Market_Return"] = 0.0

    for i in range(len(dfs)):
        signal = determine_strategy(dfs[i], strategy, indicators)
        strategy_return = signal.shift(1).fillna(0) * dfs[i]["Market_Return"].fillna(0)
        df_strategy["Market_Return"] += holding[symbols[i]] * strategy_return

    df_strategy["Market_Equity"] = calculate_equity_curve(
        df_strategy["Market_Return"],
        initial_money,
        deposit
    )
    return df_strategy["Market_Equity"]
