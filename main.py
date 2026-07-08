from data_loader import csv_reader
from plotting import open_window, show_return
import pandas as pd
from backtest import (
    simulate_strategy, 
    calculate_daily_equity, 
    calculate_daily_return,
    count_deposits
)


def main():
    symbols = ["MSFT", "JPM", "XOM", "PG", "JNJ", "Strategy"]
    holding = {
        "MSFT": 0.30,
        "JPM": 0.20,
        "XOM": 0.15,
        "PG": 0.20,
        "JNJ": 0.15,
    }

    initial_money = 10000
    deposit = 100

    dfs = csv_reader(symbols)
    calculate_daily_return(dfs)
    calculate_daily_equity(dfs, initial_money, deposit)

    num_days = len(dfs[0])
    num_deposits = count_deposits(num_days)
    final_money = initial_money + deposit * num_deposits
    
    strategy = [
        "MA", 
        "Z-Score", 
        "Momentum", 
        "RSI", 
        "Rolling Std", 
        "Avg True Range", 
        "EWMA", 
        "Annual Std", 
        "Combined",
    ]
    df_strategy = pd.DataFrame()
    for i in range(len(strategy)):
        if strategy[i] != "Combined":
            df_strategy[strategy[i]] = simulate_strategy(dfs, holding, symbols, strategy[i], initial_money, deposit)
        else:
            indicators = {
                "trend": "MA",
                "volatility": "EWMA",
                "timing": "RSI"
            }
            df_strategy[strategy[i]] = simulate_strategy(dfs, holding, symbols, strategy[i], initial_money, deposit, indicators)

    open_window(dfs, df_strategy, symbols, strategy, final_money)

if __name__ == "__main__":
    main()
