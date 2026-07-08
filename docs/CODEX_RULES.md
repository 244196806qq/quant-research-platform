# Codex Rules for Quant Backtesting Project

## Project Summary

This project is a beginner-friendly Python backtesting framework.

It loads historical stock CSV files, calculates indicators, converts indicators into strategy signals, applies portfolio weights, simulates portfolio equity with recurring deposits, then graphs and prints results.

Current flow:

```text
main.py
→ data_loader.py
→ indicators.py
→ strategies.py
→ backtest.py
→ plotting.py

File Responsibilities
main.py

Controls symbols, holdings, initial money, deposits, selected strategies, and runs the full program.

data_loader.py

Loads CSV files from the data/ folder, parses dates, sorts data, sets the date index, and aligns all assets to the same usable dates.

indicators.py

Calculates indicators such as:

Moving averages
Z-score
Momentum
RSI
Rolling standard deviation
ATR
EWMA volatility
Annualized volatility
strategies.py

Converts indicators into trading signals.

Each strategy should return a pandas Series called Signal.

backtest.py

Calculates daily returns, buy-and-hold equity curves, and strategy equity curves.

Strategy returns must use the previous day’s signal with .shift(1) to avoid lookahead bias.

plotting.py

Graphs buy-and-hold and strategy equity curves and prints percentage returns.

Required Coding Rules
Keep the code beginner-readable.
Do not over-engineer the project.
Do not introduce classes unless clearly useful.
Keep functions small and focused.
Avoid duplicated code where simple helper functions can solve it.
Preserve the current file structure unless there is a strong reason to change it.
Do not silently change strategy meaning.
Add comments for non-obvious quant logic.
Do not use future data.
Use .shift(1) when applying signals to returns.
Strategies should return a Series, not a DataFrame.
Indicator functions may add columns to the asset dataframe.
Avoid hardcoded stock names inside reusable functions.
Align all assets using exact shared trading dates.
Deposits should be calculated consistently with the 21-trading-day schedule.
Quant Logic Rules
Signals

A signal represents position exposure.

Examples:

1.0 = full exposure
0.5 = half exposure
0.0 = no exposure
2.0 = leveraged exposure

Only use values above 1.0 if leverage is intentional.

Portfolio Return

Portfolio daily return should be:

sum(asset_weight × signal × asset_daily_return)
Lookahead Bias

Do not apply today’s signal to today’s return.

Correct:

signal.shift(1) * market_return

ATR

ATR should use consistent price columns.

Do not mix raw High/Low with adjusted Close unless the full OHLC series is adjusted consistently.