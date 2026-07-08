# AGENTS.md

## Project Goal

This project is a beginner-friendly interactive quantitative investing and backtesting application built in Python.

The current project already supports:
- loading local historical stock CSV files
- calculating technical indicators
- generating trading signals
- simulating portfolio strategies
- graphing equity curves
- comparing strategies

The next stage is converting the project into an interactive app.

---

## Main Application Goals

The application should allow the user to:

1. Select stocks dynamically from a sidebar
2. Search for stocks using a search bar
3. Automatically load local CSV stock data
4. Automatically download missing stock data
5. Select and deselect strategies
6. Adjust portfolio holdings interactively
7. Adjust indicator parameters interactively
8. Run/update simulations using a button
9. View graphs and metrics interactively
10. Build toward future automatic stock-selection systems

---

## Data System

The `data/` folder acts as the local stock database.

### Stock Selection Rules

When a user searches for a ticker:

1. First check if:

```text
data/TICKER.csv
```

already exists.

2. If the file exists:
   - load local CSV data
   - do not redownload data

3. If the file does not exist:
   - use the existing `yFinance.py` download logic
   - download the stock data
   - save it into:

```text
data/TICKER.csv
```

   - then load the new file

The local CSV database is the foundation for future:
- automatic stock picking
- screening
- ranking systems

---

## Preferred UI

Preferred framework:

```text
Streamlit
```

Reason:
- simple Python integration
- sidebar support
- interactive widgets
- graphs/tables are easy to build
- beginner friendly

---

## Planned UI Features

### Sidebar

The sidebar should contain:

#### Stock Controls
- stock search bar
- add/remove stock buttons
- portfolio holdings input

#### Strategy Controls
- enable/disable strategies
- choose combined strategy indicators

#### Indicator Controls
Examples:
- MA windows
- RSI thresholds
- Z-score thresholds
- volatility thresholds
- ATR multipliers

#### Portfolio Controls
- initial money
- recurring deposit
- rebalance settings later

#### Simulation Controls
- update/run button

---

## Output Area

Main output area should display:

1. Equity curve graph
2. Strategy comparison graph
3. Performance metrics table

Later additions:
- drawdown chart
- volatility chart
- rolling returns
- correlation heatmap

---

## Current Project Structure

```text
main.py
data_loader.py
indicators.py
strategies.py
backtest.py
plotting.py
```

---

## Architectural Direction

The architecture should remain:

```text
DATA
→ INDICATORS
→ SIGNALS
→ BACKTEST
→ VISUALIZATION
```

Strategies should only generate signals.

Backtest logic should remain centralized.

---

## Long-Term Vision

Future goals include:
- automatic stock selection
- ranking stocks by indicators
- factor-based investing
- momentum rotation systems
- AI-assisted screening
- portfolio optimization
- strategy comparison dashboard

This project should remain:
- modular
- reusable
- beginner-readable
- easy to expand
