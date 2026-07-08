# Quant Research Platform

A modular quantitative research platform built in Python for researching, developing, and evaluating systematic trading factors.

Unlike traditional backtesting projects, this platform is centered on **quantitative factor research**. Backtesting serves as the final validation stage after statistical analysis and signal generation.

---

# How to Run

## 1. Clone the repository

```bash
git clone https://github.com/<your-username>/quant-research-platform.git
cd quant-research-platform
```

---

## 2. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 3. Launch the Streamlit application

```bash
streamlit run src/app.py
```

This opens the interactive interface for:
- Loading stock data
- Running quantitative factors
- Backtesting factors
- Visualizing results

---

## 4. Run quantitative research

Run the Mutual Information and network analysis pipeline:

```bash
python -m research.stock_correlation
```

This script performs:
- Pearson Correlation Analysis
- Mutual Information Analysis
- Rolling MI Analysis
- Minimum Spanning Tree Construction
- Industry & Sector Network Analysis
- High-MI Stock Ranking
- Relative Performance Analysis

---

## 5. Utility scripts

Organize downloaded stock data:

```bash
python -m scripts.organize_data_folder
```

Download missing stock data:

```bash
python -m scripts.manual_ticker_download
```

---

# Notes

- The `data/` folder stores stock metadata and downloaded market data.
- The `cache/` folder stores temporary cached files to speed up repeated research.
- Cache files (`.pkl`) and downloaded stock CSV files are excluded from Git.
- Backtesting is used to evaluate quantitative factors after they have been researched and converted into trading signals.

# Project Goals

This project is designed to provide a complete workflow for quantitative research:

- Research new quantitative trading factors
- Analyze relationships between stocks using statistical methods
- Build reusable factor-generation pipelines
- Generate systematic investment signals
- Validate factors through historical backtesting
- Compare and improve factor performance over time

---

# Research Pipeline

```
Market Data
    │
    ▼
Data Collection & Cleaning
    │
    ▼
Feature Engineering
    │
    ▼
Factor Generation
    │
    ▼
Statistical Research
    │
    ├── Correlation
    ├── Mutual Information
    ├── Rolling Analysis
    ├── Industry Analysis
    ├── Sector Analysis
    ├── Network Analysis
    └── Factor Ranking
    │
    ▼
Signal Generation
    │
    ▼
Portfolio Construction
    │
    ▼
Backtesting
    │
    ▼
Performance Evaluation
```

---

# Current Research

## Statistical Research

Current statistical methods include:

- Pearson Correlation
- Empirical Mutual Information
- Mutual Information Distance
- Rolling Mutual Information
- Minimum Spanning Trees (MST)
- Industry & Sector Network Analysis
- Mean Mutual Information Factors
- High-MI Frequency Analysis
- Relative Price Divergence Analysis

---

## Quantitative Factors

Current factors include:

- Moving Average
- Momentum
- RSI
- Z-Score
- Volatility
- ATR
- EWMA
- Mean Mutual Information Factor *(ongoing research)*

Additional factors will continue to be developed and tested.

---

## Portfolio & Backtesting

Current capabilities include:

- Portfolio simulation
- Position sizing
- Dynamic exposure control
- Strategy comparison
- Performance metrics
- Equity curve generation
- Drawdown analysis
- Risk-adjusted performance evaluation

Backtesting is used to evaluate newly developed quantitative factors rather than serving as the primary purpose of the project.

---

# Project Structure

```
quant-research-platform/
│
├── src/
│   ├── app.py
│   ├── backtest.py
│   ├── indicators.py
│   ├── strategies.py
│   ├── plotting.py
│   ├── position_sizing.py
│   ├── data_loader.py
│   ├── data_manager.py
│   ├── clean_ticker_data.py
│   ├── update_ticker_data.py
│   └── yFinance.py
│
├── research/
│   └── stock_correlation.py
│
├── scripts/
│   ├── organize_data_folder.py
│   └── manual_ticker_download.py
│
├── data/
│   ├── technology/
│   ├── healthcare/
│   ├── financials/
│   ├── ...
│
├── cache/
│
├── clean_ticker_data/
│
├── docs/
│   ├── AGENTS.md
│   ├── CODEX_RULES.md
│   └── RULES.md
│
└── README.md
```

---

# Future Research

Planned areas of research include:

- Graph-based market representations
- Dynamic factor weighting
- Multi-factor models
- Factor combination and optimization
- Machine learning factor discovery
- Cross-sector relationship analysis
- Portfolio optimization
- Risk factor decomposition
- Statistical arbitrage research
- AI-assisted quantitative research

---

# Long-Term Vision

The long-term goal is to build a flexible quantitative research environment capable of:

- Developing original quantitative trading factors
- Studying market structure through statistical methods
- Discovering persistent market relationships
- Validating research through historical simulation
- Building systematic investment strategies from data-driven research

This project is intended to grow into a reusable research platform where new ideas can be rapidly tested, evaluated, and refined.