# RULES.md

## General Coding Rules

1. Keep the project beginner-readable.
2. Do not over-engineer the codebase.
3. Prefer simple functions over complex abstractions.
4. Avoid unnecessary classes unless clearly beneficial.
5. Keep logic modular and reusable.
6. Avoid duplicated logic where possible.
7. Preserve the current project structure unless necessary.
8. Add comments for non-obvious quant logic.
9. Do not silently change strategy behavior.
10. Keep variable names descriptive.

---

## UI Rules

### Preferred Framework

Use:

```text
Streamlit
```

unless another framework is clearly necessary.

### Sidebar Requirements

Sidebar should support:
- stock search
- strategy selection
- parameter controls
- holdings input
- update button

### Graph Requirements

Graphs should:
- remain readable with many strategies
- support resizing
- clearly label strategies

### Metrics Table

Metrics table should support:
- many strategies
- future metrics expansion

---

## Data Rules

### Local Data Priority

Always prioritize:

```text
data/*.csv
```

before downloading data.

### Missing Ticker Logic

If ticker CSV does not exist:
1. download using existing yfinance logic
2. save into `data/`
3. load locally afterward

Do not repeatedly redownload existing data.

---

## Quant Rules

### Signals

Signal values represent exposure.

Examples:

```text
1.0 = full exposure
0.5 = half exposure
0.0 = no exposure
2.0 = leveraged exposure
```

Only use values above `1.0` if leverage is intentional.

### Lookahead Bias

Never use future information.

Correct:

```python
signal.shift(1) * market_return
```

Incorrect:

```python
signal * market_return
```

### Portfolio Return Logic

Portfolio return should follow:

```text
sum(asset_weight × signal × asset_return)
```

### Deposit Logic

Deposits occur every:

```text
21 trading days
```

Use consistent logic everywhere.

---

## Indicator Rules

Indicators may:
- add columns to dataframes
- reuse existing columns
- support parameter customization

Indicators should not:
- directly modify portfolio equity
- directly run backtests
- directly plot results

---

## Strategy Rules

Strategies should:
- generate signals only
- return a pandas Series named `Signal`

Strategies should not:
- calculate portfolio returns
- calculate equity curves
- perform plotting

---

## Backtest Rules

Backtest functions should:
- centralize portfolio simulation
- handle deposits consistently
- handle portfolio weighting
- apply signals using `shift(1)`

---

## Plotting Rules

Plotting should:
- remain independent from strategy logic
- avoid hardcoded stock names
- support many strategies

---

## Future Expansion Rules

The architecture should remain compatible with:
- automatic stock screening
- factor ranking
- AI-assisted systems
- portfolio optimization
- additional metrics
- additional visualizations

Do not tightly couple components together.
