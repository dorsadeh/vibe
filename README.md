# Leverage Simulator

A local research tool for simulating leveraged portfolio strategies based on technical indicators.

## Features

- **Multi-asset support**: SPY, GLD, TLT, BIL with 17+ years of historical data
- **Two portfolio modes**:
  - Fixed weights allocation
  - Momentum rotation (picks top-performing assets)
- **Flexible leverage rules**: Define when to apply leverage using technical indicators
- **Interest cost modeling**: Uses 3-Month T-Bill rate + broker spread
- **Margin monitoring**: Tracks margin utilization and forced deleveraging
- **Interactive UI**: Streamlit-based dashboard with charts and exports

## Quick Start

```bash
# Clone and enter directory
cd leverage-simulator

# Run with Docker (default)
./run.sh

# Or run with local Python venv
./run.sh --local
```

Then open http://localhost:8501 in your browser.

## Running Options

### Docker (Default)

Requires Docker installed. Data is persisted in `data/` and `exports/` directories.

```bash
./run.sh                      # Run with Docker
docker compose up -d          # Run in background
docker compose down           # Stop
```

### Local Python

Requires Python 3.10+. Creates a virtual environment automatically.

```bash
./run.sh --local              # Run with venv
```

## Configuration

### Via UI

The Streamlit sidebar provides controls for all parameters.

### Via YAML

Example configs are in `config/examples/`:

```yaml
# config/examples/momentum_rotation.yaml
portfolio:
  mode: momentum_rotation
  momentum_lookback: 126
  momentum_top_n: 1

leverage:
  signal_asset: SPY
  rule: "(SPY.close > SPY.SMA_200)"
  target_leverage: 1.5
```

## Leverage Rules

Rules use a simple expression language:

```
(SPY.close > SPY.SMA_200) AND (SPY.RSI_14 < 70) AND NOT (SPY.SMA_50 < SPY.SMA_200)
```

### Available Indicators

| Indicator | Description |
|-----------|-------------|
| `{ASSET}.close` | Adjusted closing price |
| `{ASSET}.SMA_20` | 20-day Simple Moving Average |
| `{ASSET}.SMA_50` | 50-day Simple Moving Average |
| `{ASSET}.SMA_200` | 200-day Simple Moving Average |
| `{ASSET}.EMA_12` | 12-day Exponential Moving Average |
| `{ASSET}.EMA_26` | 26-day Exponential Moving Average |
| `{ASSET}.RSI_14` | 14-day Relative Strength Index |

### Operators

- Comparison: `>`, `<`, `>=`, `<=`, `==`, `!=`
- Logical: `AND`, `OR`, `NOT`
- Grouping: `(`, `)`

## Interest Cost Calculation

```
daily_rate = (annual_rate + broker_spread) / day_count
borrowed = equity * (leverage_factor - 1)
daily_interest = borrowed * daily_rate
```

Default settings:
- Base rate: 3-Month T-Bill (from FRED)
- Broker spread: 1.5%
- Day count: 360 (money market convention)

## Performance Metrics

| Metric | Description |
|--------|-------------|
| CAGR | Compound Annual Growth Rate |
| Volatility | Annualized standard deviation |
| Max Drawdown | Largest peak-to-trough decline |
| Sharpe Ratio | Risk-adjusted return (vs T-Bill) |
| Sortino Ratio | Downside risk-adjusted return |
| Calmar Ratio | CAGR / Max Drawdown |

## Project Structure

```
leverage-simulator/
├── app/
│   └── main.py              # Streamlit UI
├── config/
│   ├── default.yaml
│   └── examples/
├── data/                    # Cached market data
├── exports/                 # CSV exports
├── src/
│   ├── data/               # Data loading and caching
│   ├── indicators/         # Technical indicators
│   ├── rules/              # Rule parsing engine
│   ├── simulation/         # Backtest engine
│   └── metrics/            # Performance metrics
├── tests/                  # Unit tests
├── requirements.txt
├── run.sh
└── README.md
```

## Running Tests

```bash
source venv/bin/activate
pytest tests/ -v
```

## Data Sources

- **Price data**: Yahoo Finance (via yfinance) with Stooq fallback
- **Interest rates**: FRED (3-Month T-Bill, Federal Funds Rate)

Data is cached locally in `data/` directory as parquet files.

## Limitations

- **Research only**: Not for production trading
- **No intraday**: Uses end-of-day data only
- **Simplified margin**: Uses single maintenance margin threshold
- **No slippage**: Assumes execution at close prices

## License

MIT License - For research and educational purposes.
