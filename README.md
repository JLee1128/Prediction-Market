# predmkt-xgb

XGBoost settlement-probability model and backtester for Kalshi's recurring markets (default: **KXBTCD**, hourly "BTC above $K"). The model's variables come from market microstructure (quotes, trade flow, order book), plus the underlying's spot price.

The design rationale, the variables used and the evaluation plan are in **[docs/STRATEGY.md](docs/STRATEGY.md)**.

## Setup (macOS, Apple silicon)
```bash
./scripts/setup.sh
```
This creates `.venv`, installs the requirements and `libomp` (XGBoost needs it on macOS), and runs the tests.

## Everyday use
```bash
# 1. Download settled history: markets, 1-min candles, trades, Coinbase spot. Resumable; reruns skip finished events.
.venv/bin/python -m predmkt download --series KXBTCD --days 30

# 2. Start recording live order books (history has none). Leave it running for weeks.
./scripts/record.sh            # poll mode, no API key needed
./scripts/record.sh ws         # every book change; needs an API key (below)

# 3. Train, evaluate, backtest, report
.venv/bin/python -m predmkt run --series KXBTCD --rebuild --ablation
.venv/bin/python -m predmkt run --series KXBTCD --mode maker     # maker variant

# Daily refresh (download the last 3 days + retrain + report)
./scripts/daily.sh
```
Each `run` writes `reports/<series>-<mode>-<timestamp>/`, containing:
- `report.md`: Brier vs. the market, calibration, P&L, drawdown, feature importance, and the go/no-go gate
- `trades.csv` and `test_predictions.parquet`
- plots

The trained model is saved in `models/`.

## Settings
Everything tunable is in `config.yaml`: series, history length, decision window, fee model, position caps, edge-threshold grid and bootstrap settings. Settings worth knowing:
- `features.feature_sets`: remove `spot` to test microstructure-only. `book` is used automatically once recorded order-book data exists.
- `strategy.mode`: `taker` or `maker`.
- `strategy.early_exit`: off by default, so positions are held to settlement.
- `strategy.compound`: off by default, so bets are sized off the starting bankroll and profits are not reinvested.
- `features.exclude`: features to leave out (day of week and hour of day by default).
- `model.use_market_offset`: the model starts from the market's own probability and learns corrections to it.

## API key (only for `record --mode ws`, and for future order code)
Create a key on Kalshi (Account → API keys), save the private key file outside this folder, then:
```bash
export KALSHI_API_KEY_ID=...
export KALSHI_PRIVATE_KEY_PATH=~/keys/kalshi.pem
```
Never commit keys; `*.pem` is git-ignored.

## Layout
```
predmkt/
  kalshi.py     REST client (public data, retries, historical-cutoff routing)
  download.py   settled markets, candles, trades
  spot.py       Coinbase 1-min candles
  recorder.py   live order-book recorder (poll / websocket)
  book.py       book rebuild, order-flow imbalance, book features
  features.py   1-minute decision grid, all variables, labels
  model.py      chronological event split, event-weighted XGBoost with market offset
  backtest.py   taker / maker simulation with fees, caps, maker trade-through fills
  metrics.py    Brier, calibration, event bootstrap, drawdown, P&L
  pipeline.py   run everything and write the report
scripts/        setup.sh, record.sh, daily.sh
tests/          unit tests (pytest)
```

## What this does not do yet
- **No live order placement.** That comes only after the gate in the report passes and two weeks of paper trading.
- Order-book variables need recorded data. Until then the model uses candles, trades and spot.
