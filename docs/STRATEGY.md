# Strategy plan (v1)

Written 2026-11-12. API facts below were checked against Kalshi's live API that day.

## Decisions in one place

| Question | Decision |
|---|---|
| Venue | **Kalshi** |
| Market | **KXBTCD**: hourly "Bitcoin above $K at the top of the hour". **KXETHD** (same structure, ETH) is the second series, used to check results carry over. |
| What the model predicts | **P(YES settles)** for each strike, once a minute during the last 60 minutes before close |
| Model | XGBoost classifier that **starts from the market's own probability** (logit of the mid) and learns corrections to it |
| How we trade | Buy YES or NO when the model's probability beats the price **after fees** by a threshold, then **hold to settlement**. Maker variant: rest an order at the best bid instead of crossing the spread. |
| Capital | $1,000. Caps: $25 per market, $75 per event, $400 open at once |
| Pass/fail | The five metrics in the "Evaluation" section, measured on a test period nobody looked at while building |

## Why these choices

### Kalshi rather than Polymarket
- CFTC-regulated and open to US residents. Polymarket's international exchange is not.
- It gives free history: every settled market, 1-minute candles (best bid/ask and trade-price OHLC, volume, open interest) and **every trade with the aggressor side**. That is enough for trade-flow features without recording anything ourselves.
- It has a demo environment for testing order code before risking money.
- Fees are published per series through the API (`fee_type`, `fee_multiplier`):
  - Crypto series (KXBTCD, KXETHD, KXBTC15M) are `quadratic`: taker fee = 0.07 × contracts × p × (1−p), and **no maker fee**.
  - Economic series (KXCPI, KXPAYROLLS, KXFED) are `quadratic_with_maker_fees`, so makers pay 0.0175 × C × p(1−p).

### A recurring hourly market rather than one long-dated market
This is the "liquid market that doesn't resolve soon" concern.
- **Brier/calibration needs many resolved events.** KXBTCD resolves about 24 independent events a day (~700 a month). A Fed-rate market resolves about 8 a year, so you could never tell skill from luck with it.
- **It is very liquid.** On 2026-11-12, at-the-money strikes traded 100k–370k contracts in their final hour.
- **For the bot, the series never resolves.** A new hourly event is always listed, and the code rolls to it automatically. Only individual contracts resolve.
- The long-dated idea is not dropped: the live recorder can record KXFED order books in the background. Once the pipeline is validated on hourly data, the same code can be pointed at it.

### Predict the settlement probability, not the price direction at t+h
- The evaluation metrics (Brier score on realized outcomes) only make sense if the model outputs a probability of the outcome.
- Fees make short-horizon round trips expensive. At 50¢ the taker fee is 1.75¢, more than one 1¢ tick. Buying and then selling within h costs about 4.5¢, so the model would need to predict moves of several ticks. Holding to settlement pays one fee and no exit spread.
- Microstructure still matters: order-flow and order-book variables are the model's **inputs**, and in phase 2 they also drive execution timing (cross the spread now or rest an order).

### Why we also use the BTC spot price
These contracts settle on the 60-second average of the CF Benchmarks BTC index just before the hour. Most of what moves the order book is where BTC trades relative to the strike, so a model that ignores spot is reading the shadow and not the object. Adding it costs one free data feed (Coinbase 1-minute candles) and one formula (the probability that spot finishes above the strike, from recent volatility). We still train a no-spot version, so the data decides how much it matters.

## Variables the model uses

Every variable is scale-free (prices in cents, ratios, z-scores), so one model can be trained across all strikes and events.

**From history (available today)**

| Group | Variables |
|---|---|
| Price state | `mid`, `spread`, `logit_mid`, `pq` = mid·(1−mid) (volatility and fees scale with it) |
| Price dynamics | mid change over 1/5/15 min; std of 1-min mid changes over 15 min; number of mid changes in 15 min; bid and ask high−low within the last minute |
| Trade flow | trade-flow imbalance (buyer − seller aggressor volume) ÷ total over 1/5/15 min; trade counts; average trade size; largest-trade share; VWAP − mid |
| Activity | log volume over 1/5/15 min; 5-min volume ÷ 60-min baseline; open-interest change over 5/15 min |
| Clock | minutes to close (and log). Hour of day and day of week are computed but excluded by default (`features.exclude`): on 120 days the model scored better without them. |
| Underlying (crypto) | log(spot/strike); 60-min realized volatility; `spot_z` = log(spot/strike) ÷ (σ·√minutes left); `p_bs` = Φ(spot_z); `p_bs − mid`; spot returns over 1/5/15 min |

**Needs the live recorder (history doesn't include order books)**

| Group | Variables |
|---|---|
| Book state | level-1 queue imbalance (bid size − ask size) ÷ total; microprice − mid; depth imbalance within 3¢ |
| Order flow | order-flow imbalance (Cont, Kukanov, Stoikov 2014) summed over 1 and 5 min, normalized by recent top-of-book depth |

Expected value of each group, from the literature and the structure of these markets:
1. `p_bs − mid` and `spot_z`: the market lagging spot is the most direct edge.
2. Queue imbalance and order-flow imbalance: the strongest short-horizon order-book predictors.
3. Trade-flow imbalance: available historically, close behind (2).
4. Everything else: mostly helps the model know *when* to trust the other signals.

The pipeline reports permutation importance on held-out data, so this ranking gets checked rather than assumed.

## Data
- **History:** `python -m predmkt download` fetches settled markets, 1-minute candles and trades for the last 2 hours of each event, plus Coinbase spot. Kalshi moves data older than its "historical cutoff" (about 2 months) to `/historical/*` endpoints; the downloader reads both.
- **Order books:** not available historically. **Start the recorder now** (`python -m predmkt record`). Book variables become usable after about 2–4 weeks of recording.

## Model
- **Rows:** each strike, each minute from 60 down to 2 minutes before close. Kept only if the mid is between 3¢ and 97¢ and the spread is at most 10¢.
- **Label:** 1 if the market settled YES.
- **Starting point:** XGBoost `base_margin` = logit(mid), so the trees learn *corrections* to the market. If the model can't find anything, it collapses back to the market price instead of inventing confidence.
- **Weights:** each event gets equal total weight. Sixty snapshots of one event are one outcome, not sixty.
- **Split:** chronological by event: 60% train, 20% validation, 20% test. Events that straddle a boundary are dropped. Validation is used only for early stopping and choosing the edge threshold. **The test period is used once, for the final numbers.**

## Trading rules
- **Taker buy YES** if `p − ask − fee(ask) > θ`. **Taker buy NO** if `(1 − p) − (1 − bid) − fee(1 − bid) > θ`.
- **Maker:** rest a bid at the current best bid (YES) or best NO bid. The backtest counts it as filled only if a later trade prints **through** our price within the order's lifetime. This is conservative, and it automatically includes adverse selection.
- **Sizing:** quarter-Kelly on the edge, capped at $25 per market, $75 per event (strikes in the same hour are one bet on BTC) and $400 open, with at most one entry per market.
- **Selling:** the entry price never enters the decision. The default is to hold to settlement. The optional early exit sells only if `bid − fee > p + exit_threshold`, i.e. the market is paying more than the model thinks the contract is worth.
- θ is chosen on the validation period from a grid, then frozen for test.

## Evaluation (the team's five metrics, plus how each is computed)
1. **Net P&L after all modeled costs:** fees (Kalshi formula, rounded up to the cent per order), the spread (we pay the ask or receive the bid), and an optional slippage setting.
2. **Drawdown:** the largest peak-to-trough decline of cumulative realized P&L, in dollars and as % of the $1,000.
3. **Calibration:** Brier score and a reliability table (predicted vs. realized frequency in 10 bins), each event weighted equally. **The key comparison is the model's Brier score vs. the market's own Brier score (using the mid).** If we can't beat the market's probability, there is no edge.
4. **Out-of-sample:** everything reported comes from the chronologically last 20% of events, untouched while building.
5. **Snapshots are not independent:** confidence intervals come from resampling **events**, not rows. P&L is aggregated per event. Position caps are per event.

Also reported:
- **Entry regret:** how much better a price was available in the next 10 minutes. This answers "how close are our orders to the optimal level".
- Brier score by time-to-close.
- Hit rate, P&L per contract, daily Sharpe ratio.
- Permutation feature importance.
- **No-model baseline:** buy every side priced at or below 10¢ at each market's first decision minute, with the same fees, caps and test rows. On 120 days this simple rule made as much as the model, because cheap contracts right after the open have won more often than their price. The model has to beat it to count.

**Gate before real money:** all must hold on test.
- Model Brier < market Brier, with the event-bootstrap 95% CI of the difference below 0.
- Net P&L > 0, with the event-bootstrap CI's lower bound near or above 0.
- Net P&L above the no-model baseline.
- Max drawdown under 20% of the starting $1,000.
- Results hold on KXETHD with no retuning.

Then paper-trade for 2 weeks before going live at the $1,000 cap.

## Roadmap
- **Phase 0 (now):** download history and start the recorder.
- **Phase 1:** train and backtest on history without book variables, and read the report.
- **Phase 2:** add book variables once enough is recorded, and compare against phase 1 on the same test events.
- **Phase 3:** paper trading, then live order code (not written yet, on purpose), using the Kalshi demo environment first.
- **Later:** sibling-strike consistency (P(above K) must fall as K rises) and other series.
