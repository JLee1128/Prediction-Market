"""Download settled-market history (markets, 1-min candles, trades) and spot prices.

Layout (one folder per event, resumable via a _done marker):
  data/raw/kalshi/<SERIES>/series.json
  data/raw/kalshi/<SERIES>/events/<EVENT>/{markets,candles,trades}.parquet
  data/raw/spot/<PRODUCT>/<YYYY-MM-DD>.parquet
"""
from __future__ import annotations

import json
import logging
import math
import time

import pandas as pd

from .config import data_dir, iso_to_ts
from .kalshi import KalshiClient
from .spot import load_spot

log = logging.getLogger(__name__)


def _f(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return math.nan


def market_row(m: dict, series: str) -> dict:
    return {
        "ticker": m["ticker"],
        "event_ticker": m["event_ticker"],
        "series": series,
        "open_ts": iso_to_ts(m.get("open_time")),
        "close_ts": iso_to_ts(m.get("close_time")),
        "result": m.get("result") or "",
        "strike_type": m.get("strike_type") or "",
        "floor_strike": _f(m.get("floor_strike")),
        "cap_strike": _f(m.get("cap_strike")),
        "volume": _f(m.get("volume_fp", m.get("volume"))),
        "expiration_value": _f(m.get("expiration_value")),
    }


def _px(d: dict, k: str) -> float:
    # Live candles label prices "close_dollars"; the historical archive labels them "close".
    return _f(d.get(f"{k}_dollars", d.get(k)))


def candle_row(c: dict, ticker: str) -> dict:
    bid, ask, px = c.get("yes_bid") or {}, c.get("yes_ask") or {}, c.get("price") or {}
    return {
        "ticker": ticker,
        "ts": int(c["end_period_ts"]),
        "bid_o": _px(bid, "open"), "bid_h": _px(bid, "high"), "bid_l": _px(bid, "low"), "bid_c": _px(bid, "close"),
        "ask_o": _px(ask, "open"), "ask_h": _px(ask, "high"), "ask_l": _px(ask, "low"), "ask_c": _px(ask, "close"),
        "px_o": _px(px, "open"), "px_h": _px(px, "high"), "px_l": _px(px, "low"), "px_c": _px(px, "close"),
        "px_mean": _px(px, "mean"),
        "volume": _f(c.get("volume_fp", c.get("volume"))),
        "oi": _f(c.get("open_interest_fp", c.get("open_interest"))),
    }


def trade_row(t: dict) -> dict:
    side = t.get("taker_outcome_side") or t.get("taker_side")
    return {
        "trade_id": t.get("trade_id"),
        "ticker": t["ticker"],
        "ts": iso_to_ts(t.get("created_time")),
        "yes_price": _f(t.get("yes_price_dollars")),
        "count": _f(t.get("count_fp", t.get("count"))),
        "taker_yes": side == "yes",   # aggressor bought YES (= lifted the YES ask)
        "is_block": bool(t.get("is_block_trade", False)),
    }


def _fetch_trades(client: KalshiClient, ticker: str, t0: int, t1: int, trades_cut: float) -> list[dict]:
    rows: list[dict] = []
    if t0 < trades_cut:
        rows += client.trades(ticker, t0, min(t1, int(trades_cut)), historical=True)
    if t1 >= trades_cut:
        rows += client.trades(ticker, max(t0, int(trades_cut)), t1, historical=False)
    return rows


def download_series(cfg: dict, client: KalshiClient, series: str, days: float) -> None:
    dcfg, fcfg = cfg["data"], cfg["features"]
    root = data_dir(cfg) / "raw" / "kalshi" / series
    (root / "events").mkdir(parents=True, exist_ok=True)

    info = client.series(series) or {}
    (root / "series.json").write_text(json.dumps(info, indent=1))

    cut = client.historical_cutoff()
    market_cut = iso_to_ts(cut.get("market_settled_ts"))
    trades_cut = iso_to_ts(cut.get("trades_created_ts"))
    log.info("%s: historical cutoff markets=%s trades=%s", series, cut.get("market_settled_ts"), cut.get("trades_created_ts"))

    now = time.time()
    events = client.settled_events(series, int(now - days * 86400))
    events.sort(key=lambda e: e.get("strike_date") or "")
    log.info("%s: %d settled events in the last %.1f days", series, len(events), days)

    window_s = (fcfg["decision_window_min"] + dcfg["lookback_minutes"]) * 60
    t_start = time.time()
    for i, ev in enumerate(events):
        et = ev["event_ticker"]
        out = root / "events" / et
        if (out / "_done").exists():
            continue
        out.mkdir(parents=True, exist_ok=True)
        strike_ts = iso_to_ts(ev.get("strike_date"))
        historical = strike_ts < market_cut
        mk = client.event_markets(et, historical=historical)
        if not mk and not historical:  # moved to the archive since we read the cutoff
            mk = client.event_markets(et, historical=True)
            historical = True
        mdf = pd.DataFrame([market_row(m, series) for m in mk])
        if mdf.empty:
            (out / "_done").write_text("no markets")
            continue
        mdf.to_parquet(out / "markets.parquet", index=False)

        pick = mdf[mdf["result"].isin(["yes", "no"]) & (mdf["volume"] >= dcfg["min_market_volume"])]
        pick = pick.sort_values("volume", ascending=False).head(dcfg["top_strikes_per_event"])

        candles, trades = [], []
        for m in pick.itertuples():
            t1 = int(m.close_ts)
            t0 = int(max(m.open_ts, m.close_ts - window_s))
            candles += [candle_row(c, m.ticker) for c in client.candles(series, m.ticker, t0, t1, historical=historical)]
            if dcfg["download_trades"]:
                trades += [trade_row(t) for t in _fetch_trades(client, m.ticker, t0, t1, trades_cut)]
        cdf = pd.DataFrame(candles)
        if not cdf.empty and cdf["bid_c"].isna().all():
            log.warning("%s: candles have no bid/ask prices (API field names changed?)", et)
        cdf.to_parquet(out / "candles.parquet", index=False)
        tdf = pd.DataFrame(trades)
        if not tdf.empty:
            tdf = tdf.drop_duplicates("trade_id")
        tdf.to_parquet(out / "trades.parquet", index=False)
        (out / "_done").write_text("ok")

        if (i + 1) % 10 == 0 or i == len(events) - 1:
            el = time.time() - t_start
            log.info("%s: %d/%d events (%s, %d strikes, %d trades) %.0fs elapsed",
                     series, i + 1, len(events), et, len(pick), len(tdf), el)

    product = dcfg.get("spot_products", {}).get(series)
    if product and events:
        s = min(iso_to_ts(e["strike_date"]) for e in events) - window_s - 3 * 3600
        e = max(iso_to_ts(e["strike_date"]) for e in events)
        load_spot(data_dir(cfg) / "raw" / "spot", product, s, e)
        log.info("%s: spot %s cached", series, product)
