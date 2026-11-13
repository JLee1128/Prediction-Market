import gzip
import json

import numpy as np
import pandas as pd
import pytest

from predmkt.backtest import longshot_baseline, run_backtest
from predmkt.book import Book, book_minute_features, book_states, ofi_increment
from predmkt.config import load_config
from predmkt.fees import fee_coefs, order_fee
from predmkt.metrics import brier_diff_ci, calibration_table, max_drawdown
from predmkt.model import event_weights, split_events


@pytest.fixture
def cfg():
    return load_config()


# ---- fees ----------------------------------------------------------------------------------

def test_fee_formula():
    assert order_fee(0.50, 100, 0.07) == pytest.approx(1.75)
    assert order_fee(0.50, 1, 0.07) == pytest.approx(0.02)   # 1.75c rounds up to 2c
    assert order_fee(0.90, 10, 0.07) == pytest.approx(0.07)  # 0.63c * 10 = 6.3c -> 7c
    assert order_fee(0.50, 100, 0.0) == 0.0


def test_fee_types(cfg):
    assert fee_coefs({"fee_type": "quadratic", "fee_multiplier": 1}, cfg) == (0.07, 0.0)
    t, m = fee_coefs({"fee_type": "quadratic_with_maker_fees", "fee_multiplier": 1}, cfg)
    assert (t, m) == (pytest.approx(0.07), pytest.approx(0.0175))


# ---- order book ----------------------------------------------------------------------------

def test_book_snapshot_and_delta():
    b = Book()
    b.snapshot([["0.40", "100"], ["0.41", "50"]], [["0.55", "30"], ["0.57", "20"]])
    st = b.state()
    assert st["bid"] == pytest.approx(0.41) and st["bid_sz"] == 50
    assert st["ask"] == pytest.approx(0.43) and st["ask_sz"] == 20  # 1 - best NO bid 0.57
    b.delta("yes", "0.41", "-50")  # best bid level removed
    assert b.state()["bid"] == pytest.approx(0.40)
    b.delta("no", "0.58", "10")    # new better NO bid -> lower YES ask
    assert b.state()["ask"] == pytest.approx(0.42)


def test_ofi_sign():
    prev = {"bid": 0.40, "ask": 0.42, "bid_sz": 100, "ask_sz": 100}
    more_bids = {**prev, "bid_sz": 150}
    fewer_asks = {**prev, "ask_sz": 40}
    assert ofi_increment(prev, more_bids) == 50
    assert ofi_increment(prev, fewer_asks) == 60
    assert ofi_increment(prev, {**prev, "bid": 0.39, "bid_sz": 80}) == -100


def test_book_states_ws_and_gap(tmp_path):
    msgs = [
        {"type": "orderbook_snapshot", "sid": 1, "seq": 1, "msg": {"market_ticker": "M", "ts_ms": 1000,
         "yes_dollars_fp": [["0.40", "100"]], "no_dollars_fp": [["0.58", "100"]]}},
        {"type": "orderbook_delta", "sid": 1, "seq": 2, "msg": {"market_ticker": "M", "price_dollars": "0.40",
         "delta_fp": "50", "side": "yes", "ts_ms": 2000}},
        {"type": "orderbook_delta", "sid": 1, "seq": 4, "msg": {"market_ticker": "M", "price_dollars": "0.40",
         "delta_fp": "50", "side": "yes", "ts_ms": 3000}},   # gap: seq 3 missing -> ignored
    ]
    f = tmp_path / "00_1.jsonl.gz"
    with gzip.open(f, "wt") as fh:
        for m in msgs:
            fh.write(json.dumps({"src": "ws", "t_recv": 0, "raw": m}) + "\n")
    st = book_states([f])
    assert list(st["bid_sz"]) == [100, 150]
    assert st["ofi"].iloc[1] == 50
    feats = book_minute_features(st)
    assert feats["qi_l1"].iloc[0] == pytest.approx((150 - 100) / 250)


# ---- metrics -------------------------------------------------------------------------------

def test_max_drawdown():
    pnl = pd.Series([10.0, -30.0, 5.0, -10.0, 50.0], index=[60, 120, 180, 240, 300])
    dd = max_drawdown(pnl, 100)
    assert dd["max_dd"] == pytest.approx(35.0)   # peak 110 -> trough 75
    assert dd["peak_ts"] == "1970-01-01 00:01" and dd["trough_ts"] == "1970-01-01 00:04"


def test_event_weighting():
    # Event A has 9 snapshots, event B has 1: each event must count equally.
    df = pd.DataFrame({"event_ticker": ["A"] * 9 + ["B"], "ts": range(10), "y": [1] * 9 + [0],
                       "p": [0.9] * 9 + [0.9], "q": [0.5] * 10})
    w = event_weights(df)
    assert w[:9].sum() == pytest.approx(w[9])
    cal = calibration_table(df, "p", 10)
    assert cal["realized"].iloc[0] == pytest.approx(0.5)
    d = brier_diff_ci(df, "p", "q", 200, 0)
    assert d["events"] == 2


# ---- split ---------------------------------------------------------------------------------

def test_split_is_chronological_and_disjoint(cfg):
    rows = []
    for e in range(50):
        close = 10_000 + e * 3600
        for k in range(5):
            rows.append({"event_ticker": f"E{e:02d}", "ts": close - 3000 + k * 60, "close_ts": close, "in_window": True})
    df = pd.DataFrame(rows)
    s = split_events(df, cfg)
    tr, va, te = (set(df.loc[s[k], "event_ticker"]) for k in ("train", "valid", "test"))
    assert not (tr & va or tr & te or va & te)
    assert max(df.loc[s["train"], "close_ts"]) < min(df.loc[s["valid"], "ts"])
    assert max(df.loc[s["valid"], "close_ts"]) < min(df.loc[s["test"], "ts"])


# ---- backtest ------------------------------------------------------------------------------

def _market(ticker, event, y, p_model, bid=0.40, ask=0.42, n=10, close=10_000, px_l=None):
    ts = close - 60 * np.arange(n, 0, -1)
    return pd.DataFrame({
        "ticker": ticker, "event_ticker": event, "series": "S", "ts": ts, "close_ts": close, "y": y,
        "bid": bid, "ask": ask, "mid": (bid + ask) / 2, "bid_h": bid, "bid_l": bid, "ask_h": ask, "ask_l": ask,
        "px_h": ask, "px_l": bid if px_l is None else px_l, "tte_min": (close - ts) / 60, "p_model": p_model,
    })


def test_taker_pnl_and_caps(cfg):
    df = _market("M1", "E1", y=1, p_model=0.70)
    trades, _ = run_backtest(df, cfg, 0.07, 0.0, threshold=0.02, mode="taker")
    assert len(trades) == 1                                # one entry per market
    t = trades.iloc[0]
    assert t["side"] == "yes" and t["price"] == pytest.approx(0.42)
    assert t["count"] * t["price"] <= cfg["strategy"]["max_per_market"] + 1e-9
    fee = order_fee(0.42, t["count"], 0.07)
    assert t["pnl"] == pytest.approx(t["count"] * (1 - 0.42) - fee)


def test_taker_buys_no_when_model_low(cfg):
    df = _market("M1", "E1", y=0, p_model=0.20)
    trades, _ = run_backtest(df, cfg, 0.07, 0.0, threshold=0.02, mode="taker")
    t = trades.iloc[0]
    assert t["side"] == "no" and t["price"] == pytest.approx(0.60) and t["pnl"] > 0


def test_event_cap(cfg):
    df = pd.concat([_market(f"M{i}", "E1", y=1, p_model=0.9) for i in range(6)])
    trades, _ = run_backtest(df, cfg, 0.07, 0.0, threshold=0.02, mode="taker")
    outlay = (trades["count"] * trades["price"] + trades["fee"]).sum()
    assert outlay <= cfg["strategy"]["max_per_event"] + 1e-9


def test_maker_needs_trade_through(cfg):
    no_fill = _market("M1", "E1", y=1, p_model=0.70)                 # trades never print below our bid
    trades, stats = run_backtest(no_fill, cfg, 0.07, 0.0, threshold=0.02, mode="maker")
    assert trades.empty and stats["orders"] >= 1
    fill = _market("M1", "E1", y=1, p_model=0.70, px_l=0.39)          # a trade prints through 0.40
    trades, stats = run_backtest(fill, cfg, 0.07, 0.0, threshold=0.02, mode="maker")
    assert len(trades) == 1 and trades.iloc[0]["price"] == pytest.approx(0.40) and trades.iloc[0]["fee"] == 0


def test_candle_row_live_and_archive_formats():
    from predmkt.download import candle_row
    live = {"end_period_ts": 120, "yes_bid": {"close_dollars": "0.4100"}, "yes_ask": {"close_dollars": "0.4300"},
            "price": {"mean_dollars": "0.4200"}, "volume_fp": "10.00", "open_interest_fp": "5.00"}
    arch = {"end_period_ts": 120, "yes_bid": {"close": "0.4100"}, "yes_ask": {"close": "0.4300"},
            "price": {"mean": "0.4200"}, "volume": "10.00", "open_interest": "5.00"}
    for c in (live, arch):
        r = candle_row(c, "T")
        assert (r["bid_c"], r["ask_c"], r["px_mean"], r["volume"], r["oi"]) == (0.41, 0.43, 0.42, 10.0, 5.0)


def test_compound_flag(cfg):
    s = cfg["strategy"]
    s.update(max_per_market=1e6, max_per_event=1e6, max_open_exposure=1e6)
    df = pd.concat([_market("M1", "E1", y=1, p_model=0.9, close=10_000),     # wins, then a later event
                    _market("M2", "E2", y=1, p_model=0.9, close=20_000)])
    stakes = {}
    for flag in (False, True):
        s["compound"] = flag
        tr, _ = run_backtest(df, cfg, 0.07, 0.0, threshold=0.02, mode="taker")
        stakes[flag] = tr.set_index("ticker")["outlay"]
    assert stakes[False]["M2"] == pytest.approx(stakes[False]["M1"], rel=0.01)   # profits not reinvested
    assert stakes[True]["M2"] > stakes[True]["M1"] * 1.1                         # profits reinvested


def test_longshot_baseline(cfg):
    df = pd.concat([
        _market("CHEAP_YES", "E1", y=1, p_model=0.5, bid=0.04, ask=0.05),
        _market("CHEAP_NO", "E2", y=0, p_model=0.5, bid=0.95, ask=0.96),
        _market("MID", "E3", y=1, p_model=0.5),                       # 40/42c: not a long shot
    ])
    tr, _ = longshot_baseline(df, cfg, 0.07, 0.10)
    got = tr.set_index("ticker")
    assert set(got.index) == {"CHEAP_YES", "CHEAP_NO"}
    assert got.loc["CHEAP_YES", "side"] == "yes" and got.loc["CHEAP_YES", "price"] == pytest.approx(0.05)
    assert got.loc["CHEAP_NO", "side"] == "no" and got.loc["CHEAP_NO", "price"] == pytest.approx(0.05)
    assert (got["signal_ts"] == df["ts"].min()).all()                 # first decision minute only
    assert (got["outlay"] <= cfg["strategy"]["max_per_market"] + 1e-9).all()
