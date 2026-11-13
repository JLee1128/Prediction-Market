"""Event-driven backtest on the 1-minute grid. Positions are held to settlement unless early exit is on.

Taker: buy at the standing ask (YES) or 1 - bid (NO) at the decision minute, plus optional slippage.
Maker: rest a bid at the current best bid (YES) or best NO bid. It counts as filled only if a later
trade prints THROUGH our price (YES bid at c fills if a trade prints below c; NO bid at c fills if a
YES trade prints above 1 - c) before the order expires. That is conservative about queue position,
and it builds in adverse selection: we only get filled when the market moves against us.
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .fees import fee_per_contract, order_fee


@dataclass
class _Arrays:
    ts: np.ndarray
    bid_h: np.ndarray
    ask_l: np.ndarray
    px_l: np.ndarray
    px_h: np.ndarray
    close_ts: int


def _arrays(g: pd.DataFrame) -> _Arrays:
    g = g.sort_values("ts")
    return _Arrays(g["ts"].to_numpy(), g["bid_h"].to_numpy(dtype=float), g["ask_l"].to_numpy(dtype=float),
                   g["px_l"].to_numpy(dtype=float), g["px_h"].to_numpy(dtype=float), int(g["close_ts"].iloc[0]))


def _window(a: _Arrays, t0: float, t1: float) -> slice:
    return slice(int(np.searchsorted(a.ts, t0, side="right")), int(np.searchsorted(a.ts, t1, side="right")))


def _maker_fill(a: _Arrays, t: float, side: str, price: float, ttl_s: float) -> float | None:
    sl = _window(a, t, min(t + ttl_s, a.close_ts))
    hit = (a.px_l[sl] < price - 1e-9) if side == "yes" else (a.px_h[sl] > 1 - price + 1e-9)
    idx = np.flatnonzero(np.nan_to_num(hit, nan=0).astype(bool))
    return float(a.ts[sl][idx[0]]) if idx.size else None


def _entry_regret(a: _Arrays, t: float, side: str, price: float, window_s: float) -> float:
    """How much cheaper the same side could be bought within `window_s` after our fill (hindsight)."""
    sl = _window(a, t, min(t + window_s, a.close_ts - 60))
    if sl.stop <= sl.start:
        return math.nan
    best = np.nanmin(a.ask_l[sl]) if side == "yes" else 1 - np.nanmax(a.bid_h[sl])
    return float(price - best)


def run_backtest(df: pd.DataFrame, cfg: dict, taker_coef: float, maker_coef: float,
                 threshold: float, mode: str | None = None) -> tuple[pd.DataFrame, dict]:
    s = cfg["strategy"]
    mode = mode or s["mode"]
    bankroll = float(s["bankroll"])
    ttl_s = s["maker_order_ttl_min"] * 60
    regret_s = s["regret_window_min"] * 60
    arrays = {tk: _arrays(g) for tk, g in df.groupby("ticker", sort=False)}
    dec = df[df["p_model"].notna()].sort_values(["ts", "ticker"])

    realized = 0.0
    open_exposure = 0.0
    event_exposure: dict[str, float] = defaultdict(float)
    entries: Counter = Counter()
    busy_until: dict[str, float] = {}
    held: dict[str, dict] = {}          # ticker -> open position
    pending_settle: list[dict] = []
    done: list[dict] = []
    stats = Counter()

    def release(pos: dict, pnl: float, when: float) -> None:
        nonlocal realized, open_exposure
        realized += pnl
        open_exposure -= pos["outlay"]
        event_exposure[pos["event_ticker"]] -= pos["outlay"]
        pos["pnl"], pos["settle_ts"] = pnl, when
        held.pop(pos["ticker"], None)
        done.append(pos)

    def settle_until(t: float) -> None:
        nonlocal pending_settle
        keep = []
        for pos in pending_settle:
            if pos.get("exited"):
                continue
            if pos["close_ts"] <= t:
                won = (pos["side"] == "yes") == (pos["y"] == 1)
                release(pos, pos["count"] * (1.0 if won else 0.0) - pos["count"] * pos["price"] - pos["fee"],
                        pos["close_ts"])
            else:
                keep.append(pos)
        pending_settle = keep

    for r in dec.itertuples(index=False):
        t, tk, p = float(r.ts), r.ticker, float(r.p_model)
        settle_until(t)

        if s["early_exit"] and tk in held:
            pos = held[tk]
            sell = r.bid if pos["side"] == "yes" else 1 - r.ask
            hold_value = p if pos["side"] == "yes" else 1 - p
            if sell - fee_per_contract(sell, taker_coef) > hold_value + s["exit_threshold"]:
                exit_fee = order_fee(sell, pos["count"], taker_coef, cfg["fees"]["round_to"])
                pos.update(exited=True, exit_ts=t, exit_price=sell, exit_fee=exit_fee)
                release(pos, pos["count"] * sell - exit_fee - pos["count"] * pos["price"] - pos["fee"], t)
                stats["early_exits"] += 1
            continue

        if entries[tk] >= s["max_entries_per_market"] or busy_until.get(tk, -1) >= t:
            continue

        if mode == "taker":
            coef = taker_coef
            cands = [("yes", r.ask + s["taker_slippage"], p), ("no", 1 - r.bid + s["taker_slippage"], 1 - p)]
        else:
            coef = maker_coef
            cands = [("yes", r.bid, p), ("no", 1 - r.ask, 1 - p)]
        side, price, q = max(cands, key=lambda c: c[2] - c[1] - fee_per_contract(c[1], coef))
        if not (0 < price < 1):
            continue
        edge = q - price - fee_per_contract(price, coef)
        if edge <= threshold:
            continue

        equity = bankroll + realized
        limit = min(s["max_per_market"], s["max_per_event"] - event_exposure[r.event_ticker],
                    s["max_open_exposure"] - open_exposure, equity - open_exposure)
        stake = min(s["kelly_fraction"] * (edge / (1 - price)) * equity, limit)
        count = math.floor(stake / (price + fee_per_contract(price, coef)))
        while count > 0 and count * price + order_fee(price, count, coef, cfg["fees"]["round_to"]) > limit + 1e-9:
            count -= 1  # caps include fees
        if count < 1:
            stats["skipped_size"] += 1
            continue

        fill_t = t
        if mode == "maker":
            stats["orders"] += 1
            f = _maker_fill(arrays[tk], t, side, price, ttl_s)
            if f is None:
                busy_until[tk] = t + ttl_s
                continue
            fill_t = f
            stats["fills"] += 1

        fee = order_fee(price, count, coef, cfg["fees"]["round_to"])
        pos = {
            "ticker": tk, "event_ticker": r.event_ticker, "series": r.series, "mode": mode, "side": side,
            "signal_ts": t, "fill_ts": fill_t, "price": price, "count": count, "fee": fee,
            "p_model": p, "p_market": float(r.mid), "edge": edge, "tte_min": float(r.tte_min),
            "close_ts": int(r.close_ts), "y": int(r.y), "outlay": count * price + fee,
            "exit_ts": np.nan, "exit_price": np.nan, "exit_fee": 0.0,
            "entry_regret": _entry_regret(arrays[tk], fill_t, side, price, regret_s),
        }
        open_exposure += pos["outlay"]
        event_exposure[r.event_ticker] += pos["outlay"]
        entries[tk] += 1
        held[tk] = pos
        busy_until[tk] = r.close_ts
        pending_settle.append(pos)

    settle_until(float("inf"))
    trades = pd.DataFrame(done)
    if not trades.empty:
        trades = trades.drop(columns=["exited"], errors="ignore").sort_values("settle_ts").reset_index(drop=True)
    if mode == "maker":
        stats["fill_rate"] = stats["fills"] / stats["orders"] if stats["orders"] else float("nan")
    return trades, dict(stats)


def tune_threshold(df_valid: pd.DataFrame, cfg: dict, taker_coef: float, maker_coef: float) -> tuple[float, pd.DataFrame]:
    """Pick the edge threshold on the VALIDATION period only."""
    s = cfg["strategy"]
    rows = []
    for th in s["threshold_grid"]:
        tr, _ = run_backtest(df_valid, cfg, taker_coef, maker_coef, th)
        rows.append({"threshold": th, "trades": len(tr), "net_pnl": float(tr["pnl"].sum()) if len(tr) else 0.0,
                     "events": int(tr["event_ticker"].nunique()) if len(tr) else 0})
    grid = pd.DataFrame(rows)
    ok = grid[grid["trades"] >= s["min_valid_trades"]]
    pool = ok if not ok.empty else grid
    best = pool.sort_values(["net_pnl", "threshold"], ascending=[False, False]).iloc[0]
    return float(best["threshold"]), grid
