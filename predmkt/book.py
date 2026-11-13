"""Order-book reconstruction from recorder files and order-book features.

Kalshi books list YES bids and NO bids only. A NO bid at price q is a YES ask at 1 - q,
so: best YES ask = 1 - best NO bid.

Recorder lines are JSON objects:
  poll mode: {"src": "poll", "t_recv": <sec>, "ticker": ..., "yes": [[price, size], ...], "no": [...]}
  ws mode:   {"src": "ws",   "t_recv": <sec>, "raw": <websocket message>}
"""
from __future__ import annotations

import gzip
import json
import logging
import math
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)
SCALE = 10_000  # prices stored as integer 1/10000 dollars to avoid float dict keys


def _px(p) -> int:
    return int(round(float(p) * SCALE))


class Book:
    __slots__ = ("yes", "no", "valid")

    def __init__(self) -> None:
        self.yes: dict[int, float] = {}
        self.no: dict[int, float] = {}
        self.valid = False

    def snapshot(self, yes_levels, no_levels) -> None:
        self.yes = {_px(p): float(s) for p, s in (yes_levels or []) if float(s) > 0}
        self.no = {_px(p): float(s) for p, s in (no_levels or []) if float(s) > 0}
        self.valid = True

    def delta(self, side: str, price, delta) -> None:
        levels = self.yes if side == "yes" else self.no
        k = _px(price)
        v = levels.get(k, 0.0) + float(delta)
        if v > 1e-9:
            levels[k] = v
        else:
            levels.pop(k, None)

    def state(self, within: float = 0.03) -> dict | None:
        """Top-of-book + depth within `within` dollars of the best price on each side."""
        if not self.valid or not self.yes or not self.no:
            return None
        bb = max(self.yes)
        bn = max(self.no)
        w = _px(within)
        return {
            "bid": bb / SCALE,
            "ask": (SCALE - bn) / SCALE,
            "bid_sz": self.yes[bb],
            "ask_sz": self.no[bn],
            "dep_bid": sum(s for p, s in self.yes.items() if p >= bb - w),
            "dep_ask": sum(s for p, s in self.no.items() if p >= bn - w),
        }


def iter_lines(files: Iterable[Path]) -> Iterator[dict]:
    """Read .jsonl / .jsonl.gz files, tolerating a truncated last line from a crash."""
    for f in sorted(files):
        opener = gzip.open if str(f).endswith(".gz") else open
        try:
            with opener(f, "rt") as fh:
                for line in fh:
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        continue
        except (EOFError, OSError) as e:
            log.warning("stopped reading %s early: %s", f, e)


def ofi_increment(prev: dict, cur: dict) -> float:
    """Order-flow imbalance between two consecutive top-of-book states (Cont, Kukanov & Stoikov 2014)."""
    e = 0.0
    if cur["bid"] >= prev["bid"]:
        e += cur["bid_sz"]
    if cur["bid"] <= prev["bid"]:
        e -= prev["bid_sz"]
    if cur["ask"] <= prev["ask"]:
        e -= cur["ask_sz"]
    if cur["ask"] >= prev["ask"]:
        e += prev["ask_sz"]
    return e


def book_states(files: Iterable[Path]) -> pd.DataFrame:
    """Replay recorder files into one row per (ticker, second): last top-of-book state + summed OFI."""
    books: dict[str, Book] = {}
    prev_state: dict[str, dict] = {}
    seqs: dict[int, int] = {}
    rows: list[tuple] = []

    def emit(ticker: str, t: float) -> None:
        st = books[ticker].state()
        if st is None:
            return
        p = prev_state.get(ticker)
        ofi = ofi_increment(p, st) if p is not None else 0.0
        prev_state[ticker] = st
        rows.append((ticker, t, st["bid"], st["ask"], st["bid_sz"], st["ask_sz"], st["dep_bid"], st["dep_ask"], ofi))

    for rec in iter_lines(files):
        if rec.get("src") == "poll":
            tk = rec["ticker"]
            books.setdefault(tk, Book()).snapshot(rec.get("yes"), rec.get("no"))
            emit(tk, rec["t_recv"])
            continue
        msg = rec.get("raw") or {}
        typ, body = msg.get("type"), msg.get("msg") or {}
        sid, seq = msg.get("sid"), msg.get("seq")
        if sid is not None and seq is not None and typ in ("orderbook_snapshot", "orderbook_delta"):
            last = seqs.get(sid)
            if last is not None and seq != last + 1 and typ == "orderbook_delta":
                # Missed messages: books on this connection are unreliable until fresh snapshots arrive.
                for b in books.values():
                    b.valid = False
                prev_state.clear()
            seqs[sid] = seq
        tk = body.get("market_ticker")
        if not tk:
            continue
        t = (body.get("ts_ms") or msg.get("sending_ts_ms") or rec["t_recv"] * 1000) / 1000.0
        if typ == "orderbook_snapshot":
            books.setdefault(tk, Book()).snapshot(body.get("yes_dollars_fp"), body.get("no_dollars_fp"))
            prev_state.pop(tk, None)
            emit(tk, t)
        elif typ == "orderbook_delta" and tk in books and books[tk].valid:
            books[tk].delta(body["side"], body["price_dollars"], body["delta_fp"])
            emit(tk, t)

    cols = ["ticker", "t", "bid", "ask", "bid_sz", "ask_sz", "dep_bid", "dep_ask", "ofi"]
    df = pd.DataFrame(rows, columns=cols)
    if df.empty:
        return df
    df["sec"] = np.ceil(df["t"]).astype("int64")
    agg = df.groupby(["ticker", "sec"], sort=True).agg(
        bid=("bid", "last"), ask=("ask", "last"), bid_sz=("bid_sz", "last"), ask_sz=("ask_sz", "last"),
        dep_bid=("dep_bid", "last"), dep_ask=("dep_ask", "last"), ofi=("ofi", "sum"))
    return agg.reset_index()


BOOK_FEATURES = ["qi_l1", "micro_minus_mid", "depth_imb_3c", "ofi_norm_1", "ofi_norm_5", "book_spread"]


def book_minute_features(states: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-second book states to the 1-minute decision grid (value at minute end t uses data <= t)."""
    if states.empty:
        return pd.DataFrame(columns=["ticker", "ts", *BOOK_FEATURES])
    s = states.copy()
    s["ts"] = (np.ceil(s["sec"] / 60) * 60).astype("int64")
    g = s.groupby(["ticker", "ts"], sort=True).agg(
        bid=("bid", "last"), ask=("ask", "last"), bid_sz=("bid_sz", "last"), ask_sz=("ask_sz", "last"),
        dep_bid=("dep_bid", "last"), dep_ask=("dep_ask", "last"), ofi=("ofi", "sum"),
        l1_depth=("bid_sz", "mean"), l1_depth_a=("ask_sz", "mean")).reset_index()
    tot = g["bid_sz"] + g["ask_sz"]
    mid = (g["bid"] + g["ask"]) / 2
    micro = (g["bid"] * g["ask_sz"] + g["ask"] * g["bid_sz"]) / tot
    g["qi_l1"] = (g["bid_sz"] - g["ask_sz"]) / tot
    g["micro_minus_mid"] = (micro - mid) * 100  # cents
    g["depth_imb_3c"] = (g["dep_bid"] - g["dep_ask"]) / (g["dep_bid"] + g["dep_ask"])
    g["book_spread"] = (g["ask"] - g["bid"]) * 100
    depth = (g["l1_depth"] + g["l1_depth_a"]) / 2
    out = []
    for _, d in g.groupby("ticker", sort=False):
        d = d.copy()
        norm = depth.loc[d.index].rolling(5, min_periods=1).mean().replace(0, math.nan)
        d["ofi_norm_1"] = d["ofi"] / norm
        d["ofi_norm_5"] = d["ofi"].rolling(5, min_periods=1).sum() / norm
        out.append(d)
    g = pd.concat(out)
    return g[["ticker", "ts", *BOOK_FEATURES]].reset_index(drop=True)


def build_book_features(raw_dir: Path) -> pd.DataFrame:
    files = sorted(list(raw_dir.rglob("*.jsonl")) + list(raw_dir.rglob("*.jsonl.gz")))
    if not files:
        return pd.DataFrame(columns=["ticker", "ts", *BOOK_FEATURES])
    log.info("replaying %d recorder files", len(files))
    return book_minute_features(book_states(files))
