"""Live order-book recorder. History has no order books, so this must run continuously to collect them.

poll mode: one REST request per second for all selected markets (no API key needed).
ws mode:   websocket orderbook_delta stream with every change (needs an API key; see README).

Files: data/raw/live/<YYYYMMDD>/<HH>.jsonl, gzipped when the hour rolls over.
"""
from __future__ import annotations

import asyncio
import datetime as dt
import gzip
import json
import logging
import shutil
import time
from pathlib import Path

from .auth import KalshiSigner
from .config import data_dir, iso_to_ts
from .kalshi import KalshiClient

log = logging.getLogger(__name__)


class RawWriter:
    """Append JSON lines to one file per UTC hour (named <HH>_<start unix>.jsonl so names sort
    chronologically even across restarts); gzip each file when it is closed."""

    def __init__(self, root: Path):
        self.root = root
        self.hour_key: str | None = None
        self.path: Path | None = None
        self.fh = None

    def write(self, rec: dict) -> None:
        d = dt.datetime.fromtimestamp(rec["t_recv"], dt.timezone.utc)
        key = d.strftime("%Y%m%d%H")
        if key != self.hour_key:
            self.close()
            self.path = self.root / d.strftime("%Y%m%d") / f"{d.strftime('%H')}_{int(rec['t_recv'])}.jsonl"
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.fh, self.hour_key = open(self.path, "a"), key
        self.fh.write(json.dumps(rec, separators=(",", ":")) + "\n")

    def flush(self) -> None:
        if self.fh:
            self.fh.flush()

    def close(self) -> None:
        if self.fh:
            self.fh.close()
            with open(self.path, "rb") as src, gzip.open(self.path.with_suffix(".jsonl.gz"), "wb") as dst:
                shutil.copyfileobj(src, dst)
            self.path.unlink()
        self.fh, self.path, self.hour_key = None, None, None


def select_markets(client: KalshiClient, series_list: list[str], events_ahead: int, strikes_per_event: int) -> list[str]:
    """Next `events_ahead` events per series; in each, the strikes whose mid is closest to 50c."""
    tickers: list[str] = []
    now = time.time()
    for s in series_list:
        by_event: dict[str, list[dict]] = {}
        for m in client.open_markets(s):
            if iso_to_ts(m.get("close_time")) > now:
                by_event.setdefault(m["event_ticker"], []).append(m)
        upcoming = sorted(by_event.items(), key=lambda kv: min(iso_to_ts(m["close_time"]) for m in kv[1]))
        for _, ms in upcoming[:events_ahead]:
            def dist(m: dict) -> float:
                try:
                    mid = (float(m["yes_bid_dollars"]) + float(m["yes_ask_dollars"])) / 2
                except (KeyError, TypeError, ValueError):
                    return 1.0
                return abs(mid - 0.5) - 1e-9 * float(m.get("volume_fp") or 0)
            tickers += [m["ticker"] for m in sorted(ms, key=dist)[:strikes_per_event]]
    return tickers


def run_poll(cfg: dict, client: KalshiClient, duration_s: float | None = None) -> None:
    rc = cfg["recorder"]
    writer = RawWriter(data_dir(cfg) / "raw" / "live")
    tickers: list[str] = []
    next_refresh = 0.0
    t_end = time.time() + duration_s if duration_s else float("inf")
    n = 0
    try:
        while time.time() < t_end:
            t0 = time.time()
            if t0 >= next_refresh:
                tickers = select_markets(client, rc["series"], rc["events_ahead"], rc["strikes_per_event"])
                next_refresh = t0 + rc["refresh_markets_sec"]
                log.info("recording %d markets: %s ...", len(tickers), ", ".join(tickers[:4]))
            if tickers:
                for ob in client.orderbooks(tickers):
                    book = ob.get("orderbook_fp") or {}
                    writer.write({"src": "poll", "t_recv": time.time(), "ticker": ob.get("ticker"),
                                  "yes": book.get("yes_dollars") or [], "no": book.get("no_dollars") or []})
                n += 1
                if n % 60 == 0:
                    writer.flush()
            time.sleep(max(0.0, rc["poll_interval_sec"] - (time.time() - t0)))
    finally:
        writer.close()


async def _run_ws_once(cfg: dict, client: KalshiClient, signer: KalshiSigner, writer: RawWriter, t_end: float) -> None:
    import websockets

    rc = cfg["recorder"]
    url = cfg["api"]["ws_url"]
    headers = signer.headers("GET", "/trade-api/ws/v2")
    async with websockets.connect(url, additional_headers=headers, max_size=2**24) as ws:
        tickers = await asyncio.to_thread(select_markets, client, rc["series"], rc["events_ahead"], rc["strikes_per_event"])
        msg_id = 1
        await ws.send(json.dumps({"id": msg_id, "cmd": "subscribe",
                                  "params": {"channels": ["orderbook_delta"], "market_tickers": tickers}}))
        log.info("ws subscribed to %d markets", len(tickers))
        sid: int | None = None
        last_seq: int | None = None
        next_refresh = time.time() + rc["refresh_markets_sec"]
        while time.time() < t_end:
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=5)
            except asyncio.TimeoutError:
                raw = None
            if raw is not None:
                msg = json.loads(raw)
                writer.write({"src": "ws", "t_recv": time.time(), "raw": msg})
                typ = msg.get("type")
                if typ == "subscribed":
                    sid = (msg.get("msg") or {}).get("sid", sid)
                elif typ == "error":
                    log.error("ws error: %s", msg)
                elif typ in ("orderbook_snapshot", "orderbook_delta"):
                    seq = msg.get("seq")
                    if last_seq is not None and seq is not None and seq != last_seq + 1 and sid is not None:
                        log.warning("seq gap %s -> %s; requesting fresh snapshots", last_seq, seq)
                        msg_id += 1
                        await ws.send(json.dumps({"id": msg_id, "cmd": "update_subscription",
                                                  "params": {"sids": [sid], "market_tickers": tickers,
                                                             "action": "get_snapshot"}}))
                    last_seq = seq if seq is not None else last_seq
            if time.time() >= next_refresh and sid is not None:
                new = await asyncio.to_thread(select_markets, client, rc["series"], rc["events_ahead"], rc["strikes_per_event"])
                add, drop = sorted(set(new) - set(tickers)), sorted(set(tickers) - set(new))
                for action, mk in (("add_markets", add), ("delete_markets", drop)):
                    if mk:
                        msg_id += 1
                        await ws.send(json.dumps({"id": msg_id, "cmd": "update_subscription",
                                                  "params": {"sids": [sid], "market_tickers": mk, "action": action}}))
                if add or drop:
                    log.info("ws markets +%d -%d", len(add), len(drop))
                tickers = new
                next_refresh = time.time() + rc["refresh_markets_sec"]
                writer.flush()


def run_ws(cfg: dict, client: KalshiClient, signer: KalshiSigner, duration_s: float | None = None) -> None:
    writer = RawWriter(data_dir(cfg) / "raw" / "live")
    t_end = time.time() + duration_s if duration_s else float("inf")
    backoff = 1
    try:
        while time.time() < t_end:
            try:
                asyncio.run(_run_ws_once(cfg, client, signer, writer, t_end))
                backoff = 1
            except Exception as e:  # reconnect on any network/protocol error
                log.warning("ws disconnected (%s); reconnecting in %ds", e, backoff)
                time.sleep(backoff)
                backoff = min(60, backoff * 2)
    finally:
        writer.close()
