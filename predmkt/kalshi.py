"""Thin Kalshi REST client for public market data (no auth needed) with throttling and retries."""
from __future__ import annotations

import logging
import time
from typing import Iterator
from urllib.parse import urlparse

import requests

from .auth import KalshiSigner

log = logging.getLogger(__name__)


class KalshiClient:
    def __init__(self, base_url: str, max_rps: float = 8.0, signer: KalshiSigner | None = None, timeout: float = 20.0):
        self.base_url = base_url.rstrip("/")
        self.path_prefix = urlparse(self.base_url).path  # e.g. /trade-api/v2 (part of the signed path)
        self.session = requests.Session()
        self.session.headers["User-Agent"] = "predmkt-xgb/0.1"
        self.signer = signer
        self.timeout = timeout
        self._min_dt = 1.0 / max_rps
        self._last = 0.0

    @classmethod
    def from_config(cls, cfg: dict, signer: KalshiSigner | None = None) -> "KalshiClient":
        return cls(cfg["api"]["base_url"], cfg["api"]["max_requests_per_sec"], signer)

    def _throttle(self) -> None:
        wait = self._min_dt - (time.monotonic() - self._last)
        if wait > 0:
            time.sleep(wait)
        self._last = time.monotonic()

    def get(self, path: str, params=None, auth: bool = False, retries: int = 6) -> dict | None:
        """GET with backoff on 429/5xx/network errors. Returns None on 404."""
        url = self.base_url + path
        for attempt in range(retries):
            self._throttle()
            headers = self.signer.headers("GET", self.path_prefix + path) if (auth and self.signer) else {}
            try:
                r = self.session.get(url, params=params, headers=headers, timeout=self.timeout)
            except requests.RequestException as e:
                log.warning("GET %s failed (%s); retry %d", path, e, attempt + 1)
                time.sleep(min(30, 2**attempt))
                continue
            if r.status_code == 429 or r.status_code >= 500:
                log.warning("GET %s -> %d; retry %d", path, r.status_code, attempt + 1)
                time.sleep(min(30, 2**attempt))
                continue
            if r.status_code == 404:
                return None
            r.raise_for_status()
            return r.json()
        raise RuntimeError(f"GET {path} failed after {retries} attempts")

    def paginate(self, path: str, key: str, params: dict | None = None, limit: int = 1000) -> Iterator[dict]:
        params = dict(params or {})
        params["limit"] = limit
        while True:
            d = self.get(path, params)
            if not d:
                return
            yield from d.get(key) or []
            cursor = d.get("cursor")
            if not cursor:
                return
            params["cursor"] = cursor

    # ---- endpoints -------------------------------------------------------------------------
    def series(self, series_ticker: str) -> dict | None:
        d = self.get(f"/series/{series_ticker}")
        return d["series"] if d else None

    def historical_cutoff(self) -> dict:
        return self.get("/historical/cutoff") or {}

    def settled_events(self, series_ticker: str, min_close_ts: int) -> list[dict]:
        params = {"series_ticker": series_ticker, "status": "settled", "min_close_ts": int(min_close_ts)}
        return list(self.paginate("/events", "events", params, limit=200))

    def event_markets(self, event_ticker: str, historical: bool = False) -> list[dict]:
        path = "/historical/markets" if historical else "/markets"
        return list(self.paginate(path, "markets", {"event_ticker": event_ticker}))

    def open_markets(self, series_ticker: str) -> list[dict]:
        return list(self.paginate("/markets", "markets", {"series_ticker": series_ticker, "status": "open"}))

    def candles(self, series_ticker: str, ticker: str, start_ts: int, end_ts: int,
                historical: bool = False, period_min: int = 1) -> list[dict]:
        path = (f"/historical/markets/{ticker}/candlesticks" if historical
                else f"/series/{series_ticker}/markets/{ticker}/candlesticks")
        d = self.get(path, {"start_ts": int(start_ts), "end_ts": int(end_ts), "period_interval": period_min})
        return (d or {}).get("candlesticks") or []

    def trades(self, ticker: str, min_ts: int, max_ts: int, historical: bool = False) -> list[dict]:
        path = "/historical/trades" if historical else "/markets/trades"
        return list(self.paginate(path, "trades", {"ticker": ticker, "min_ts": int(min_ts), "max_ts": int(max_ts)}))

    def orderbooks(self, tickers: list[str]) -> list[dict]:
        """Full-depth books for several markets in one request (YES bids and NO bids only)."""
        out: list[dict] = []
        for i in range(0, len(tickers), 100):
            d = self.get("/markets/orderbooks", [("tickers", t) for t in tickers[i:i + 100]])
            out.extend((d or {}).get("orderbooks") or [])
        return out
