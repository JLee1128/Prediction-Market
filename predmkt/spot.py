"""Underlying spot prices (Coinbase public 1-minute candles), cached per UTC day."""
from __future__ import annotations

import datetime as dt
import logging
import time
from pathlib import Path

import pandas as pd
import requests

log = logging.getLogger(__name__)
COINBASE = "https://api.exchange.coinbase.com"
MAX_CANDLES = 300


def _fetch(session: requests.Session, product: str, start: int, end: int) -> list[list]:
    params = {
        "granularity": 60,
        "start": dt.datetime.fromtimestamp(start, dt.timezone.utc).isoformat(),
        "end": dt.datetime.fromtimestamp(end, dt.timezone.utc).isoformat(),
    }
    for attempt in range(6):
        try:
            r = session.get(f"{COINBASE}/products/{product}/candles", params=params, timeout=20)
            if r.status_code == 429 or r.status_code >= 500:
                raise requests.HTTPError(str(r.status_code))
            if r.status_code >= 400:
                raise RuntimeError(f"coinbase {product} {params}: {r.status_code} {r.text[:200]}")
            return r.json()
        except requests.RequestException as e:
            log.warning("coinbase %s retry %d (%s)", product, attempt + 1, e)
            time.sleep(min(30, 2**attempt))
    raise RuntimeError(f"coinbase candles failed for {product} {start}-{end}")


def fetch_minutes(product: str, start_ts: int, end_ts: int) -> pd.DataFrame:
    """1-minute candles covering [start_ts, end_ts). `ts` is the minute END (matches Kalshi's end_period_ts)."""
    session = requests.Session()
    rows: list[list] = []
    end_ts = min(end_ts, int(time.time() // 60 * 60))  # Coinbase rejects ranges in the future
    t = start_ts
    while t < end_ts:
        t2 = min(end_ts, t + 60 * MAX_CANDLES)
        rows.extend(_fetch(session, product, t, t2 - 60))  # Coinbase 'end' is inclusive of that bucket start
        t = t2
        time.sleep(0.15)
    df = pd.DataFrame(rows, columns=["start", "low", "high", "open", "close", "volume"])
    if df.empty:
        return pd.DataFrame(columns=["ts", "open", "high", "low", "close", "volume"])
    df["ts"] = df["start"].astype("int64") + 60
    df = df.drop(columns="start").drop_duplicates("ts").sort_values("ts")
    return df[(df["ts"] > start_ts) & (df["ts"] <= end_ts)].reset_index(drop=True)


def load_spot(cache_dir: Path, product: str, start_ts: float, end_ts: float) -> pd.DataFrame:
    """Return minute spot candles for [start_ts, end_ts], downloading and caching whole UTC days."""
    cache = cache_dir / product
    cache.mkdir(parents=True, exist_ok=True)
    day0 = dt.datetime.fromtimestamp(start_ts, dt.timezone.utc).date()
    day1 = dt.datetime.fromtimestamp(end_ts, dt.timezone.utc).date()
    today = dt.datetime.now(dt.timezone.utc).date()
    frames = []
    d = day0
    while d <= day1:
        f = cache / f"{d.isoformat()}.parquet"
        if f.exists():
            frames.append(pd.read_parquet(f))
        else:
            s = int(dt.datetime(d.year, d.month, d.day, tzinfo=dt.timezone.utc).timestamp())
            df = fetch_minutes(product, s, s + 86400)
            if d < today:  # only cache complete days
                df.to_parquet(f, index=False)
            frames.append(df)
        d += dt.timedelta(days=1)
    out = pd.concat(frames, ignore_index=True).drop_duplicates("ts").sort_values("ts")
    return out[(out["ts"] >= start_ts) & (out["ts"] <= end_ts)].reset_index(drop=True)
