"""Build the 1-minute decision grid with features and labels.

Timing rule (no look-ahead): the row at minute-end `ts` only uses candles whose period ends at or
before `ts` and trades executed at or before `ts`. Decisions made at `ts` trade at the quotes
standing at `ts` (the close of that minute's candle).
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import ndtr

from .book import BOOK_FEATURES, build_book_features
from .config import data_dir
from .spot import load_spot

log = logging.getLogger(__name__)

META = ["ticker", "event_ticker", "series", "ts", "close_ts", "strike", "y", "in_window",
        "bid", "ask", "bid_h", "bid_l", "ask_h", "ask_l", "px_h", "px_l"]


def base_features(windows: list[int]) -> list[str]:
    f = ["mid", "spread", "logit_mid", "pq", "vol_mid_15", "nchg_15", "bid_rng_1", "ask_rng_1",
         "vol_ratio_5_60", "doi_5", "doi_15", "avg_trsz_15", "max_tr_share_15", "vwap_dev_1",
         "last_px_dev", "tte_min", "log_tte", "hour_utc", "dow"]
    for w in windows:
        f += [f"dmid_{w}", f"lvol_{w}", f"tfi_{w}", f"ntr_{w}"]
    return f


def spot_features(windows: list[int]) -> list[str]:
    return ["spot_logm", "spot_sigma_60", "spot_z", "p_bs", "p_bs_minus_mid"] + [f"spot_ret_{w}" for w in windows]


def feature_columns(cfg: dict, df: pd.DataFrame | None = None) -> list[str]:
    w = cfg["features"]["windows_min"]
    sets = cfg["features"]["feature_sets"]
    cols: list[str] = []
    if "base" in sets:
        cols += base_features(w)
    if "spot" in sets:
        cols += spot_features(w)
    if "book" in sets:
        cols += BOOK_FEATURES
    exclude = set(cfg["features"].get("exclude") or [])
    cols = [c for c in cols if c not in exclude]
    if df is not None:  # drop columns that are absent or entirely empty (e.g. no recorder data yet)
        cols = [c for c in cols if c in df.columns and df[c].notna().any()]
    return cols


def _logit(p):
    p = np.clip(p, 0.01, 0.99)
    return np.log(p / (1 - p))


def market_frame(m, cd: pd.DataFrame, td: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    fcfg = cfg["features"]
    windows = fcfg["windows_min"]
    c = cd.sort_values("ts").drop_duplicates("ts", keep="last").set_index("ts")
    close = int(m.close_ts)
    grid = np.arange(int(c.index.min()), close + 1, 60, dtype="int64")
    c = c.reindex(grid)
    for side in ("bid", "ask"):
        c[f"{side}_c"] = c[f"{side}_c"].ffill()
        for k in ("o", "h", "l"):
            c[f"{side}_{k}"] = c[f"{side}_{k}"].fillna(c[f"{side}_c"])
    c["oi"] = c["oi"].ffill()
    vol = c["volume"].fillna(0.0)

    f = pd.DataFrame(index=grid)
    bid, ask = c["bid_c"], c["ask_c"]
    mid = (bid + ask) / 2
    f["mid"] = mid
    f["spread"] = (ask - bid) * 100
    f["logit_mid"] = _logit(mid)
    f["pq"] = mid * (1 - mid)
    dm = mid.diff() * 100
    for w in windows:
        f[f"dmid_{w}"] = (mid - mid.shift(w)) * 100
    f["vol_mid_15"] = dm.rolling(15, min_periods=3).std()
    f["nchg_15"] = (dm.abs() > 1e-9).astype(float).rolling(15, min_periods=1).sum()
    f["bid_rng_1"] = (c["bid_h"] - c["bid_l"]) * 100
    f["ask_rng_1"] = (c["ask_h"] - c["ask_l"]) * 100
    for w in windows:
        f[f"lvol_{w}"] = np.log1p(vol.rolling(w, min_periods=1).sum())
    f["vol_ratio_5_60"] = vol.rolling(5, min_periods=1).sum() / (vol.rolling(60, min_periods=1).sum() / 12 + 1)
    for w in (5, 15):
        prev = c["oi"].shift(w)
        f[f"doi_{w}"] = (c["oi"] - prev) / (prev + 1)

    # Trade flow, bucketed to the minute that contains each trade (ceil to minute end).
    if td is not None and not td.empty:
        t = td.copy()
        t["bucket"] = (np.ceil(t["ts"] / 60) * 60).astype("int64")
        t["buy"] = np.where(t["taker_yes"], t["count"], 0.0)
        t["sell"] = np.where(t["taker_yes"], 0.0, t["count"])
        g = t.groupby("bucket").agg(buy=("buy", "sum"), sell=("sell", "sum"), n=("count", "size"),
                                    mx=("count", "max")).reindex(grid).fillna(0.0)
        for w in windows:
            b = g["buy"].rolling(w, min_periods=1).sum()
            s = g["sell"].rolling(w, min_periods=1).sum()
            f[f"tfi_{w}"] = ((b - s) / (b + s)).fillna(0.0)
            f[f"ntr_{w}"] = np.log1p(g["n"].rolling(w, min_periods=1).sum())
        tot15 = (g["buy"] + g["sell"]).rolling(15, min_periods=1).sum()
        n15 = g["n"].rolling(15, min_periods=1).sum()
        f["avg_trsz_15"] = np.log1p(tot15 / n15.replace(0, np.nan))
        f["max_tr_share_15"] = g["mx"].rolling(15, min_periods=1).max() / tot15.replace(0, np.nan)
    else:
        for w in windows:
            f[f"tfi_{w}"] = np.nan
            f[f"ntr_{w}"] = np.nan
        f["avg_trsz_15"] = np.nan
        f["max_tr_share_15"] = np.nan

    f["vwap_dev_1"] = (c["px_mean"] - mid) * 100
    f["last_px_dev"] = (c["px_c"].ffill() - mid) * 100
    tte = (close - grid) / 60.0
    f["tte_min"] = tte
    f["log_tte"] = np.log1p(tte)
    f["hour_utc"] = (grid // 3600) % 24
    f["dow"] = ((grid // 86400) + 3) % 7  # Monday = 0

    # Meta + prices the backtester needs.
    f["ticker"] = m.ticker
    f["event_ticker"] = m.event_ticker
    f["series"] = m.series
    f["ts"] = grid
    f["close_ts"] = close
    f["strike"] = m.floor_strike if m.strike_type != "less" else m.cap_strike
    f["strike_type"] = m.strike_type
    f["y"] = 1 if m.result == "yes" else 0
    f["bid"], f["ask"] = bid, ask
    f["bid_h"], f["bid_l"], f["ask_h"], f["ask_l"] = c["bid_h"], c["bid_l"], c["ask_h"], c["ask_l"]
    f["px_h"], f["px_l"] = c["px_h"], c["px_l"]
    lo, hi = fcfg["mid_range"]
    f["in_window"] = ((tte <= fcfg["decision_window_min"]) & (tte >= fcfg["min_minutes_to_close"])
                      & mid.between(lo, hi) & ((ask - bid) <= fcfg["max_spread"]) & (bid > 0) & (ask < 1))
    return f.reset_index(drop=True)


def add_spot_features(df: pd.DataFrame, spot: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    s = spot.set_index("ts")["close"].astype(float)
    s = s[~s.index.duplicated()].sort_index()
    full = pd.RangeIndex(int(s.index.min()), int(s.index.max()) + 60, 60)
    s = s.reindex(full).ffill(limit=5)
    lr = np.log(s).diff()
    sigma = lr.rolling(60, min_periods=20).std().clip(lower=1e-5)
    S = s.reindex(df["ts"].to_numpy()).to_numpy()
    sig = sigma.reindex(df["ts"].to_numpy()).to_numpy()
    K = df["strike"].to_numpy(dtype=float)
    sign = np.where(df["strike_type"].to_numpy() == "less", -1.0, 1.0)
    valid = np.isin(df["strike_type"].to_numpy(), ["greater", "less"])
    tau = np.maximum(df["tte_min"].to_numpy() - 0.5, 0.5)  # settles on the average of the last 60s
    logm = np.log(S / K)
    z = sign * logm / (sig * np.sqrt(tau))
    out = df.copy()
    out["spot_logm"] = np.where(valid, sign * logm * 1e4, np.nan)
    out["spot_sigma_60"] = sig * 1e4
    out["spot_z"] = np.where(valid, z, np.nan)
    out["p_bs"] = np.where(valid, ndtr(z), np.nan)
    out["p_bs_minus_mid"] = (out["p_bs"] - out["mid"]) * 100
    logS = np.log(s)
    for w in windows:
        r = (logS - logS.shift(w)).reindex(df["ts"].to_numpy()).to_numpy()
        out[f"spot_ret_{w}"] = sign * r / (sig * np.sqrt(w))
    return out


def build_series_features(cfg: dict, series: str) -> pd.DataFrame:
    root = data_dir(cfg) / "raw" / "kalshi" / series / "events"
    frames = []
    n_events = 0
    for ev in sorted(p for p in root.iterdir() if (p / "_done").exists() and (p / "candles.parquet").exists()):
        mk = pd.read_parquet(ev / "markets.parquet")
        cd = pd.read_parquet(ev / "candles.parquet")
        if cd.empty:
            continue
        td_all = pd.read_parquet(ev / "trades.parquet") if (ev / "trades.parquet").exists() else pd.DataFrame()
        n_events += 1
        for m in mk[mk["ticker"].isin(cd["ticker"].unique())].itertuples():
            td = td_all[td_all["ticker"] == m.ticker] if not td_all.empty else td_all
            frames.append(market_frame(m, cd[cd["ticker"] == m.ticker], td, cfg))
    if not frames:
        raise SystemExit(f"no downloaded events for {series}; run `python -m predmkt download` first")
    df = pd.concat(frames, ignore_index=True)
    log.info("%s: %d events, %d markets, %d rows (%d in decision window)",
             series, n_events, df["ticker"].nunique(), len(df), int(df["in_window"].sum()))

    product = cfg["data"].get("spot_products", {}).get(series)
    if product:
        spot = load_spot(data_dir(cfg) / "raw" / "spot", product, df["ts"].min() - 3 * 3600, df["ts"].max())
        df = add_spot_features(df, spot, cfg["features"]["windows_min"])

    book = build_book_features(data_dir(cfg) / "raw" / "live")
    if not book.empty:
        df = df.merge(book, on=["ticker", "ts"], how="left")
        log.info("%s: order-book features matched %d decision rows",
                 series, int(df.loc[df["in_window"], "qi_l1"].notna().sum()))
    out = data_dir(cfg) / "features"
    out.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out / f"{series}.parquet", index=False)
    return df
