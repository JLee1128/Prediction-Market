"""Chronological event split, event-weighted XGBoost training from a market-price offset."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

log = logging.getLogger(__name__)


def market_logit(df: pd.DataFrame) -> np.ndarray:
    p = np.clip(df["mid"].to_numpy(dtype=float), 0.01, 0.99)
    return np.log(p / (1 - p))


def split_events(df: pd.DataFrame, cfg: dict) -> dict[str, pd.Series]:
    """Chronological split by event. Boundaries fall on event close times; any event whose decision
    window overlaps a boundary (plus embargo) is dropped, so no label is shared across splits."""
    mcfg, fcfg = cfg["model"], cfg["features"]
    rows = df[df["in_window"]]
    ev = rows.groupby("event_ticker").agg(first=("ts", "min"), close=("close_ts", "max")).sort_values("close")
    n = len(ev)
    if n < 10:
        raise SystemExit(f"only {n} events with decision rows; download more history")
    q_train = mcfg["split"]["train"]
    q_valid = q_train + mcfg["split"]["valid"]
    b1 = ev["close"].iloc[int(n * q_train) - 1]
    b2 = ev["close"].iloc[int(n * q_valid) - 1]
    gap = mcfg.get("embargo_minutes", 0) * 60
    train = ev.index[ev["close"] <= b1]
    valid = ev.index[(ev["first"] > b1 + gap) & (ev["close"] <= b2)]
    test = ev.index[ev["first"] > b2 + gap]
    dropped = n - len(train) - len(valid) - len(test)
    log.info("split: %d train / %d valid / %d test events (%d dropped at boundaries)",
             len(train), len(valid), len(test), dropped)
    e = df["event_ticker"]
    return {"train": e.isin(train) & df["in_window"],
            "valid": e.isin(valid) & df["in_window"],
            "test": e.isin(test) & df["in_window"],
            "bounds": {"train_end": float(b1), "valid_end": float(b2)}}


def event_weights(df: pd.DataFrame) -> np.ndarray:
    """Each event gets equal total weight; scaled so the mean row weight is 1."""
    counts = df.groupby("event_ticker")["ts"].transform("size").to_numpy(dtype=float)
    w = 1.0 / counts
    return w * len(w) / w.sum()


def _dmatrix(df: pd.DataFrame, feats: list[str], cfg: dict, label: bool = True,
             offset: np.ndarray | None = None) -> xgb.DMatrix:
    mcfg = cfg["model"]
    kw = {}
    if label:
        kw["label"] = df["y"].to_numpy()
        kw["weight"] = event_weights(df) if mcfg["weight_by_event"] else None
    if mcfg["use_market_offset"]:
        kw["base_margin"] = market_logit(df) if offset is None else offset
    return xgb.DMatrix(df[feats].astype(float), feature_names=feats, **kw)


def train_model(df: pd.DataFrame, splits: dict, feats: list[str], cfg: dict) -> xgb.Booster:
    mcfg = cfg["model"]
    tr, va = df[splits["train"]], df[splits["valid"]]
    dtr, dva = _dmatrix(tr, feats, cfg), _dmatrix(va, feats, cfg)
    booster = xgb.train(dict(mcfg["xgb_params"]), dtr, num_boost_round=mcfg["num_boost_round"],
                        evals=[(dtr, "train"), (dva, "valid")],
                        early_stopping_rounds=mcfg["early_stopping_rounds"], verbose_eval=False)
    log.info("trained: best iteration %d, valid logloss %.5f", booster.best_iteration, booster.best_score)
    return booster


def predict(booster: xgb.Booster, df: pd.DataFrame, feats: list[str], cfg: dict,
            offset: np.ndarray | None = None) -> np.ndarray:
    dm = _dmatrix(df, feats, cfg, label=False, offset=offset)
    best = getattr(booster, "best_iteration", None)
    rng = (0, best + 1) if best is not None else (0, 0)
    return booster.predict(dm, iteration_range=rng)


def permutation_importance(booster: xgb.Booster, df: pd.DataFrame, feats: list[str], cfg: dict,
                           seed: int = 0) -> pd.DataFrame:
    """Increase in event-weighted Brier score when one feature is shuffled (on held-out rows).
    The market-price offset is held fixed, so this measures each feature's contribution on top of it."""
    rng = np.random.default_rng(seed)
    y = df["y"].to_numpy()
    w = event_weights(df)
    offset = market_logit(df)

    def brier(p):
        return float(np.average((p - y) ** 2, weights=w))

    base = brier(predict(booster, df, feats, cfg, offset))
    rows = []
    for f in feats:
        d = df.copy()
        d[f] = rng.permutation(d[f].to_numpy())
        rows.append({"feature": f, "brier_increase": brier(predict(booster, d, feats, cfg, offset)) - base})
    return pd.DataFrame(rows).sort_values("brier_increase", ascending=False).reset_index(drop=True)


def save_model(booster: xgb.Booster, feats: list[str], meta: dict, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    booster.save_model(out / "model.json")
    (out / "meta.json").write_text(json.dumps({"features": feats, "best_iteration": booster.best_iteration, **meta},
                                              indent=1, default=str))


def load_model(path: Path) -> tuple[xgb.Booster, dict]:
    b = xgb.Booster()
    b.load_model(path / "model.json")
    meta = json.loads((path / "meta.json").read_text())
    b.best_iteration = meta.get("best_iteration")  # type: ignore[attr-defined]
    return b, meta
