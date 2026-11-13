"""Evaluation metrics. Every uncertainty estimate resamples EVENTS, never individual snapshots."""
from __future__ import annotations

import numpy as np
import pandas as pd


# ---- probability quality ---------------------------------------------------------------------

def per_event_brier(df: pd.DataFrame, pcol: str) -> pd.Series:
    """Mean squared error of `pcol` vs outcome within each event (so each event counts once)."""
    se = (df[pcol] - df["y"]) ** 2
    return se.groupby(df["event_ticker"]).mean()


def brier_report(df: pd.DataFrame, pcols: list[str], n_boot: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ev = pd.DataFrame({c: per_event_brier(df, c) for c in pcols})
    idx = rng.integers(0, len(ev), size=(n_boot, len(ev)))
    rows = []
    for c in pcols:
        v = ev[c].to_numpy()
        boots = v[idx].mean(axis=1)
        rows.append({"model": c, "brier": v.mean(), "ci_low": np.quantile(boots, 0.025),
                     "ci_high": np.quantile(boots, 0.975), "events": len(v)})
    return pd.DataFrame(rows)


def brier_diff_ci(df: pd.DataFrame, pcol: str, base_col: str, n_boot: int, seed: int) -> dict:
    """Model minus baseline Brier (negative = model better), with an event-bootstrap 95% CI."""
    rng = np.random.default_rng(seed)
    d = (per_event_brier(df, pcol) - per_event_brier(df, base_col)).to_numpy()
    boots = d[rng.integers(0, len(d), size=(n_boot, len(d)))].mean(axis=1)
    return {"diff": float(d.mean()), "ci_low": float(np.quantile(boots, 0.025)),
            "ci_high": float(np.quantile(boots, 0.975)), "p_model_better": float((boots < 0).mean()),
            "events": len(d)}


def brier_by_tte(df: pd.DataFrame, pcols: list[str], edges: list[float]) -> pd.DataFrame:
    b = pd.cut(df["tte_min"], bins=edges, include_lowest=True)
    out = {}
    for c in pcols:
        se = (df[c] - df["y"]) ** 2
        # event-weighted within each bucket
        out[c] = se.groupby([b, df["event_ticker"]], observed=True).mean().groupby(level=0, observed=True).mean()
    res = pd.DataFrame(out)
    res["events"] = df.groupby(b, observed=True)["event_ticker"].nunique()
    res.index = res.index.astype(str)
    return res


def calibration_table(df: pd.DataFrame, pcol: str, bins: int) -> pd.DataFrame:
    """Reliability table, event-weighted (each event's rows share weight 1)."""
    w = 1.0 / df.groupby("event_ticker")["ts"].transform("size")
    b = pd.cut(df[pcol], bins=np.linspace(0, 1, bins + 1), include_lowest=True)
    g = pd.DataFrame({"p": df[pcol] * w, "y": df["y"] * w, "w": w, "b": b, "ev": df["event_ticker"]})
    t = g.groupby("b", observed=True).agg(p=("p", "sum"), y=("y", "sum"), w=("w", "sum"), events=("ev", "nunique"))
    out = pd.DataFrame({"mean_pred": t["p"] / t["w"], "realized": t["y"] / t["w"],
                        "weight": t["w"], "events": t["events"]})
    out.index = out.index.astype(str).rename("pred_bin")
    return out


def log_loss(df: pd.DataFrame, pcol: str) -> float:
    p = np.clip(df[pcol].to_numpy(), 1e-6, 1 - 1e-6)
    y = df["y"].to_numpy()
    ll = -(y * np.log(p) + (1 - y) * np.log(1 - p))
    return float(pd.Series(ll).groupby(df["event_ticker"].to_numpy()).mean().mean())


# ---- P&L ---------------------------------------------------------------------------------------

def max_drawdown(pnl: pd.Series, bankroll: float) -> dict:
    """Largest peak-to-trough decline of cumulative P&L (series indexed by time, in order)."""
    if pnl.empty:
        return {"max_dd": 0.0, "max_dd_pct": 0.0, "max_dd_pct_bankroll": 0.0, "peak_ts": None, "trough_ts": None}
    pnl = pnl.groupby(level=0).sum().sort_index()
    equity = np.concatenate([[bankroll], bankroll + pnl.cumsum().to_numpy()])
    times = [None, *pnl.index.tolist()]
    peak = np.maximum.accumulate(equity)
    dd = peak - equity
    i = int(dd.argmax())
    j = int(equity[: i + 1].argmax())

    def fmt(t):
        return None if t is None else pd.to_datetime(t, unit="s", utc=True).strftime("%Y-%m-%d %H:%M")

    return {"max_dd": float(dd[i]), "max_dd_pct": float(dd[i] / peak[i] * 100),
            "max_dd_pct_bankroll": float(dd[i] / bankroll * 100),
            "peak_ts": fmt(times[j]), "trough_ts": fmt(times[i])}


def pnl_summary(trades: pd.DataFrame, bankroll: float, n_boot: int, seed: int) -> dict:
    if trades.empty:
        return {"trades": 0, "net_pnl": 0.0}
    t = trades.sort_values("settle_ts")
    by_event = t.groupby("event_ticker")["pnl"].sum()
    rng = np.random.default_rng(seed)
    v = by_event.to_numpy()
    boots = v[rng.integers(0, len(v), size=(n_boot, len(v)))].sum(axis=1)
    daily = t.groupby(pd.to_datetime(t["settle_ts"], unit="s").dt.date)["pnl"].sum()
    dd = max_drawdown(t.set_index("settle_ts")["pnl"], bankroll)
    contracts = t["count"].sum()
    return {
        "trades": len(t),
        "events_traded": int(by_event.size),
        "contracts": float(contracts),
        "net_pnl": float(t["pnl"].sum()),
        "net_pnl_ci_low": float(np.quantile(boots, 0.025)),
        "net_pnl_ci_high": float(np.quantile(boots, 0.975)),
        "p_pnl_positive": float((boots > 0).mean()),
        "fees_paid": float(t["fee"].sum() + t.get("exit_fee", pd.Series(0.0)).sum()),
        "pnl_per_contract_cents": float(t["pnl"].sum() / contracts * 100) if contracts else 0.0,
        "hit_rate": float((t["pnl"] > 0).mean()),
        "avg_predicted_edge_cents": float(t["edge"].mean() * 100),
        "avg_entry_regret_cents": float(t["entry_regret"].mean() * 100),
        "return_pct": float(t["pnl"].sum() / bankroll * 100),
        "daily_sharpe": float(daily.mean() / daily.std() * np.sqrt(365)) if len(daily) > 2 and daily.std() > 0 else float("nan"),
        "days": int(len(daily)),
        **dd,
    }
