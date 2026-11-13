"""Train -> predict -> evaluate -> backtest -> report, for one series."""
from __future__ import annotations

import datetime as dt
import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import metrics as M  # noqa: E402
from .backtest import longshot_baseline, run_backtest, tune_threshold  # noqa: E402
from .config import data_dir, project_path  # noqa: E402
from .features import base_features, build_series_features, feature_columns, spot_features  # noqa: E402
from .fees import load_fee_coefs  # noqa: E402
from .model import permutation_importance, predict, save_model, split_events, train_model  # noqa: E402

log = logging.getLogger(__name__)


def _md(df: pd.DataFrame, floatfmt: str = "{:.4f}", index: bool = True) -> str:
    d = df.reset_index() if index else df
    cols = [str(c) for c in d.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for row in d.itertuples(index=False):
        cells = [floatfmt.format(v) if isinstance(v, (float, np.floating)) and np.isfinite(v) else str(v) for v in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _date(ts) -> str:
    return dt.datetime.fromtimestamp(float(ts), dt.timezone.utc).strftime("%Y-%m-%d %H:%M")


def load_or_build_features(cfg: dict, series: str, rebuild: bool) -> pd.DataFrame:
    f = data_dir(cfg) / "features" / f"{series}.parquet"
    if f.exists() and not rebuild:
        return pd.read_parquet(f)
    return build_series_features(cfg, series)


def run(cfg: dict, series: str, rebuild: bool = False, ablation: bool = False, mode: str | None = None) -> Path:
    if mode:
        cfg["strategy"]["mode"] = mode
    ecfg, scfg = cfg["evaluation"], cfg["strategy"]
    df = load_or_build_features(cfg, series, rebuild)
    feats = feature_columns(cfg, df)
    splits = split_events(df, cfg)
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = f"{series}-{scfg['mode']}-{stamp}"
    out = project_path("reports") / run_name
    out.mkdir(parents=True, exist_ok=True)

    booster = train_model(df, splits, feats, cfg)
    save_model(booster, feats, {"series": series, "bounds": splits["bounds"], "config": cfg},
               project_path("models") / run_name)

    df["p_model"] = np.nan
    for k in ("valid", "test"):
        df.loc[splits[k], "p_model"] = predict(booster, df[splits[k]], feats, cfg)
    df["p_market"] = df["mid"]

    test_rows = df[splits["test"]].copy()
    valid_rows = df[splits["valid"]].copy()
    pcols = ["p_model", "p_market"] + (["p_bs"] if "p_bs" in df and df["p_bs"].notna().any() else [])

    # Optional ablations: same split, different feature sets.
    if ablation:
        variants = {"p_xgb_base_only": base_features(cfg["features"]["windows_min"])}
        if "p_bs" in pcols:
            variants["p_xgb_spot_only"] = ["mid", "logit_mid", "tte_min", "log_tte"] + spot_features(cfg["features"]["windows_min"])
        for name, fs in variants.items():
            fs = [f for f in fs if f in df.columns and df[f].notna().any()]
            b = train_model(df, splits, fs, cfg)
            test_rows[name] = predict(b, test_rows, fs, cfg)
            pcols.append(name)

    if "p_bs" in pcols:  # p_bs is undefined for a few rows (missing spot); evaluate on rows where all exist
        test_rows = test_rows[test_rows[pcols].notna().all(axis=1)]

    # ---- calibration / Brier --------------------------------------------------------------------
    nb, seed = ecfg["bootstrap_samples"], ecfg["seed"]
    brier = M.brier_report(test_rows, pcols, nb, seed)
    brier["log_loss"] = [M.log_loss(test_rows, c) for c in pcols]
    diff = M.brier_diff_ci(test_rows, "p_model", "p_market", nb, seed)
    by_tte = M.brier_by_tte(test_rows, pcols, ecfg["tte_buckets_min"])
    calib = M.calibration_table(test_rows, "p_model", ecfg["calibration_bins"])
    calib_mkt = M.calibration_table(test_rows, "p_market", ecfg["calibration_bins"])

    # ---- strategy: tune threshold on valid, run once on test ------------------------------------
    taker_coef, maker_coef = load_fee_coefs(data_dir(cfg) / "raw" / "kalshi" / series / "series.json", cfg)
    valid_events = df[df["event_ticker"].isin(valid_rows["event_ticker"].unique())].copy()
    valid_events.loc[~splits["valid"].loc[valid_events.index], "p_model"] = np.nan
    th, grid = tune_threshold(valid_events, cfg, taker_coef, maker_coef)
    test_events = df[df["event_ticker"].isin(test_rows["event_ticker"].unique())].copy()
    test_events.loc[~test_events.index.isin(test_rows.index), "p_model"] = np.nan
    trades, bstats = run_backtest(test_events, cfg, taker_coef, maker_coef, th)
    pnl = M.pnl_summary(trades, scfg["bankroll"], nb, seed)
    bmax = ecfg.get("baseline_max_price", 0.10)
    base_trades, _ = longshot_baseline(test_events, cfg, taker_coef, bmax)
    base_pnl = M.pnl_summary(base_trades, scfg["bankroll"], nb, seed)
    imp = permutation_importance(booster, valid_rows, feats, cfg, seed)

    # ---- artifacts --------------------------------------------------------------------------------
    test_rows[["ticker", "event_ticker", "ts", "tte_min", "y", "bid", "ask", *pcols]].to_parquet(out / "test_predictions.parquet")
    trades.to_csv(out / "trades.csv", index=False)
    base_trades.to_csv(out / "baseline_trades.csv", index=False)
    _plots(out, calib, calib_mkt, trades, scfg["bankroll"], base_trades)
    gate = _gate(diff, pnl, base_pnl)
    summary = {"series": series, "threshold": th, "brier_diff": diff, "pnl": pnl, "baseline_pnl": base_pnl,
               "baseline_max_price": bmax, "backtest_stats": bstats,
               "gate": gate, "features": feats, "best_iteration": int(booster.best_iteration),
               "fee_coefs": {"taker": taker_coef, "maker": maker_coef}}
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))

    ev_counts = {k: int(df.loc[splits[k], "event_ticker"].nunique()) for k in ("train", "valid", "test")}
    span = {k: (_date(df.loc[splits[k], "ts"].min()), _date(df.loc[splits[k], "close_ts"].max())) for k in ("train", "valid", "test")}
    lines = [
        f"# Backtest report: {series} ({scfg['mode']})",
        f"Generated {dt.datetime.now():%Y-%m-%d %H:%M}. Model: `models/{run_name}`.",
        "",
        "## Data split (chronological, by event)",
        _md(pd.DataFrame({"events": ev_counts, "from (UTC)": {k: v[0] for k, v in span.items()},
                          "to (UTC)": {k: v[1] for k, v in span.items()}})),
        "",
        f"{len(feats)} features; best boosting round {booster.best_iteration}. "
        f"Fees: taker coef {taker_coef:.4f}, maker coef {maker_coef:.4f}.",
        "",
        "## 1. Calibration on the test period (each event weighted equally)",
        "Brier score (lower is better), with 95% CIs from resampling events:",
        _md(brier, index=False),
        "",
        f"**Model minus market Brier: {diff['diff']:+.5f}** (95% CI {diff['ci_low']:+.5f} to {diff['ci_high']:+.5f}; "
        f"model better in {diff['p_model_better']:.0%} of resamples, {diff['events']} events). Negative means the model beats the market price.",
        "",
        "Brier by minutes to close:",
        _md(by_tte),
        "",
        "Reliability (model). `realized` should track `mean_pred`:",
        _md(calib),
        "",
        "## 2. Strategy on the test period",
        f"Edge threshold chosen on the validation period: **{th:.3f}** (grid below), then applied once to test.",
        _md(grid, index=False, floatfmt="{:.2f}"),
        "",
        _md(pd.DataFrame({"value": {k: v for k, v in pnl.items()}}), floatfmt="{:.3f}"),
        "",
        f"Backtest counters: {bstats}",
        "",
        f"**No-model benchmark:** at each market's first decision minute, buy any side priced at or below {bmax:.2f}, "
        "with the same fees, caps and test rows. If the model can't beat it, its P&L comes from cheap contracts winning, not from the model.",
        _md(pd.DataFrame({"model": pnl, "no-model baseline": base_pnl}).loc[
            [k for k in _COMPARE if k in pnl or k in base_pnl]], floatfmt="{:.2f}"),
        "",
        "Net P&L includes fees and the spread. Drawdown is peak-to-trough of cumulative realized P&L; the gate compares it with the starting bankroll. "
        "Entry regret = how much cheaper the same side was within the next "
        f"{scfg['regret_window_min']} minutes (hindsight).",
        "",
        "![equity](equity.png) ![calibration](calibration.png)",
        "",
        "## 3. Permutation importance (validation period)",
        _md(imp.head(25), index=False, floatfmt="{:.6f}"),
        "",
        "## 4. Gate before real money",
        _md(pd.DataFrame(gate, columns=["check", "pass"]), index=False),
    ]
    (out / "report.md").write_text("\n".join(lines) + "\n")
    log.info("report written to %s", out / "report.md")
    return out


_COMPARE = ["trades", "events_traded", "net_pnl", "net_pnl_ci_low", "net_pnl_ci_high", "pnl_per_contract_cents",
            "hit_rate", "max_dd", "max_dd_pct_bankroll"]


def _gate(diff: dict, pnl: dict, base_pnl: dict) -> list[tuple[str, bool]]:
    return [
        ("Model Brier beats market (95% CI below 0)", diff["ci_high"] < 0),
        ("Net P&L beats the no-model long-shot baseline", pnl.get("net_pnl", 0) > base_pnl.get("net_pnl", 0)),
        ("Net P&L > 0 after fees", pnl.get("net_pnl", 0) > 0),
        ("Event-bootstrap P&L CI lower bound > 0", pnl.get("net_pnl_ci_low", -1) > 0),
        ("Max drawdown < 20% of starting bankroll", pnl.get("max_dd_pct_bankroll", 0) < 20),
        ("At least 30 events traded", pnl.get("events_traded", 0) >= 30),
    ]


def _plots(out: Path, calib: pd.DataFrame, calib_mkt: pd.DataFrame, trades: pd.DataFrame, bankroll: float,
           base_trades: pd.DataFrame | None = None) -> None:
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.plot([0, 1], [0, 1], color="#999", lw=1, ls="--")
    ax.plot(calib_mkt["mean_pred"], calib_mkt["realized"], "o-", label="market mid", color="#888")
    ax.plot(calib["mean_pred"], calib["realized"], "o-", label="model", color="#1f6feb")
    ax.set_xlabel("predicted P(YES)")
    ax.set_ylabel("realized frequency")
    ax.set_title("Calibration (test)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "calibration.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 3.5))
    for tr, label, color in ((base_trades, "no-model baseline", "#999"), (trades, "model", "#1f6feb")):
        if tr is not None and not tr.empty:
            t = tr.sort_values("settle_ts")
            ax.plot(pd.to_datetime(t["settle_ts"], unit="s"), bankroll + t["pnl"].cumsum(), color=color, label=label)
    ax.legend()
    ax.axhline(bankroll, color="#ccc", lw=1, ls="--")
    ax.set_title("Equity (test, realized)")
    ax.set_ylabel("$")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out / "equity.png", dpi=120)
    plt.close(fig)
