"""Command line: python -m predmkt <command> [options]

  download   fetch settled-market history + spot prices
  record     run the live order-book recorder (leave it running)
  features   rebuild the feature tables from downloaded data
  run        train, evaluate, backtest and write a report
  all        download + features + run
"""
from __future__ import annotations

import argparse
import logging

from .auth import signer_from_env
from .config import load_config, setup_logging
from .kalshi import KalshiClient

log = logging.getLogger("predmkt")


def main() -> None:
    ap = argparse.ArgumentParser(prog="predmkt", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None, help="path to config.yaml")
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("download")
    d.add_argument("--series", nargs="+")
    d.add_argument("--days", type=float)

    r = sub.add_parser("record")
    r.add_argument("--mode", choices=["poll", "ws"])
    r.add_argument("--series", nargs="+")
    r.add_argument("--hours", type=float, help="stop after this many hours (default: run forever)")

    f = sub.add_parser("features")
    f.add_argument("--series", nargs="+")

    for name in ("run", "all"):
        p = sub.add_parser(name)
        p.add_argument("--series", nargs="+")
        p.add_argument("--mode", choices=["taker", "maker"])
        p.add_argument("--ablation", action="store_true", help="also train base-only and spot-only models")
        p.add_argument("--rebuild", action="store_true", help="rebuild features even if cached")
        if name == "all":
            p.add_argument("--days", type=float)

    args = ap.parse_args()
    setup_logging()
    cfg = load_config(args.config)
    series = getattr(args, "series", None) or (cfg["recorder"]["series"] if args.cmd == "record" else cfg["data"]["series"])

    if args.cmd in ("download", "all"):
        from .download import download_series
        client = KalshiClient.from_config(cfg)
        for s in series:
            download_series(cfg, client, s, args.days or cfg["data"]["history_days"])

    if args.cmd == "record":
        from .recorder import run_poll, run_ws
        cfg["recorder"]["series"] = series
        client = KalshiClient.from_config(cfg)
        mode = args.mode or cfg["recorder"]["mode"]
        dur = args.hours * 3600 if args.hours else None
        if mode == "ws":
            signer = signer_from_env(cfg)
            if signer is None:
                raise SystemExit(f"ws mode needs {cfg['api']['key_id_env']} and {cfg['api']['private_key_path_env']} set; "
                                 "use --mode poll otherwise")
            run_ws(cfg, client, signer, dur)
        else:
            run_poll(cfg, client, dur)

    if args.cmd == "features":
        from .features import build_series_features
        for s in series:
            build_series_features(cfg, s)

    if args.cmd in ("run", "all"):
        from .pipeline import run
        for s in series:
            out = run(cfg, s, rebuild=args.rebuild or args.cmd == "all", ablation=args.ablation, mode=args.mode)
            print(f"report: {out / 'report.md'}")


if __name__ == "__main__":
    main()
