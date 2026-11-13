from __future__ import annotations

import datetime as dt
import logging
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | Path | None = None) -> dict:
    path = Path(path) if path else ROOT / "config.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


def project_path(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def data_dir(cfg: dict) -> Path:
    return project_path(cfg["data"]["root"])


def iso_to_ts(s: str | None) -> float:
    """ISO-8601 string (Kalshi style, 'Z' suffix, optional fractional seconds) -> unix seconds."""
    if not s:
        return float("nan")
    return dt.datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")
