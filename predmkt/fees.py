"""Kalshi trading fees.

fee = ceil_to_cent(coef * contracts * price * (1 - price)), with taker coef 0.07 * multiplier.
Maker coef depends on the series' fee_type (0 for plain 'quadratic' series such as KXBTCD).
Settlement is free.
"""
from __future__ import annotations

import json
import logging
import math
from pathlib import Path

log = logging.getLogger(__name__)


def fee_coefs(series_info: dict, cfg: dict) -> tuple[float, float]:
    fcfg = cfg["fees"]
    fee_type = series_info.get("fee_type") or "quadratic"
    mult = float(series_info.get("fee_multiplier") or 1.0)
    if fee_type not in fcfg["maker_coef"]:
        log.warning("unknown fee_type %r; assuming taker 0.07 and maker 0.0175 (check the fee schedule)", fee_type)
    taker = fcfg["taker_coef"] * mult
    maker = fcfg["maker_coef"].get(fee_type, 0.0175) * mult
    return taker, maker


def load_fee_coefs(series_json: Path, cfg: dict) -> tuple[float, float]:
    info = json.loads(series_json.read_text()) if series_json.exists() else {}
    return fee_coefs(info, cfg)


def order_fee(price: float, count: float, coef: float, round_to: float = 0.01) -> float:
    if coef <= 0 or count <= 0:
        return 0.0
    raw = coef * count * price * (1.0 - price)
    return math.ceil(raw / round_to - 1e-9) * round_to


def fee_per_contract(price: float, coef: float) -> float:
    """Unrounded per-contract fee, used for edge estimates before sizing."""
    return coef * price * (1.0 - price)
