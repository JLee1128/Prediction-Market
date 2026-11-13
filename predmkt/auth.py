"""Kalshi API-key request signing (RSA-PSS over timestamp + method + path)."""
from __future__ import annotations

import base64
import os
import time

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding


class KalshiSigner:
    def __init__(self, key_id: str, private_key_path: str):
        self.key_id = key_id
        with open(os.path.expanduser(private_key_path), "rb") as f:
            self._key = serialization.load_pem_private_key(f.read(), password=None)

    def headers(self, method: str, path: str) -> dict[str, str]:
        """`path` is the URL path without query string, e.g. /trade-api/v2/portfolio/balance."""
        ts = str(int(time.time() * 1000))
        sig = self._key.sign(
            (ts + method.upper() + path).encode(),
            padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.DIGEST_LENGTH),
            hashes.SHA256(),
        )
        return {
            "KALSHI-ACCESS-KEY": self.key_id,
            "KALSHI-ACCESS-TIMESTAMP": ts,
            "KALSHI-ACCESS-SIGNATURE": base64.b64encode(sig).decode(),
        }


def signer_from_env(cfg: dict) -> KalshiSigner | None:
    key_id = os.environ.get(cfg["api"]["key_id_env"])
    key_path = os.environ.get(cfg["api"]["private_key_path_env"])
    if key_id and key_path:
        return KalshiSigner(key_id, key_path)
    return None
