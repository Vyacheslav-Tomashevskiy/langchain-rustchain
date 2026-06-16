# SPDX-License-Identifier: MIT
"""Thin, read-only HTTP client for RustChain's public endpoints.

Everything here is unauthenticated and read-only — these are the same public
surfaces a human can hit in a browser (rustchain.org/facts.json, /payouts.json,
/api/miners, /health). No keys, no writes, no wallet operations. The LangChain
tools in ``tools.py`` wrap these methods; this module is deliberately
framework-free so it can be tested (and reused) without LangChain installed.
"""
from __future__ import annotations

import requests

DEFAULT_BASE_URL = "https://rustchain.org"
DEFAULT_TIMEOUT = 15


class RustChainClient:
    """Read-only client for the RustChain public API.

    Args:
        base_url: RustChain node/site base URL (default https://rustchain.org).
        timeout: per-request timeout in seconds.
        verify: TLS verification (default True; set False only for self-signed
            dev nodes).
    """

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        timeout: int = DEFAULT_TIMEOUT,
        verify: bool = True,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.verify = verify

    def _get_json(self, path: str) -> dict:
        url = f"{self.base_url}/{path.lstrip('/')}"
        resp = requests.get(url, timeout=self.timeout, verify=self.verify)
        resp.raise_for_status()
        return resp.json()

    # --- public, read-only surfaces -------------------------------------
    def network_stats(self) -> dict:
        """Self-verifying on-chain activity facts (facts.json)."""
        return self._get_json("/facts.json")

    def payouts(self) -> dict:
        """Chain-computed payout totals + recipient counts (payouts.json)."""
        return self._get_json("/payouts.json")

    def metrics(self) -> dict:
        """Repo/ecosystem metrics snapshot (metrics.json)."""
        return self._get_json("/metrics.json")

    def miners(self) -> dict:
        """Currently attesting miners, with device arch + antiquity multipliers."""
        return self._get_json("/api/miners")

    def health(self) -> dict:
        """Node health (ok, db_rw, version, backup age)."""
        return self._get_json("/health")
