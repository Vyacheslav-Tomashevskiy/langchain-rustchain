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

#: Public GitHub repo whose ``bounty``-labelled open issues are the canonical
#: list of open RustChain bounties (the same source the official
#: bounty-concierge aggregator reads). Unauthenticated, read-only.
DEFAULT_BOUNTY_REPO = "Scottcjn/rustchain-bounties"
GITHUB_API = "https://api.github.com"


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

    def epoch(self) -> dict:
        """Current epoch/slot, enrolled miners, epoch pot, total supply (/epoch)."""
        return self._get_json("/epoch")

    def hall_of_fame(self) -> dict:
        """Hall-of-Fame leaderboard of the oldest/most-attested machines
        (/hall/leaderboard), ranked by ``rust_score`` (antiquity × attestations)."""
        return self._get_json("/hall/leaderboard")

    def bounties(self, repo: str = DEFAULT_BOUNTY_REPO) -> list:
        """Open RustChain bounties — ``bounty``-labelled open GitHub issues.

        Reads the public, unauthenticated GitHub issues API for ``repo``
        (default :data:`DEFAULT_BOUNTY_REPO`). Pull requests that surface
        through the issues endpoint are filtered out. Returns the raw issue
        list so callers/summarizers can parse rewards themselves.
        """
        url = f"{GITHUB_API}/repos/{repo}/issues"
        params = {"labels": "bounty", "state": "open", "per_page": 100}
        resp = requests.get(
            url,
            params=params,
            timeout=self.timeout,
            verify=self.verify,
            headers={"Accept": "application/vnd.github+json"},
        )
        resp.raise_for_status()
        return [i for i in resp.json() if "pull_request" not in i]
