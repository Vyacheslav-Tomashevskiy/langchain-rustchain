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

    def balance(self, miner_id: str) -> dict:
        """RTC balance for a wallet / miner id.

        Uses the live /wallet/balance endpoint (the bare /balance path 404s).
        """
        url = f"{self.base_url}/wallet/balance"
        resp = requests.get(
            url, params={"miner_id": miner_id}, timeout=self.timeout, verify=self.verify
        )
        resp.raise_for_status()
        return resp.json()

    def epoch(self) -> dict:
        """Current epoch: number, slot, enrolled miners, reward pot, total supply."""
        return self._get_json("/epoch")

    def bounties(self, limit: int = 10) -> list:
        """Open RustChain bounties (GitHub issues on Scottcjn/rustchain-bounties).

        Read-only GitHub search; returns a list of {number, title, reward, url, created}.
        """
        limit = max(1, min(int(limit), 50))
        url = (
            "https://api.github.com/search/issues?"
            "q=repo:Scottcjn/rustchain-bounties+state:open+is:issue&"
            f"per_page={limit}&sort=created&order=desc"
        )
        resp = requests.get(
            url, timeout=self.timeout,
            headers={"Accept": "application/vnd.github.v3+json"},
        )
        resp.raise_for_status()
        items = resp.json().get("items", [])[:limit]
        import re

        out = []
        for it in items:
            body = it.get("body", "") or ""
            m = re.search(r"(\d+)\s*RTC", body)
            out.append({
                "number": it.get("number"),
                "title": (it.get("title") or "")[:100],
                "reward": f"{m.group(1)} RTC" if m else "see issue",
                "url": it.get("html_url"),
                "created": (it.get("created_at") or "")[:10],
            })
        return out

    def beacon_agents(self) -> list:
        """Registered Beacon agent-identity cards (id ``bcn_<hex>``, name, status)."""
        data = self._get_json("/beacon/api/agents")
        return data if isinstance(data, list) else []

    def beacon_contracts(self) -> list:
        """Open Beacon economic contracts (leases/offers between agents)."""
        data = self._get_json("/beacon/api/contracts")
        return data if isinstance(data, list) else []

    def provenance(self, agent_id: str) -> dict:
        """RIP-0310 Proof-of-Provenance status for a Beacon agent id.

        Read-only / keyless. Composes the deployed provenance signals for a
        ``bcn_<id>`` identity: its Beacon agent card (Agent layer) and any
        Beacon contracts it is party to (Economic layer). The Content-binding
        layer (a live ``BindingCert``) is specified in RIP-0310 but not yet
        deployed, so it is reported as such rather than fabricated. Returns a
        structured dict; ``summarize_provenance`` turns it into an agent-friendly
        string.
        """
        agent_id = (agent_id or "").strip()
        agents = self.beacon_agents()
        card = next((a for a in agents if a.get("agent_id") == agent_id), None)
        if card is None:
            # tolerate being given a display name instead of a bcn_ id
            card = next(
                (a for a in agents
                 if (a.get("name") or "").lower() == agent_id.lower()),
                None,
            )

        contracts = []
        if card is not None:
            aid = card.get("agent_id")
            for c in self.beacon_contracts():
                if c.get("from") == aid or c.get("to") == aid:
                    role = "payer" if c.get("from") == aid else "payee"
                    other = c.get("to") if role == "payer" else c.get("from")
                    contracts.append({
                        "id": c.get("id"),
                        "type": c.get("type"),
                        "amount": c.get("amount"),
                        "currency": c.get("currency"),
                        "state": c.get("state"),
                        "role": role,
                        "counterparty": other,
                    })

        return {
            "agent_id": agent_id,
            "found": card is not None,
            "registered_agents": len(agents),
            "identity": card,
            "contracts": contracts,
        }
