# SPDX-License-Identifier: MIT
"""LangChain tools that let an agent read RustChain's live, public state.

Read-only and keyless. The ``summarize_*`` functions are plain and unit-tested
on their own; the LangChain ``BaseTool`` wrappers below call the client and feed
the JSON through them, returning a compact, agent-friendly string (agents do
better with a 3-line summary than a 5KB JSON blob).

Usage:

    from rustchain_langchain import get_rustchain_tools
    tools = get_rustchain_tools()              # list[BaseTool], drop into any agent
    # e.g. with langgraph / AgentExecutor / bind_tools(...)
"""
from __future__ import annotations

from typing import List

from .client import RustChainClient


# --- pure summarizers (framework-free, unit-tested) ---------------------
def summarize_network(data: dict) -> str:
    facts = {f.get("id"): f for f in data.get("facts", [])}
    a = (facts.get("onchain_activity") or {}).get("value", {})
    parts = [f"RustChain on-chain activity (as of {data.get('as_of', '?')}):"]
    if a:
        parts.append(
            f"- {a.get('wallet_transfers', '?')} wallet transfers, "
            f"{a.get('rtc_moved_in_transfers', '?')} RTC moved across "
            f"{a.get('distinct_wallets', '?')} distinct wallets"
        )
        parts.append(f"- {a.get('ledger_entries_total', '?')} total ledger entries")
    if not a:
        parts.append(f"- {data.get('tagline', 'verifiable agent economy')}")
    return "\n".join(parts)


def summarize_payouts(data: dict) -> str:
    return (
        f"RustChain payouts: {data.get('total_paid_rtc', '?')} RTC paid to "
        f"{data.get('unique_recipients', '?')} distinct recipients across "
        f"{data.get('transactions', '?')} transactions "
        f"(updated {data.get('updated_at', '?')})."
    )


def summarize_miners(data) -> str:
    # /api/miners returns {"miners": [...]}, but tolerate a bare list too.
    if isinstance(data, list):
        miners = data
    elif isinstance(data, dict):
        miners = data.get("miners", [])
    else:
        miners = []
    if not isinstance(miners, list):
        miners = []
    n = len(miners)
    from collections import Counter

    arches = Counter(m.get("device_arch", "unknown") for m in miners)
    top = ", ".join(f"{a}×{c}" for a, c in arches.most_common(6))
    return (
        f"RustChain has {n} attesting miner(s). By architecture: {top or 'n/a'}. "
        "Older/exotic hardware earns higher antiquity multipliers."
    )


def summarize_health(data: dict) -> str:
    return (
        f"RustChain node health: ok={data.get('ok', '?')}, "
        f"db_rw={data.get('db_rw', '?')}, version={data.get('version', '?')}, "
        f"backup_age_hours={round(data.get('backup_age_hours', 0), 1)}."
    )


# --- LangChain tool wrappers --------------------------------------------
def get_rustchain_tools(
    base_url: str = "https://rustchain.org",
    timeout: int = 15,
    verify: bool = True,
) -> List["object"]:
    """Return a list of LangChain ``BaseTool`` objects for RustChain.

    Imported lazily so the client/summarizers stay usable without LangChain.
    Requires ``langchain-core``.
    """
    from langchain_core.tools import BaseTool  # lazy import

    client = RustChainClient(base_url=base_url, timeout=timeout, verify=verify)

    def _make(name: str, description: str, fetch, summarize):
        class _Tool(BaseTool):
            def _run(self, *args, **kwargs) -> str:
                try:
                    return summarize(fetch())
                except Exception as e:  # never raise inside an agent loop
                    return f"RustChain query failed ({type(e).__name__}): {e}"

            async def _arun(self, *args, **kwargs) -> str:
                return self._run(*args, **kwargs)

        return _Tool(name=name, description=description)

    return [
        _make(
            "rustchain_network_stats",
            "Get RustChain's live on-chain activity (wallet transfers, RTC moved, "
            "distinct wallets). Use when asked about RustChain network activity or size.",
            client.network_stats,
            summarize_network,
        ),
        _make(
            "rustchain_payouts",
            "Get total RTC paid out and the number of distinct recipients on RustChain. "
            "Use for questions about how much RustChain has paid contributors/miners.",
            client.payouts,
            summarize_payouts,
        ),
        _make(
            "rustchain_miners",
            "Get the current attesting miners and a breakdown by hardware architecture "
            "(PowerPC G4/G5, POWER8, x86, Apple Silicon, etc.). Use for questions about "
            "who is mining or what hardware is on the network.",
            client.miners,
            summarize_miners,
        ),
        _make(
            "rustchain_node_health",
            "Check whether the RustChain node is healthy (ok, db read-write, version, "
            "backup age). Use to verify the network is up before relying on it.",
            client.health,
            summarize_health,
        ),
    ]
