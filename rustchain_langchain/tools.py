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

import re
from typing import List

from .client import RustChainClient


# Matches a reward like "5 RTC" / "67RTC" / "1.5 RTC" but not "RTC5f3a..."
# wallet addresses. Mirrors the official bounty-concierge parser.
_RTC_PATTERN = re.compile(r"(?<![A-Za-z0-9])(\d+(?:[.,]\d+)?)\s*RTC\b", re.IGNORECASE)


def _parse_reward(title: str, body: str = "") -> float:
    """Best-effort RTC reward from an issue title (falling back to body).

    Picks the largest plain number followed by ``RTC`` (so a "50-200 RTC"
    range reports the upper bound). Returns ``0.0`` when none is found.
    """
    best = 0.0
    for text in (title or "", body or ""):
        for m in _RTC_PATTERN.finditer(text):
            try:
                val = float(m.group(1).replace(",", ""))
            except ValueError:
                continue
            best = max(best, val)
        if best:
            break
    return best


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


def summarize_epoch(data: dict) -> str:
    return (
        f"RustChain is in epoch {data.get('epoch', '?')} "
        f"(slot {data.get('slot', '?')}, {data.get('blocks_per_epoch', '?')} "
        f"blocks/epoch). {data.get('enrolled_miners', '?')} miner(s) enrolled, "
        f"epoch pot {data.get('epoch_pot', '?')} RTC, "
        f"total supply {data.get('total_supply_rtc', '?')} RTC."
    )


def summarize_hall_of_fame(data, top: int = 5) -> str:
    # /hall/leaderboard returns {"leaderboard": [...]}; tolerate a bare list.
    if isinstance(data, list):
        board = data
    elif isinstance(data, dict):
        board = data.get("leaderboard", [])
    else:
        board = []
    if not isinstance(board, list):
        board = []
    if not board:
        return "RustChain Hall of Fame: no machines reported."
    parts = [f"RustChain Hall of Fame — top {min(top, len(board))} oldest/most-attested machines:"]
    for m in board[:top]:
        model = m.get("device_model") or m.get("device_arch") or "unknown"
        year = m.get("manufacture_year")
        year_s = f", {year}" if year else ""
        parts.append(
            f"#{m.get('rank', '?')} {model}{year_s} — "
            f"rust_score {m.get('rust_score', '?')}, "
            f"{m.get('total_attestations', '?')} attestations"
            + (f" [{m.get('badge')}]" if m.get("badge") else "")
        )
    return "\n".join(parts)


def summarize_bounties(data, top: int = 5) -> str:
    issues = data if isinstance(data, list) else []
    issues = [i for i in issues if isinstance(i, dict) and "pull_request" not in i]
    if not issues:
        return "RustChain has no open bounties right now."
    scored = sorted(
        ((_parse_reward(i.get("title", ""), i.get("body", "") or ""), i) for i in issues),
        key=lambda t: t[0],
        reverse=True,
    )
    total = sum(r for r, _ in scored)
    parts = [
        f"{len(issues)} open RustChain bounties "
        f"(~{round(total)} RTC across those with a stated reward). Top:"
    ]
    for reward, i in scored[:top]:
        reward_s = f"{reward:g} RTC" if reward else "reward TBD"
        title = (i.get("title", "") or "").strip()
        parts.append(f"- [{reward_s}] #{i.get('number', '?')} {title[:80]}")
    return "\n".join(parts)


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
        _make(
            "rustchain_epoch",
            "Get the current RustChain epoch and slot, blocks per epoch, number of "
            "enrolled miners, the epoch reward pot, and total RTC supply. Use for "
            "questions about where the chain is in its reward cycle right now.",
            client.epoch,
            summarize_epoch,
        ),
        _make(
            "rustchain_hall_of_fame",
            "Get the RustChain Hall of Fame leaderboard — the oldest and most-attested "
            "machines, ranked by rust score (antiquity × attestations), with device "
            "model, manufacture year and badge. Use for questions about the oldest or "
            "top-ranked hardware on the network.",
            client.hall_of_fame,
            summarize_hall_of_fame,
        ),
        _make(
            "rustchain_bounties",
            "List the open RustChain bounties (paid tasks) with their RTC rewards, "
            "sorted by reward. Use when asked what bounties are open, what work pays "
            "RTC, or how an agent can earn on RustChain.",
            client.bounties,
            summarize_bounties,
        ),
    ]
