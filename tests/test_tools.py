# SPDX-License-Identifier: MIT
"""Tests for langchain-rustchain. No network: requests is monkeypatched."""
import json
from unittest import mock

from rustchain_langchain import (
    RustChainClient,
    summarize_network,
    summarize_payouts,
    summarize_miners,
    summarize_health,
    summarize_balance,
    summarize_epoch,
    summarize_bounties,
    summarize_provenance,
)


class _Resp:
    def __init__(self, payload):
        self._p = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._p


def _client_returning(payload):
    c = RustChainClient(base_url="https://example.test")
    with mock.patch("rustchain_langchain.client.requests.get", return_value=_Resp(payload)) as g:
        return c, g


def test_client_builds_url_and_parses():
    c, g = _client_returning({"ok": True})
    with mock.patch("rustchain_langchain.client.requests.get", return_value=_Resp({"ok": True})) as g:
        out = c.health()
    assert out == {"ok": True}
    called_url = g.call_args[0][0]
    assert called_url == "https://example.test/health"


def test_summarize_network():
    data = {
        "as_of": "2026-06-16",
        "facts": [
            {"id": "onchain_activity", "value": {
                "wallet_transfers": 3112, "rtc_moved_in_transfers": 124848.19,
                "distinct_wallets": 1354, "ledger_entries_total": 7584}},
        ],
    }
    s = summarize_network(data)
    assert "3112 wallet transfers" in s
    assert "1354 distinct wallets" in s


def test_summarize_payouts():
    s = summarize_payouts({"total_paid_rtc": "66,531+", "unique_recipients": 1061,
                           "transactions": 3234, "updated_at": "2026-06-16"})
    assert "66,531+ RTC paid" in s
    assert "1061 distinct recipients" in s


def test_summarize_miners_counts_by_arch():
    data = {"miners": [
        {"device_arch": "G4"}, {"device_arch": "G4"}, {"device_arch": "POWER8"},
        {"device_arch": "modern"},
    ]}
    s = summarize_miners(data)
    assert "4 attesting miner" in s
    assert "G4×2" in s


def test_summarize_miners_tolerates_bare_list():
    s = summarize_miners([{"device_arch": "M4"}])
    assert "1 attesting miner" in s


def test_summarize_health():
    s = summarize_health({"ok": True, "db_rw": True, "version": "2.2.1", "backup_age_hours": 10.87})
    assert "ok=True" in s and "version=2.2.1" in s


def test_tool_run_never_raises_on_failure():
    # if langchain-core is unavailable, skip the wrapper test gracefully
    try:
        from rustchain_langchain import get_rustchain_tools
        tools = get_rustchain_tools(base_url="https://example.test")
    except Exception:
        return
    tool = next(t for t in tools if t.name == "rustchain_payouts")
    with mock.patch("rustchain_langchain.client.requests.get", side_effect=RuntimeError("boom")):
        out = tool._run()
    assert "RustChain query failed" in out  # graceful, not an exception


def test_tool_run_summarizes_on_success():
    try:
        from rustchain_langchain import get_rustchain_tools
        tools = get_rustchain_tools(base_url="https://example.test")
    except Exception:
        return
    tool = next(t for t in tools if t.name == "rustchain_payouts")
    payload = {"total_paid_rtc": "66,531+", "unique_recipients": 1061, "transactions": 3234, "updated_at": "x"}
    with mock.patch("rustchain_langchain.client.requests.get", return_value=_Resp(payload)):
        out = tool._run()
    assert "66,531+ RTC paid" in out


def test_summarize_balance():
    s = summarize_balance({"miner_id": "dual-g4-125", "amount_rtc": 718.51})
    assert "dual-g4-125" in s and "718.51 RTC" in s


def test_summarize_epoch():
    s = summarize_epoch({"epoch": 195, "slot": 28191, "enrolled_miners": 24,
                         "epoch_pot": 1.5, "blocks_per_epoch": 144, "total_supply_rtc": 8388608})
    assert "epoch 195" in s and "24 enrolled" in s and "1.5 RTC" in s


def test_summarize_bounties():
    s = summarize_bounties([{"number": 3074, "reward": "17 RTC", "title": "LangChain tool",
                             "url": "http://x", "created": "2026-03"}])
    assert "#3074" in s and "17 RTC" in s


def test_summarize_bounties_empty():
    assert "No open" in summarize_bounties([])


def test_client_balance_uses_wallet_balance_endpoint():
    c = RustChainClient(base_url="https://example.test")
    with mock.patch("rustchain_langchain.client.requests.get", return_value=_Resp({"amount_rtc": 5.0, "miner_id": "x"})) as g:
        out = c.balance("x")
    assert out["amount_rtc"] == 5.0
    assert g.call_args[0][0] == "https://example.test/wallet/balance"  # NOT bare /balance


_AGENTS = [
    {"agent_id": "bcn_sophia_elya", "name": "Sophia Elya", "status": "active",
     "relay": False, "model_id": "gpt-x", "capabilities": ["chat", "curate"]},
    {"agent_id": "bcn_quiet_one", "name": "QuietOne", "status": "presumed_dead", "relay": True},
]
_CONTRACTS = [
    {"id": "ctr_05ce6bf7", "type": "lease_to_own", "amount": 5.0, "currency": "RTC",
     "state": "offered", "from": "bcn_sophia_elya", "to": "relay_sh_sophia_elya"},
    {"id": "ctr_other", "type": "lease", "amount": 2.0, "currency": "RTC",
     "state": "active", "from": "bcn_nobody", "to": "bcn_else"},
]


def test_client_provenance_composes_identity_and_contracts():
    c = RustChainClient(base_url="https://example.test")
    # provenance() calls beacon_agents() then beacon_contracts()
    with mock.patch("rustchain_langchain.client.requests.get",
                    side_effect=[_Resp(_AGENTS), _Resp(_CONTRACTS)]):
        out = c.provenance("bcn_sophia_elya")
    assert out["found"] is True
    assert out["identity"]["name"] == "Sophia Elya"
    assert out["registered_agents"] == 2
    assert len(out["contracts"]) == 1
    assert out["contracts"][0]["role"] == "payer"
    assert out["contracts"][0]["counterparty"] == "relay_sh_sophia_elya"


def test_client_provenance_matches_display_name():
    c = RustChainClient(base_url="https://example.test")
    with mock.patch("rustchain_langchain.client.requests.get",
                    side_effect=[_Resp(_AGENTS), _Resp(_CONTRACTS)]):
        out = c.provenance("Sophia Elya")
    assert out["found"] is True and out["identity"]["agent_id"] == "bcn_sophia_elya"


def test_client_provenance_not_found():
    c = RustChainClient(base_url="https://example.test")
    with mock.patch("rustchain_langchain.client.requests.get", return_value=_Resp(_AGENTS)):
        out = c.provenance("bcn_ghost")
    assert out["found"] is False and out["identity"] is None and out["contracts"] == []


def test_summarize_provenance_found():
    s = summarize_provenance({
        "agent_id": "bcn_sophia_elya", "found": True, "registered_agents": 2,
        "identity": _AGENTS[0],
        "contracts": [{"id": "ctr_05ce6bf7", "type": "lease_to_own", "amount": 5.0,
                       "currency": "RTC", "state": "offered", "role": "payer",
                       "counterparty": "relay_sh_sophia_elya"}],
    })
    assert "Sophia Elya" in s
    assert "IDENTITY PRESENT" in s
    assert "1 Beacon contract" in s and "ctr_05ce6bf7" in s
    assert "not yet deployed" in s  # honest about the un-built content layer


def test_summarize_provenance_not_found():
    s = summarize_provenance({"agent_id": "bcn_ghost", "found": False,
                              "registered_agents": 116, "identity": None, "contracts": []})
    assert "No Beacon agent found" in s and "116" in s


def test_provenance_tool_run():
    try:
        from rustchain_langchain import get_rustchain_tools
        tools = get_rustchain_tools(base_url="https://example.test")
    except Exception:
        return
    tool = next(t for t in tools if t.name == "rustchain_provenance")
    with mock.patch("rustchain_langchain.client.requests.get",
                    side_effect=[_Resp(_AGENTS), _Resp(_CONTRACTS)]):
        out = tool._run("bcn_sophia_elya")
    assert "Sophia Elya" in out and "Agent layer" in out


def test_provenance_tool_never_raises():
    try:
        from rustchain_langchain import get_rustchain_tools
        tools = get_rustchain_tools(base_url="https://example.test")
    except Exception:
        return
    tool = next(t for t in tools if t.name == "rustchain_provenance")
    with mock.patch("rustchain_langchain.client.requests.get", side_effect=RuntimeError("boom")):
        out = tool._run("bcn_x")
    assert "RustChain query failed" in out
