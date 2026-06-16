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
