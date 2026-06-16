# SPDX-License-Identifier: MIT
"""langchain-rustchain — read-only LangChain tools for the RustChain agent economy."""
from .client import RustChainClient
from .tools import (
    get_rustchain_tools,
    summarize_network,
    summarize_payouts,
    summarize_miners,
    summarize_health,
    summarize_epoch,
    summarize_hall_of_fame,
    summarize_bounties,
)

__version__ = "0.2.0"
__all__ = [
    "RustChainClient",
    "get_rustchain_tools",
    "summarize_network",
    "summarize_payouts",
    "summarize_miners",
    "summarize_health",
    "summarize_epoch",
    "summarize_hall_of_fame",
    "summarize_bounties",
]
