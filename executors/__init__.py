"""Native coding-CLI executors.

Adapters build argv arrays and parse JSONL streams; the process supervisor owns
spawning. ``CodexExecutor`` pins Codex threads by id, ``ClaudeExecutor`` pins
Claude sessions by an explicit UUID.
"""

from __future__ import annotations

from executors.base import (
    ExecutorAdapter,
    ExecutorConfigError,
    ExecutorResult,
    FollowUpNotSupportedError,
    STATUSES,
)
from executors.claude import ClaudeExecutor
from executors.codex import CodexExecutor

__all__ = [
    "ClaudeExecutor",
    "CodexExecutor",
    "ExecutorAdapter",
    "ExecutorConfigError",
    "ExecutorResult",
    "FollowUpNotSupportedError",
    "STATUSES",
]
