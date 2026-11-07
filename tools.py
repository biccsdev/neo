"""Worker-facing tool boundary for NEO.

A worker process holds ``NEO_TASK_ID`` and a random ``NEO_TASK_TOKEN``. The
plugin stores only a SHA-256 digest of that token, so a database read never
yields a usable credential. Every worker transition verifies the digest inside
the same transaction as its state change, and a rejected token changes nothing.

This module deliberately knows nothing about process management, supervisors,
previews or artifacts; it is only the authorization seam and the update call.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
from typing import Optional, Union

from events import TaskService
from registry import InvalidWorkerTokenError, Registry
from state_machine import Evidence, TaskState

TOKEN_BYTES = 32


class WorkerToolError(RuntimeError):
    """Base class for a refused worker call."""


def _digest(token: str) -> str:
    """Return the stored form of a worker token: its SHA-256 hex digest."""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


class WorkerTokenStore:
    """Issue and verify per-task worker tokens without persisting them."""

    def __init__(self, registry: Registry) -> None:
        self._registry = registry

    def issue(self, task_id: str) -> str:
        """Return a fresh raw token for ``task_id`` and store only its digest.

        Issuing again rotates the token: the previous one is invalidated.
        """
        token = secrets.token_urlsafe(TOKEN_BYTES)
        self._registry.store_worker_token(task_id, _digest(token))
        return token

    def verify(self, task_id: str, token: object) -> bool:
        """Return whether ``token`` is the current token for ``task_id``."""
        if not isinstance(task_id, str) or not task_id:
            return False
        if not isinstance(token, str) or not token:
            return False
        stored = self._registry.worker_token_hash(task_id)
        if stored is None:
            return False
        return hmac.compare_digest(stored, _digest(token))


class WorkerTaskTools:
    """The tools a worker process is allowed to call."""

    def __init__(self, service: TaskService, tokens: WorkerTokenStore) -> None:
        self._service = service
        self._tokens = tokens

    def neo_task_update(
        self,
        task_id: str,
        token: Optional[str],
        state: Union[TaskState, str],
        expected_version: int,
        progress: Optional[float] = None,
        detail: Optional[str] = None,
        summary: Optional[str] = None,
        evidence: Optional[Evidence] = None,
    ) -> dict:
        """Apply one authenticated worker transition and optional progress event.

        The token digest is checked inside the state-change transaction. A token
        rotated before or during this call therefore cannot commit stale work.
        """
        if not isinstance(token, str) or not token:
            raise InvalidWorkerTokenError(
                "worker token rejected for task %s" % task_id
            )
        event_extra = {}
        if progress is not None:
            event_extra["progress"] = progress
        if detail is not None:
            event_extra["detail"] = detail
        return self._service.transition(
            task_id,
            state,
            expected_version=expected_version,
            summary=summary,
            evidence=evidence,
            event_extra=event_extra or None,
            worker_token_digest=_digest(token),
        )

    def worker_task_update(
        self,
        task_id: str,
        token: Optional[str],
        target: Union[TaskState, str],
        expected_version: int,
        summary: Optional[str] = None,
        evidence: Optional[Evidence] = None,
    ) -> dict:
        """Compatibility wrapper for the original internal worker update name."""
        return self.neo_task_update(
            task_id,
            token,
            state=target,
            expected_version=expected_version,
            summary=summary,
            evidence=evidence,
        )


__all__ = [
    "InvalidWorkerTokenError",
    "TOKEN_BYTES",
    "WorkerTaskTools",
    "WorkerTokenStore",
    "WorkerToolError",
]
