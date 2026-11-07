"""Worker-facing tool boundary for NEO.

A worker process holds ``NEO_TASK_ID`` and a random ``NEO_TASK_TOKEN``. The
plugin stores only a SHA-256 digest of that token, so a database read never
yields a usable credential. Every worker call verifies the token before it
reaches the task service, and a rejected token changes nothing.

This module deliberately knows nothing about process management, supervisors,
previews or artifacts; it is only the authorization seam and the update call.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
from typing import Optional, Union

from events import TaskService
from registry import Registry
from state_machine import Evidence, TaskState

TOKEN_BYTES = 32


class WorkerToolError(RuntimeError):
    """Base class for a refused worker call."""


class InvalidWorkerTokenError(WorkerToolError):
    """The presented worker token is missing, wrong, or not issued for this task.

    One error type for every failure keeps the boundary from confirming whether
    a task exists or whether a token was ever issued.
    """


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

    def worker_task_update(
        self,
        task_id: str,
        token: Optional[str],
        target: Union[TaskState, str],
        expected_version: int,
        summary: Optional[str] = None,
        evidence: Optional[Evidence] = None,
    ) -> dict:
        """Verify the worker's token, then apply exactly one transition.

        The token is checked before the service is touched, so a wrong or
        missing token cannot advance the task or leave an event behind.
        """
        if not self._tokens.verify(task_id, token):
            raise InvalidWorkerTokenError(
                "worker token rejected for task %s" % task_id
            )
        return self._service.transition(
            task_id,
            target,
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
