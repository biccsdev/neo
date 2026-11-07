"""NEO task service: the only writer of task state and its public events.

The registry owns durability; this service owns the rule that a state change
and its public event are one indivisible fact. Every accepted create leaves
exactly one ``task.created`` event and every accepted transition leaves exactly
one event named after the state the task entered. A refused change leaves
nothing, because the event row is written inside the registry's own
transaction.

Reads are conversation-scoped: a caller that knows only its conversation key
can never observe another conversation's tasks or events.
"""

from __future__ import annotations

import json
from types import MappingProxyType
from typing import Optional, Union

from registry import Registry, RegistryError
from state_machine import Evidence, TaskState, UnknownStateError

TASK_CREATED = "task.created"
TASK_UPDATED = "task.updated"

TERMINAL_EVENT_NAMES = MappingProxyType(
    {
        TaskState.COMPLETED: "task.completed",
        TaskState.FAILED: "task.failed",
        TaskState.CANCELLED: "task.cancelled",
    }
)
"""States that carry their own public event name; every other state is an update."""


def event_name_for(target: Union[TaskState, str]) -> str:
    """Return the normalized event name for entering ``target``."""
    if isinstance(target, TaskState):
        state = target
    elif isinstance(target, str):
        try:
            state = TaskState(target)
        except ValueError:
            raise UnknownStateError(
                "unknown task target state: %r" % target
            ) from None
    else:
        raise UnknownStateError(
            "task target state must be a state name, got %s" % type(target).__name__
        )
    return TERMINAL_EVENT_NAMES.get(state, TASK_UPDATED)


def _decode_event(row: dict) -> dict:
    """Return an event row with its JSON payload decoded for callers."""
    event = dict(row)
    try:
        payload = json.loads(event.get("payload_json") or "{}")
    except ValueError as error:
        raise RegistryError("event payload is not valid JSON: %s" % error) from error
    if not isinstance(payload, dict):
        raise RegistryError("event payload must be a JSON object")
    event["payload"] = payload
    return event


class TaskService:
    """Apply the task rules on top of the registry's transactions."""

    def __init__(self, registry: Registry) -> None:
        self._registry = registry

    @property
    def registry(self) -> Registry:
        return self._registry

    def create_task(
        self,
        *,
        conversation_key: str,
        kind: str,
        intent: str,
        workdir: Optional[str] = None,
        task_id: Optional[str] = None,
        dedupe_key: Optional[str] = None,
        resource_key: Optional[str] = None,
        summary: Optional[str] = None,
    ) -> dict:
        """Create a task and its ``task.created`` event in one transaction."""
        return self._registry.create_task(
            conversation_key=conversation_key,
            kind=kind,
            intent=intent,
            workdir=workdir,
            task_id=task_id,
            dedupe_key=dedupe_key,
            resource_key=resource_key,
            event_kind=TASK_CREATED,
            event_summary=summary,
        )

    def transition(
        self,
        task_id: str,
        target: Union[TaskState, str],
        *,
        expected_version: int,
        evidence: Optional[Evidence] = None,
        summary: Optional[str] = None,
    ) -> dict:
        """Validate, persist and announce one transition.

        An invalid edge, a stale version or missing completion evidence raises
        before anything is written, so no event is emitted.
        """
        return self._registry.transition_task(
            task_id,
            target,
            expected_version=expected_version,
            evidence=evidence,
            event_kind=event_name_for(target),
            event_summary=summary,
        )

    def get_task(self, task_id: str) -> Optional[dict]:
        return self._registry.get_task(task_id)

    def events_since(self, conversation_key: str, since: int = 0) -> list:
        """Return this conversation's events newer than the cursor."""
        return [
            _decode_event(row)
            for row in self._registry.events_for_conversation(
                conversation_key, since
            )
        ]

    def task_events(
        self, conversation_key: str, task_id: str, since: int = 0
    ) -> list:
        """Return one task's events, provided the task is in this conversation."""
        return [
            _decode_event(row)
            for row in self._registry.events_for_conversation(
                conversation_key, since, task_id=task_id
            )
        ]


__all__ = [
    "TASK_CREATED",
    "TASK_UPDATED",
    "TERMINAL_EVENT_NAMES",
    "TaskService",
    "event_name_for",
]
