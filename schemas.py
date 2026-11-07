"""OpenAI-style function schemas for the NEO worker bridge.

Workers are handed a task id and a per-task token. These schemas are the
machine-readable half of that contract: every tool requires ``task_id`` and
``token``, and none of them accepts a shell command, argv array or script,
because the bridge executes nothing on a worker's behalf.

The schemas are plain data so they can be served to any OpenAI-compatible
executor without importing Hermes.
"""

from __future__ import annotations

import copy
from types import MappingProxyType

from state_machine import TaskState

_STATE_NAMES = tuple(state.value for state in TaskState)

_TASK_ID = {
    "type": "string",
    "description": "The id of the caller's own task, taken from NEO_TASK_ID.",
}
_TOKEN = {
    "type": "string",
    "description": "The caller's per-task secret, taken from NEO_TASK_TOKEN.",
}

_BASE_PROPERTIES = MappingProxyType({"task_id": _TASK_ID, "token": _TOKEN})

_UPDATED = {
    "type": "function",
    "function": {
        "name": "neo_task_update",
        "description": (
            "Report progress or enter a new lifecycle state for the caller's own "
            "task. The state machine decides whether the edge is allowed."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "task_id": _TASK_ID,
                "token": _TOKEN,
                "state": {
                    "type": "string",
                    "enum": list(_STATE_NAMES),
                    "description": "The state this task should enter.",
                },
                "expected_version": {
                    "type": "integer",
                    "minimum": 0,
                    "description": (
                        "Optional optimistic-concurrency guard. Omit to use the "
                        "task's current version."
                    ),
                },
                "progress": {
                    "type": "number",
                    "description": "Optional completion fraction or percentage.",
                },
                "detail": {
                    "type": "string",
                    "description": "Optional short human-readable progress detail.",
                },
                "summary": {
                    "type": "string",
                    "description": "Optional summary recorded with the event.",
                },
            },
            "required": ["task_id", "token", "state"],
            "additionalProperties": False,
        },
    },
}

_COMPLETED = {
    "type": "function",
    "function": {
        "name": "neo_task_complete",
        "description": (
            "Complete the caller's own task. NEO verifies the expected artifacts "
            "beneath the task workdir before accepting completion; an assertion "
            "from the worker is never enough."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "task_id": _TASK_ID,
                "token": _TOKEN,
                "expected_artifacts": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Optional relative paths of result files produced in the "
                        "task workdir. Verified and persisted before completion."
                    ),
                },
                "summary": {
                    "type": "string",
                    "description": "Optional one-line result summary.",
                },
            },
            "required": ["task_id", "token"],
            "additionalProperties": False,
        },
    },
}

_PUBLISHED = {
    "type": "function",
    "function": {
        "name": "neo_task_publish_preview",
        "description": (
            "Copy one image from the caller's task workdir into that task's own "
            "preview directory. Only png, jpg, jpeg and webp are accepted, and "
            "the configured byte cap applies."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "task_id": _TASK_ID,
                "token": _TOKEN,
                "source_path": {
                    "type": "string",
                    "description": (
                        "Path of a regular image file inside the task workdir, "
                        "relative to that workdir or absolute beneath it."
                    ),
                },
            },
            "required": ["task_id", "token", "source_path"],
            "additionalProperties": False,
        },
    },
}

_HEARTBEAT = {
    "type": "function",
    "function": {
        "name": "neo_task_heartbeat",
        "description": (
            "Record that the worker is still alive. Advances only the task "
            "heartbeat clock; it never changes lifecycle state."
        ),
        "parameters": {
            "type": "object",
            "properties": {"task_id": _TASK_ID, "token": _TOKEN},
            "required": ["task_id", "token"],
            "additionalProperties": False,
        },
    },
}

_GET = {
    "type": "function",
    "function": {
        "name": "neo_task_get",
        "description": (
            "Read the caller's own task. Conversation routing data and token "
            "digests are never returned."
        ),
        "parameters": {
            "type": "object",
            "properties": {"task_id": _TASK_ID, "token": _TOKEN},
            "required": ["task_id", "token"],
            "additionalProperties": False,
        },
    },
}

WORKER_TOOL_SCHEMAS = (_UPDATED, _COMPLETED, _PUBLISHED, _HEARTBEAT, _GET)
"""Every bridge tool the worker prompt may offer, in a stable order."""

WORKER_TOOL_NAMES = tuple(schema["function"]["name"] for schema in WORKER_TOOL_SCHEMAS)


def worker_tool_schemas() -> list:
    """Return independent copies, so a caller cannot mutate the catalog."""
    return [copy.deepcopy(schema) for schema in WORKER_TOOL_SCHEMAS]


def schema_for(name: str) -> dict:
    """Return one bridge schema by tool name."""
    for schema in WORKER_TOOL_SCHEMAS:
        if schema["function"]["name"] == name:
            return copy.deepcopy(schema)
    raise KeyError(name)


__all__ = [
    "WORKER_TOOL_NAMES",
    "WORKER_TOOL_SCHEMAS",
    "schema_for",
    "worker_tool_schemas",
]
