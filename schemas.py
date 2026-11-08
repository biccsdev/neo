"""OpenAI-style function schemas for the NEO worker bridge.

Workers are handed a task id and a per-task token. These schemas are the
machine-readable half of that contract: every tool requires ``task_id`` and
``token``, and none of them accepts a shell command, argv array or script,
because the bridge executes nothing on a worker's behalf.

**These schemas are the conversational half of NEO.** They are what the model
sees when a user asks NEO to start, inspect, refine, cancel or open a task. No
schema accepts a shell command, argv array or script: NEO builds every command
line itself from typed fields.

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


# -- conversational tools -----------------------------------------------------

_KIND = {
    "type": "string",
    "enum": ["auto", "coding", "blender", "browser", "general"],
    "description": (
        "Which kind of work this is. 'auto' infers it from the intent; coding "
        "routes to a coding CLI, blender and browser to a detached Hermes worker."
    ),
}
_TASK_REF = {
    "type": "string",
    "description": "The NEO task id, as returned by neo_submit_task or neo_task_list.",
}

_SUBMIT = {
    "type": "function",
    "function": {
        "name": "neo_submit_task",
        "description": (
            "Start a durable background task (coding, Blender, browser or general) "
            "and return its task id. Use this whenever the user asks for work that "
            "should continue outside the current chat turn. Equivalent active "
            "submissions are deduplicated."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "intent": {
                    "type": "string",
                    "description": "The task, in the user's own words, complete enough to work from alone.",
                },
                "kind": _KIND,
                "workdir": {
                    "type": "string",
                    "description": (
                        "Optional absolute directory the task should work in. Omit to "
                        "use NEO's own task workspace; a path NEO cannot contain is ignored."
                    ),
                },
                "expected_artifacts": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Optional relative paths of result files NEO must verify before "
                        "the task can complete."
                    ),
                },
                "capability_id": {
                    "type": "string",
                    "description": "Optional user-selected capability id for this task.",
                },
            },
            "required": ["intent"],
            "additionalProperties": False,
        },
    },
}

_STATUS = {
    "type": "function",
    "function": {
        "name": "neo_task_status",
        "description": (
            "Read one task's state, progress, error, artifacts and preview "
            "availability, in this conversation only. Use it to answer 'how is it "
            "going?' It never creates or changes a task. Accepts a task id or a "
            "distinctive phrase from the task's intent; an ambiguous reference "
            "returns one clarifying question with candidate ids."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "task_id": _TASK_REF,
                "intent": {
                    "type": "string",
                    "description": (
                        "A distinctive phrase from the task's intent, when the user did "
                        "not give an id. NEO resolves it only if exactly one task matches."
                    ),
                },
            },
            "required": [],
            "additionalProperties": False,
        },
    },
}

_LIST = {
    "type": "function",
    "function": {
        "name": "neo_task_list",
        "description": (
            "List this conversation's NEO tasks, newest first, optionally filtered by "
            "state or kind. Read-only: it never creates a task."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "state": {
                    "type": "string",
                    "enum": list(_STATE_NAMES),
                    "description": "Only tasks in this state.",
                },
                "kind": {
                    "type": "string",
                    "enum": ["coding", "blender", "browser", "general"],
                    "description": "Only tasks of this kind.",
                },
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 50,
                    "description": "Maximum number of tasks to return (default 10).",
                },
            },
            "required": [],
            "additionalProperties": False,
        },
    },
}

_REFINE = {
    "type": "function",
    "function": {
        "name": "neo_task_refine",
        "description": (
            "Send a follow-up to the exact executor session of an existing task, so it "
            "keeps its context. A completed or failed task gets a child task resumed on "
            "the same native session; an active Codex task queues into its thread. If "
            "the session id is missing, or Claude is still working, NEO refuses rather "
            "than starting a replacement session."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "task_id": _TASK_REF,
                "follow_up": {
                    "type": "string",
                    "description": "The refinement, in the user's own words.",
                },
            },
            "required": ["task_id", "follow_up"],
            "additionalProperties": False,
        },
    },
}

_FRAME_FOLLOW_UP = {
    "type": "function",
    "function": {
        "name": "neo_task_frame_follow_up",
        "description": (
            "Continue a Blender task: create a child Blender worker task whose prompt "
            "carries the previous task's intent and session context. It resumes the "
            "recorded Blender session when there is one and says so honestly when "
            "there is not."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "task_id": _TASK_REF,
                "follow_up": {
                    "type": "string",
                    "description": "The Blender change the user wants next.",
                },
            },
            "required": ["task_id", "follow_up"],
            "additionalProperties": False,
        },
    },
}

_CANCEL = {
    "type": "function",
    "function": {
        "name": "neo_task_cancel",
        "description": (
            "Cancel a running NEO task. Cancelling an already-terminal task returns it "
            "unchanged, so repeating the request is safe."
        ),
        "parameters": {
            "type": "object",
            "properties": {"task_id": _TASK_REF},
            "required": ["task_id"],
            "additionalProperties": False,
        },
    },
}

_OPEN = {
    "type": "function",
    "function": {
        "name": "neo_task_open",
        "description": (
            "Open the verified result of a completed task in this conversation. A task "
            "that is not completed, or whose expected artifacts do not verify, is "
            "skipped honestly rather than opened."
        ),
        "parameters": {
            "type": "object",
            "properties": {"task_id": _TASK_REF},
            "required": ["task_id"],
            "additionalProperties": False,
        },
    },
}

CONVERSATION_TOOL_SCHEMAS = (
    _SUBMIT,
    _STATUS,
    _LIST,
    _REFINE,
    _CANCEL,
    _FRAME_FOLLOW_UP,
    _OPEN,
)
"""Every conversational tool, in a stable order that mirrors the manifest."""

CONVERSATION_TOOL_NAMES = tuple(
    schema["function"]["name"] for schema in CONVERSATION_TOOL_SCHEMAS
)


def conversation_tool_schemas() -> list:
    """Return independent copies, so a caller cannot mutate the catalog."""
    return [copy.deepcopy(schema) for schema in CONVERSATION_TOOL_SCHEMAS]


def conversation_schema_for(name: str) -> dict:
    """Return one conversational schema by tool name."""
    for schema in CONVERSATION_TOOL_SCHEMAS:
        if schema["function"]["name"] == name:
            return copy.deepcopy(schema)
    raise KeyError(name)


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
    "CONVERSATION_TOOL_NAMES",
    "CONVERSATION_TOOL_SCHEMAS",
    "WORKER_TOOL_NAMES",
    "WORKER_TOOL_SCHEMAS",
    "conversation_schema_for",
    "conversation_tool_schemas",
    "schema_for",
    "worker_tool_schemas",
]
