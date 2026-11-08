"""Pure task-state rules for NEO.

The state machine owns one question: given a task's current state, which state
may it enter next? It performs no I/O, keeps no clock and touches no database,
so the same rules serve the SQLite task service, the worker bridge and the API.

The transition table is the one approved in the implementation plan, section
4.2, with one deliberate amendment: ``needs_approval`` and ``needs_input`` may
fall to ``failed`` or ``unknown``. A worker can die while its task waits for a
human, and a dead worker must be recorded honestly instead of leaving the row
pretending to be alive. Those edges never reach ``completed``.

Two further rules need more than a table:

* Terminal states are immutable. Retrying is a new child task, never a
  reopening of the finished one.
* ``Completed`` means the executor finished *and* the expected results were
  verified. A process receipt is evidence that a process ended; on its own it
  can never complete a task.

Callers hold the task row and its version; this module only validates the edge.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Iterable, Mapping, Optional, Tuple, Union


class TaskState(str, Enum):
    """Every state a NEO task may occupy."""

    QUEUED = "queued"
    STARTING = "starting"
    RUNNING = "running"
    WAITING = "waiting"
    NEEDS_APPROVAL = "needs_approval"
    NEEDS_INPUT = "needs_input"
    CANCELLING = "cancelling"
    UNKNOWN = "unknown"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


TERMINAL_STATES = frozenset(
    {TaskState.COMPLETED, TaskState.FAILED, TaskState.CANCELLED}
)
"""States a task leaves only by being replaced with a fresh child task."""


ALLOWED_TRANSITIONS: Mapping[TaskState, "frozenset[TaskState]"] = MappingProxyType(
    {
        TaskState.QUEUED: frozenset(
            {TaskState.STARTING, TaskState.CANCELLED, TaskState.FAILED}
        ),
        TaskState.STARTING: frozenset(
            {TaskState.RUNNING, TaskState.QUEUED, TaskState.FAILED, TaskState.UNKNOWN}
        ),
        TaskState.RUNNING: frozenset(
            {
                TaskState.WAITING,
                TaskState.NEEDS_APPROVAL,
                TaskState.NEEDS_INPUT,
                TaskState.COMPLETED,
                TaskState.FAILED,
                TaskState.CANCELLING,
            }
        ),
        TaskState.WAITING: frozenset(
            {TaskState.RUNNING, TaskState.FAILED, TaskState.CANCELLED}
        ),
        TaskState.NEEDS_APPROVAL: frozenset(
            {
                TaskState.RUNNING,
                TaskState.QUEUED,
                TaskState.CANCELLED,
                TaskState.FAILED,
                TaskState.UNKNOWN,
            }
        ),
        TaskState.NEEDS_INPUT: frozenset(
            {
                TaskState.RUNNING,
                TaskState.QUEUED,
                TaskState.CANCELLED,
                TaskState.FAILED,
                TaskState.UNKNOWN,
            }
        ),
        TaskState.CANCELLING: frozenset(
            {TaskState.CANCELLED, TaskState.FAILED, TaskState.UNKNOWN}
        ),
        TaskState.UNKNOWN: frozenset(
            {TaskState.RUNNING, TaskState.FAILED, TaskState.CANCELLED}
        ),
        TaskState.COMPLETED: frozenset(),
        TaskState.FAILED: frozenset(),
        TaskState.CANCELLED: frozenset(),
    }
)


class StateMachineError(ValueError):
    """Base class for a refused state change."""


class UnknownStateError(StateMachineError):
    """The supplied state name is not part of the vocabulary."""


class InvalidTransitionError(StateMachineError):
    """The edge does not exist in the transition table."""


class TerminalStateError(InvalidTransitionError):
    """The source state is terminal and therefore immutable."""


class IncompleteEvidenceError(StateMachineError):
    """A completion was attempted without executor confirmation and verified results."""


@dataclass(frozen=True)
class CompletionEvidence:
    """Proof that a task may enter ``completed``.

    ``executor_confirmed`` must be set from the executor's own terminal result,
    never inferred from a process exit. Every name in ``expected_artifacts``
    must appear in ``verified_artifacts``; extra verified artifacts are allowed.
    """

    executor_confirmed: bool
    expected_artifacts: Iterable[str] = ()
    verified_artifacts: Iterable[str] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "expected_artifacts", tuple(self.expected_artifacts))
        object.__setattr__(self, "verified_artifacts", tuple(self.verified_artifacts))

    def missing_artifacts(self) -> Tuple[str, ...]:
        """Expected artifacts that were not verified, sorted for stable messages."""
        verified = set(self.verified_artifacts)
        missing = {name for name in self.expected_artifacts if name not in verified}
        return tuple(sorted(missing))


@dataclass(frozen=True)
class ProcessReceipt:
    """A supervisor's record that a worker process ended.

    Existence of a receipt says nothing about whether the work succeeded, so the
    state machine refuses it as completion evidence.
    """

    pid: int
    exit_code: int
    receipt_path: Optional[str] = None


Evidence = Union[CompletionEvidence, ProcessReceipt]


def _coerce_state(value: object, *, field: str) -> TaskState:
    """Return a ``TaskState`` for a state or state name, or refuse the value."""
    if isinstance(value, TaskState):
        return value
    if isinstance(value, str):
        try:
            return TaskState(value)
        except ValueError:
            raise UnknownStateError(
                f"unknown task {field}: {value!r}"
            ) from None
    raise UnknownStateError(
        f"task {field} must be a state name, got {type(value).__name__}"
    )


def allowed_targets(state: Union[TaskState, str]) -> "frozenset[TaskState]":
    """Return the states reachable from ``state`` (empty for terminal states)."""
    return ALLOWED_TRANSITIONS[_coerce_state(state, field="state")]


def _validate_completion_evidence(evidence: Optional[Evidence]) -> None:
    if evidence is None:
        raise IncompleteEvidenceError(
            "completed requires completion evidence: executor terminal evidence "
            "plus verification of the expected results"
        )
    if isinstance(evidence, ProcessReceipt):
        raise IncompleteEvidenceError(
            "a process receipt cannot complete a task: "
            f"exit code {evidence.exit_code} records that the process ended, "
            "not that the executor succeeded"
        )
    if not isinstance(evidence, CompletionEvidence):
        raise IncompleteEvidenceError(
            "completion evidence must be a CompletionEvidence instance, "
            f"got {type(evidence).__name__}"
        )
    if not evidence.executor_confirmed:
        raise IncompleteEvidenceError(
            "completion requires executor terminal evidence; "
            "the executor has not confirmed that the work succeeded"
        )
    missing = evidence.missing_artifacts()
    if missing:
        raise IncompleteEvidenceError(
            "completion requires every expected artifact to be verified; "
            f"missing expected artifacts: {', '.join(missing)}"
        )


def transition(
    current: Union[TaskState, str],
    target: Union[TaskState, str],
    *,
    evidence: Optional[Evidence] = None,
) -> TaskState:
    """Validate one edge and return the resulting state.

    ``evidence`` is consulted only when ``target`` is ``completed``; it is
    ignored for every other transition.
    """
    source = _coerce_state(current, field="current state")
    destination = _coerce_state(target, field="target state")

    if source in TERMINAL_STATES:
        raise TerminalStateError(
            f"{source.value} is terminal and cannot transition to {destination.value}"
        )
    if destination not in ALLOWED_TRANSITIONS[source]:
        raise InvalidTransitionError(
            f"undefined transition {source.value} -> {destination.value}"
        )
    if destination is TaskState.COMPLETED:
        _validate_completion_evidence(evidence)
    return destination


def can_transition(
    current: Union[TaskState, str],
    target: Union[TaskState, str],
    *,
    evidence: Optional[Evidence] = None,
) -> bool:
    """Return whether the edge is permitted; unknown state names still raise."""
    try:
        transition(current, target, evidence=evidence)
    except UnknownStateError:
        raise
    except StateMachineError:
        return False
    return True


__all__ = [
    "ALLOWED_TRANSITIONS",
    "TERMINAL_STATES",
    "CompletionEvidence",
    "IncompleteEvidenceError",
    "InvalidTransitionError",
    "ProcessReceipt",
    "StateMachineError",
    "TaskState",
    "TerminalStateError",
    "UnknownStateError",
    "allowed_targets",
    "can_transition",
    "transition",
]
