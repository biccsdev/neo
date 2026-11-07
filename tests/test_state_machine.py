"""Task 3 — the pure-Python NEO task state machine.

The state machine is the only authority on which task states may follow which.
It carries no I/O, no timestamps and no persistence, so the rules can be proven
here once and reused by the SQLite task service later.

Transition table from the implementation plan, section 4.2.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from state_machine import (  # noqa: E402  (path setup must run first)
    ALLOWED_TRANSITIONS,
    TERMINAL_STATES,
    CompletionEvidence,
    IncompleteEvidenceError,
    InvalidTransitionError,
    ProcessReceipt,
    StateMachineError,
    TaskState,
    TerminalStateError,
    UnknownStateError,
    allowed_targets,
    can_transition,
    transition,
)

QUEUED = TaskState.QUEUED
STARTING = TaskState.STARTING
RUNNING = TaskState.RUNNING
WAITING = TaskState.WAITING
NEEDS_APPROVAL = TaskState.NEEDS_APPROVAL
NEEDS_INPUT = TaskState.NEEDS_INPUT
CANCELLING = TaskState.CANCELLING
UNKNOWN = TaskState.UNKNOWN
COMPLETED = TaskState.COMPLETED
FAILED = TaskState.FAILED
CANCELLED = TaskState.CANCELLED

ALL_STATES = tuple(TaskState)

# Written out independently of the implementation so the test, not the code,
# defines the contract.
EXPECTED_TRANSITIONS: dict[TaskState, set[TaskState]] = {
    QUEUED: {STARTING, CANCELLED, FAILED},
    STARTING: {RUNNING, QUEUED, FAILED, UNKNOWN},
    RUNNING: {WAITING, NEEDS_APPROVAL, NEEDS_INPUT, COMPLETED, FAILED, CANCELLING},
    WAITING: {RUNNING, FAILED, CANCELLED},
    NEEDS_APPROVAL: {RUNNING, QUEUED, CANCELLED},
    NEEDS_INPUT: {RUNNING, QUEUED, CANCELLED},
    CANCELLING: {CANCELLED, FAILED, UNKNOWN},
    UNKNOWN: {RUNNING, FAILED, CANCELLED},
    COMPLETED: set(),
    FAILED: set(),
    CANCELLED: set(),
}

EXPECTED_TERMINAL_STATES = frozenset({COMPLETED, FAILED, CANCELLED})

EXPECTED_TRANSITION_PAIRS = {
    (source, target)
    for source, targets in EXPECTED_TRANSITIONS.items()
    for target in targets
}

ALL_PAIRS = tuple((source, target) for source in ALL_STATES for target in ALL_STATES)

CONFIRMED_EVIDENCE = CompletionEvidence(
    executor_confirmed=True,
    expected_artifacts=("model.blend",),
    verified_artifacts=("model.blend",),
)

PROCESS_RECEIPT = ProcessReceipt(
    pid=4242,
    exit_code=0,
    receipt_path="/tmp/neo-task.receipt.json",  # noqa: S108 - test fixture only
)


class StateVocabularyTests(unittest.TestCase):
    def test_states_match_the_plan(self) -> None:
        self.assertEqual(
            {state.value for state in ALL_STATES},
            {
                "queued",
                "starting",
                "running",
                "waiting",
                "needs_approval",
                "needs_input",
                "cancelling",
                "unknown",
                "completed",
                "failed",
                "cancelled",
            },
        )

    def test_terminal_states_are_exactly_the_finished_outcomes(self) -> None:
        self.assertEqual(set(TERMINAL_STATES), set(EXPECTED_TERMINAL_STATES))
        self.assertTrue(EXPECTED_TERMINAL_STATES.issubset(set(ALL_STATES)))

    def test_published_transition_table_matches_the_plan(self) -> None:
        published = {
            source: set(targets) for source, targets in ALLOWED_TRANSITIONS.items()
        }
        self.assertEqual(published, EXPECTED_TRANSITIONS)

    def test_unknown_state_name_is_rejected(self) -> None:
        with self.assertRaises(UnknownStateError) as raised:
            transition("exploded", RUNNING)
        self.assertIn("exploded", str(raised.exception))


class AllowedTransitionTests(unittest.TestCase):
    def test_every_allowed_transition_succeeds(self) -> None:
        self.assertTrue(EXPECTED_TRANSITION_PAIRS, "the contract table is empty")

        for source, target in sorted(
            EXPECTED_TRANSITION_PAIRS, key=lambda pair: (pair[0].value, pair[1].value)
        ):
            with self.subTest(source=source.value, target=target.value):
                evidence = CONFIRMED_EVIDENCE if target is COMPLETED else None
                self.assertTrue(can_transition(source, target, evidence=evidence))
                self.assertIs(
                    transition(source, target, evidence=evidence),
                    target,
                    "transition() returns the resulting state",
                )

    def test_allowed_targets_match_the_plan_per_state(self) -> None:
        for state in ALL_STATES:
            with self.subTest(state=state.value):
                self.assertEqual(
                    set(allowed_targets(state)),
                    EXPECTED_TRANSITIONS[state],
                )

    def test_allowed_transitions_accept_plain_strings(self) -> None:
        self.assertIs(transition("queued", "starting"), STARTING)
        self.assertEqual(
            {state.value for state in allowed_targets("running")},
            {state.value for state in EXPECTED_TRANSITIONS[RUNNING]},
        )


class UndefinedTransitionTests(unittest.TestCase):
    def test_every_undefined_transition_raises(self) -> None:
        undefined_pairs = [pair for pair in ALL_PAIRS if pair not in EXPECTED_TRANSITION_PAIRS]
        self.assertEqual(
            len(undefined_pairs),
            len(ALL_PAIRS) - len(EXPECTED_TRANSITION_PAIRS),
            "undefined transitions are the complement of the contract table",
        )

        for source, target in undefined_pairs:
            with self.subTest(source=source.value, target=target.value):
                if source in EXPECTED_TERMINAL_STATES:
                    expected_error = TerminalStateError
                else:
                    expected_error = InvalidTransitionError
                with self.assertRaises(expected_error) as raised:
                    transition(source, target)
                message = str(raised.exception)
                self.assertIn(source.value, message)
                self.assertIn(target.value, message)

    def test_self_transitions_are_undefined_for_non_terminal_states(self) -> None:
        for state in ALL_STATES:
            if state in EXPECTED_TERMINAL_STATES:
                continue
            with self.subTest(state=state.value):
                self.assertFalse(can_transition(state, state))
                with self.assertRaises(InvalidTransitionError):
                    transition(state, state)

    def test_can_transition_reports_false_instead_of_raising(self) -> None:
        self.assertFalse(can_transition(QUEUED, RUNNING))
        self.assertFalse(can_transition(UNKNOWN, WAITING))
        self.assertFalse(can_transition(COMPLETED, RUNNING))

    def test_can_transition_still_rejects_unknown_names(self) -> None:
        with self.assertRaises(UnknownStateError):
            can_transition(RUNNING, "teleported")

    def test_errors_are_state_machine_value_errors(self) -> None:
        self.assertTrue(issubclass(InvalidTransitionError, StateMachineError))
        self.assertTrue(issubclass(TerminalStateError, InvalidTransitionError))
        self.assertTrue(issubclass(UnknownStateError, StateMachineError))
        self.assertTrue(issubclass(IncompleteEvidenceError, StateMachineError))
        self.assertTrue(issubclass(StateMachineError, ValueError))


class TerminalStateTests(unittest.TestCase):
    def test_terminal_states_have_no_allowed_targets(self) -> None:
        for state in EXPECTED_TERMINAL_STATES:
            with self.subTest(state=state.value):
                self.assertEqual(allowed_targets(state), frozenset())

    def test_terminal_states_are_immutable(self) -> None:
        for state in EXPECTED_TERMINAL_STATES:
            for target in ALL_STATES:
                with self.subTest(source=state.value, target=target.value):
                    with self.assertRaises(TerminalStateError) as raised:
                        transition(
                            state,
                            target,
                            evidence=CONFIRMED_EVIDENCE if target is COMPLETED else None,
                        )
                    self.assertIn("terminal", str(raised.exception))
                    self.assertIn(state.value, str(raised.exception))

    def test_terminal_state_cannot_be_reentered(self) -> None:
        for state in EXPECTED_TERMINAL_STATES:
            with self.subTest(state=state.value):
                with self.assertRaises(TerminalStateError):
                    transition(state, state)


class CompletionEvidenceTests(unittest.TestCase):
    def test_completion_requires_evidence(self) -> None:
        with self.assertRaises(IncompleteEvidenceError) as raised:
            transition(RUNNING, COMPLETED)
        self.assertIn("evidence", str(raised.exception))

    def test_completion_succeeds_with_confirmed_executor_and_verified_artifacts(self) -> None:
        evidence = CompletionEvidence(
            executor_confirmed=True,
            expected_artifacts=("scene.blend", "render.png"),
            verified_artifacts=("render.png", "scene.blend"),
        )
        self.assertIs(transition(RUNNING, COMPLETED, evidence=evidence), COMPLETED)

    def test_completion_without_expected_artifacts_needs_only_executor_evidence(self) -> None:
        evidence = CompletionEvidence(
            executor_confirmed=True,
            expected_artifacts=(),
            verified_artifacts=(),
        )
        self.assertIs(transition(RUNNING, COMPLETED, evidence=evidence), COMPLETED)

    def test_completion_rejects_unconfirmed_executor(self) -> None:
        evidence = CompletionEvidence(
            executor_confirmed=False,
            expected_artifacts=("model.blend",),
            verified_artifacts=("model.blend",),
        )
        with self.assertRaises(IncompleteEvidenceError) as raised:
            transition(RUNNING, COMPLETED, evidence=evidence)
        self.assertIn("executor", str(raised.exception))

    def test_completion_rejects_missing_expected_artifact(self) -> None:
        evidence = CompletionEvidence(
            executor_confirmed=True,
            expected_artifacts=("scene.blend", "render.png"),
            verified_artifacts=("scene.blend",),
        )
        with self.assertRaises(IncompleteEvidenceError) as raised:
            transition(RUNNING, COMPLETED, evidence=evidence)
        message = str(raised.exception)
        self.assertIn("render.png", message)
        self.assertIn("expected", message)

    def test_process_receipt_alone_cannot_complete_a_task(self) -> None:
        with self.assertRaises(IncompleteEvidenceError) as raised:
            transition(RUNNING, COMPLETED, evidence=PROCESS_RECEIPT)
        message = str(raised.exception)
        self.assertIn("process receipt", message)
        self.assertIn("exit code", message)

    def test_process_receipt_cannot_be_padded_into_completion(self) -> None:
        # A receipt is refused on its own terms, not merely because it lacks
        # artifact fields, so no amount of surrounding data can promote it.
        receipt = ProcessReceipt(
            pid=4242,
            exit_code=0,
            receipt_path="/tmp/neo-task.receipt.json",  # noqa: S108 - test fixture only
        )
        with self.assertRaises(IncompleteEvidenceError):
            transition(RUNNING, COMPLETED, evidence=receipt)
        self.assertFalse(can_transition(RUNNING, COMPLETED, evidence=receipt))

    def test_process_receipt_is_rejected_as_an_arbitrary_object(self) -> None:
        with self.assertRaises(IncompleteEvidenceError) as raised:
            transition(RUNNING, COMPLETED, evidence={"exit_code": 0, "pid": 4242})  # type: ignore[arg-type]
        self.assertIn("evidence", str(raised.exception))

    def test_evidence_is_ignored_for_non_completion_targets(self) -> None:
        self.assertIs(
            transition(RUNNING, WAITING, evidence=PROCESS_RECEIPT),
            WAITING,
        )


if __name__ == "__main__":
    unittest.main()
