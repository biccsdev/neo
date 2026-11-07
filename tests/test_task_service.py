"""Task 5 — the task service: one durable event per accepted state change.

Every accepted create or transition must leave exactly one normalized event in
the same SQLite transaction as the state change. A refused change must leave
nothing behind. Events are read per conversation and never leak across one.
"""

from __future__ import annotations

import json
import sys
import tempfile
import threading
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from events import TaskService, event_name_for  # noqa: E402
from registry import Registry, VersionConflictError  # noqa: E402
from state_machine import (  # noqa: E402
    CompletionEvidence,
    IncompleteEvidenceError,
    InvalidTransitionError,
    TaskState,
    UnknownStateError,
)


class TaskServiceTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tempdir.name) / "registry.db"
        self.registry = Registry(path=self.db_path, max_parallel=4)
        self.service = TaskService(self.registry)

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    # -- helpers ---------------------------------------------------------

    def create(self, conversation_key: str = "conversation", summary=None, **options):
        return self.service.create_task(
            conversation_key=conversation_key,
            kind="coding",
            intent="do the work",
            summary=summary,
            **options,
        )

    def raw_events(self) -> list:
        return self.registry.events_since(0)

    def task_events(self, task_id: str) -> list:
        return [event for event in self.raw_events() if event["task_id"] == task_id]

    def payload(self, event: dict) -> dict:
        return json.loads(event["payload_json"])

    # -- creation --------------------------------------------------------

    def test_create_emits_exactly_one_task_created_event(self) -> None:
        task = self.create()

        events = self.task_events(task["id"])
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["kind"], "task.created")
        self.assertEqual(
            self.payload(events[0]),
            {
                "task_id": task["id"],
                "state": task["state"],
                "version": task["version"],
            },
        )
        self.assertEqual(events[0]["delivered"], 0)

    def test_create_summary_is_optional_and_recorded(self) -> None:
        with_summary = self.create(summary="Wiring the preview route")
        plain = self.create(summary=None)

        self.assertEqual(
            self.payload(self.task_events(with_summary["id"])[0])["summary"],
            "Wiring the preview route",
        )
        self.assertNotIn("summary", self.payload(self.task_events(plain["id"])[0]))

    def test_duplicate_submission_emits_no_second_created_event(self) -> None:
        first = self.create(dedupe_key="request-fingerprint")
        second = self.create(dedupe_key="request-fingerprint")

        self.assertEqual(first["id"], second["id"])
        self.assertEqual([event["kind"] for event in self.raw_events()], ["task.created"])

    def test_refused_create_leaves_neither_task_nor_event(self) -> None:
        with self.assertRaises(ValueError):
            self.registry.create_task(
                conversation_key="conversation",
                kind="coding",
                intent="never lands",
                event_kind=object(),
            )

        self.assertEqual(self.registry.list_tasks(), [])
        self.assertEqual(self.raw_events(), [])

    # -- transitions -----------------------------------------------------

    def test_transition_names_follow_the_resulting_state(self) -> None:
        updated = self.create(conversation_key="one")
        updated = self.service.transition(
            updated["id"], "running", expected_version=updated["version"]
        )

        completed = self.create(conversation_key="two")
        completed = self.service.transition(
            completed["id"], "running", expected_version=completed["version"]
        )
        self.service.transition(
            completed["id"],
            "completed",
            expected_version=completed["version"],
            evidence=CompletionEvidence(executor_confirmed=True),
        )

        failed = self.create(conversation_key="three")
        failed = self.service.transition(
            failed["id"], "running", expected_version=failed["version"]
        )
        self.service.transition(
            failed["id"], "failed", expected_version=failed["version"]
        )

        cancelled = self.create(conversation_key="four")
        cancelled = self.service.transition(
            cancelled["id"], "running", expected_version=cancelled["version"]
        )
        cancelled = self.service.transition(
            cancelled["id"], "cancelling", expected_version=cancelled["version"]
        )
        self.service.transition(
            cancelled["id"], "cancelled", expected_version=cancelled["version"]
        )

        self.assertEqual(
            [event["kind"] for event in self.task_events(updated["id"])],
            ["task.created", "task.updated"],
        )
        self.assertEqual(
            [event["kind"] for event in self.task_events(completed["id"])],
            ["task.created", "task.updated", "task.completed"],
        )
        self.assertEqual(
            [event["kind"] for event in self.task_events(failed["id"])],
            ["task.created", "task.updated", "task.failed"],
        )
        self.assertEqual(
            [event["kind"] for event in self.task_events(cancelled["id"])],
            ["task.created", "task.updated", "task.updated", "task.cancelled"],
        )

    def test_transition_payload_carries_state_version_and_summary(self) -> None:
        task = self.create()
        running = self.service.transition(
            task["id"],
            "running",
            expected_version=task["version"],
            summary="Executor started",
        )

        events = self.task_events(task["id"])
        self.assertEqual(len(events), 2)
        self.assertEqual(
            self.payload(events[-1]),
            {
                "task_id": task["id"],
                "state": running["state"],
                "version": running["version"],
                "summary": "Executor started",
            },
        )
        self.assertEqual(running["version"], task["version"] + 1)

    def test_invalid_transition_emits_nothing(self) -> None:
        task = self.create()

        with self.assertRaises(InvalidTransitionError):
            self.service.transition(
                task["id"], "waiting", expected_version=task["version"]
            )

        self.assertEqual([event["kind"] for event in self.task_events(task["id"])], ["task.created"])
        self.assertEqual(self.registry.get_task(task["id"]), task)

    def test_stale_version_emits_nothing(self) -> None:
        task = self.create()
        running = self.service.transition(
            task["id"], "running", expected_version=task["version"]
        )

        with self.assertRaises(VersionConflictError):
            self.service.transition(
                task["id"], "failed", expected_version=task["version"]
            )

        self.assertEqual(
            [event["kind"] for event in self.task_events(task["id"])],
            ["task.created", "task.updated"],
        )
        self.assertEqual(self.registry.get_task(task["id"]), running)

    def test_completion_without_evidence_emits_nothing(self) -> None:
        task = self.create()
        running = self.service.transition(
            task["id"], "running", expected_version=task["version"]
        )

        with self.assertRaises(IncompleteEvidenceError):
            self.service.transition(
                running["id"], "completed", expected_version=running["version"]
            )

        self.assertEqual(
            [event["kind"] for event in self.task_events(task["id"])],
            ["task.created", "task.updated"],
        )
        self.assertEqual(self.registry.get_task(task["id"])["state"], "running")

    def test_unknown_target_emits_nothing(self) -> None:
        task = self.create()

        with self.assertRaises(UnknownStateError):
            self.service.transition(
                task["id"], "not-a-state", expected_version=task["version"]
            )

        self.assertEqual(len(self.task_events(task["id"])), 1)
        self.assertEqual(self.registry.get_task(task["id"]), task)

    def test_state_and_event_are_written_in_one_transaction(self) -> None:
        task = self.create()

        with self.assertRaises(ValueError):
            self.registry.transition_task(
                task["id"],
                "running",
                expected_version=task["version"],
                event_kind=object(),
            )

        self.assertEqual(self.registry.get_task(task["id"]), task)
        self.assertEqual([event["kind"] for event in self.raw_events()], ["task.created"])

    def test_concurrent_transitions_emit_exactly_one_event(self) -> None:
        task = self.create()
        barrier = threading.Barrier(2)
        results = []
        errors = []

        def attempt() -> None:
            try:
                barrier.wait()
                results.append(
                    self.service.transition(
                        task["id"], "running", expected_version=task["version"]
                    )
                )
            except Exception as error:  # surfaced in the main thread below
                errors.append(error)

        threads = [threading.Thread(target=attempt) for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        self.assertEqual(len(results), 1)
        self.assertEqual(len(errors), 1)
        self.assertIsInstance(errors[0], VersionConflictError)
        self.assertEqual(
            [event["kind"] for event in self.task_events(task["id"])],
            ["task.created", "task.updated"],
        )

    # -- conversation scoping -------------------------------------------

    def test_events_are_scoped_to_the_conversation(self) -> None:
        mine = self.create(conversation_key="conversation-one")
        theirs = self.create(conversation_key="conversation-two")
        self.service.transition(
            theirs["id"], "running", expected_version=theirs["version"]
        )

        self.assertEqual(
            [event["task_id"] for event in self.service.events_since("conversation-one")],
            [mine["id"]],
        )
        self.assertEqual(
            len(self.service.events_since("conversation-two")),
            2,
        )
        self.assertEqual(self.service.events_since("conversation-three"), [])

    def test_task_events_are_scoped_to_the_conversation(self) -> None:
        theirs = self.create(conversation_key="conversation-two")

        self.assertEqual(self.service.task_events("conversation-one", theirs["id"]), [])
        self.assertEqual(
            len(self.service.task_events("conversation-two", theirs["id"])), 1
        )

    def test_conversation_key_is_required_for_scoped_reads(self) -> None:
        for key in ("", None, 7):
            with self.subTest(key=key):
                with self.assertRaises(ValueError):
                    self.service.events_since(key)
        with self.assertRaises(ValueError):
            self.service.events_since("conversation", since=-1)

    def test_event_cursor_returns_only_newer_events(self) -> None:
        first = self.create()
        second = self.create()

        newest = self.service.events_since("conversation", since=1)
        self.assertEqual([event["task_id"] for event in newest], [second["id"]])
        self.assertEqual(len(self.service.events_since("conversation", since=newest[0]["id"])), 0)

    def test_scoped_reads_decode_the_payload(self) -> None:
        task = self.create(summary="decoded")

        event = self.service.events_since("conversation")[0]
        self.assertEqual(event["payload"]["task_id"], task["id"])
        self.assertEqual(event["payload"]["summary"], "decoded")
        self.assertIn("payload_json", event)


class EventNameTests(unittest.TestCase):
    def test_only_terminal_states_get_their_own_name(self) -> None:
        self.assertEqual(event_name_for("completed"), "task.completed")
        self.assertEqual(event_name_for(TaskState.FAILED), "task.failed")
        self.assertEqual(event_name_for("cancelled"), "task.cancelled")
        for state in TaskState:
            if state in {TaskState.COMPLETED, TaskState.FAILED, TaskState.CANCELLED}:
                continue
            with self.subTest(state=state):
                self.assertEqual(event_name_for(state), "task.updated")

    def test_unknown_state_name_keeps_the_state_machine_error(self) -> None:
        with self.assertRaises(UnknownStateError):
            event_name_for("teleporting")


if __name__ == "__main__":
    unittest.main()
