"""Task 4 — durable SQLite task registry."""

from __future__ import annotations

import sqlite3
import sys
import tempfile
import threading
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from registry import Registry, VersionConflictError  # noqa: E402
from state_machine import (  # noqa: E402
    CompletionEvidence,
    IncompleteEvidenceError,
    InvalidTransitionError,
)


class RegistryTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tempdir.name) / "registry.db"
        self.registry = Registry(path=self.db_path, max_parallel=2)

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def test_schema_migrates_idempotently(self) -> None:
        self.registry.migrate()
        reopened = Registry(path=self.db_path, max_parallel=2)
        reopened.migrate()

        with sqlite3.connect(self.db_path) as connection:
            tables = {
                row[0]
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table'"
                )
            }
            self.assertTrue(
                {"tasks", "task_events", "workers", "capability_consents"}.issubset(
                    tables
                )
            )
            self.assertEqual(connection.execute("PRAGMA user_version").fetchone()[0], 1)
            self.assertEqual(connection.execute("PRAGMA journal_mode").fetchone()[0], "wal")

    def test_task_crud_is_consistent_across_two_threads(self) -> None:
        errors = []
        created_ids = []
        lock = threading.Lock()

        def worker(offset: int) -> None:
            try:
                for number in range(offset, offset + 4):
                    task = self.registry.create_task(
                        conversation_key="conversation",
                        kind="coding",
                        intent="intent-%s" % number,
                        workdir=self.tempdir.name,
                    )
                    with lock:
                        created_ids.append(task["id"])
                    self.assertEqual(self.registry.get_task(task["id"]), task)
                    updated = self.registry.update_task(
                        task["id"],
                        expected_version=task["version"],
                        result_summary="saved-%s" % number,
                    )
                    self.assertEqual(updated["result_summary"], "saved-%s" % number)
                    self.assertEqual(updated["version"], task["version"] + 1)
            except Exception as error:  # surface thread failures in the main test thread
                errors.append(error)

        threads = [threading.Thread(target=worker, args=(offset,)) for offset in (0, 4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        self.assertEqual(errors, [])
        self.assertEqual(len(created_ids), 8)
        self.assertEqual(len(set(created_ids)), 8)
        tasks = self.registry.list_tasks()
        self.assertEqual(len(tasks), 8)
        self.assertTrue(all(task["version"] == 2 for task in tasks))
        self.assertTrue(self.registry.delete_task(created_ids[0]))
        self.assertIsNone(self.registry.get_task(created_ids[0]))
        self.assertFalse(self.registry.delete_task(created_ids[0]))
        self.assertEqual(len(self.registry.list_tasks()), 7)

    def test_optimistic_version_conflict_is_rejected(self) -> None:
        task = self.registry.create_task(
            conversation_key="conversation", kind="coding", intent="change me"
        )
        updated = self.registry.update_task(
            task["id"], expected_version=task["version"], intent="first writer"
        )

        with self.assertRaises(VersionConflictError) as raised:
            self.registry.update_task(
                task["id"], expected_version=task["version"], intent="stale writer"
            )

        self.assertEqual(raised.exception.task_id, task["id"])
        self.assertEqual(raised.exception.expected_version, task["version"])
        self.assertEqual(raised.exception.actual_version, updated["version"])
        self.assertEqual(self.registry.get_task(task["id"]), updated)

    def test_since_event_cursor_returns_only_newer_events(self) -> None:
        task = self.registry.create_task(
            conversation_key="conversation", kind="coding", intent="one"
        )
        other = self.registry.create_task(
            conversation_key="conversation", kind="coding", intent="two"
        )
        first = self.registry.append_event(task["id"], "task.updated", {"value": 1})
        second = self.registry.append_event(task["id"], "task.updated", {"value": 2})
        third = self.registry.append_event(other["id"], "task.failed", {"code": "x"})

        newer = self.registry.events_since(first["id"])
        self.assertEqual([event["id"] for event in newer], [second["id"], third["id"]])
        self.assertEqual(newer[0]["payload_json"], '{"value": 2}')
        scoped = self.registry.events_since(first["id"], task_id=task["id"])
        self.assertEqual([event["id"] for event in scoped], [second["id"]])
        self.assertEqual(self.registry.events_since(third["id"]), [])

    def test_active_task_dedup_is_atomic_across_two_threads(self) -> None:
        barrier = threading.Barrier(2)
        results = []
        errors = []

        def submit() -> None:
            try:
                barrier.wait()
                results.append(
                    self.registry.create_task(
                        conversation_key="conversation",
                        kind="coding",
                        intent="equivalent request",
                        dedupe_key="request-fingerprint",
                    )
                )
            except Exception as error:
                errors.append(error)

        threads = [threading.Thread(target=submit) for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        self.assertEqual(errors, [])
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["id"], results[1]["id"])
        self.assertEqual(len(self.registry.list_tasks()), 1)

    def test_resource_lock_submission_is_atomic_across_two_threads(self) -> None:
        barrier = threading.Barrier(2)
        results = []
        errors = []

        def submit(intent: str) -> None:
            try:
                barrier.wait()
                results.append(
                    self.registry.create_task(
                        conversation_key="conversation",
                        kind="blender",
                        intent=intent,
                        dedupe_key=intent,
                        resource_key="blender:9876",
                    )
                )
            except Exception as error:
                errors.append(error)

        threads = [
            threading.Thread(target=submit, args=(intent,)) for intent in ("one", "two")
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        self.assertEqual(errors, [])
        self.assertEqual(len(results), 2)
        self.assertNotEqual(results[0]["id"], results[1]["id"])
        states = sorted(task["state"] for task in self.registry.list_tasks())
        self.assertEqual(states, ["queued", "starting"])

    def test_dedup_is_scoped_to_the_conversation(self) -> None:
        first = self.registry.create_task(
            conversation_key="conversation-one",
            kind="coding",
            intent="same request",
            dedupe_key="request-fingerprint",
        )
        second = self.registry.create_task(
            conversation_key="conversation-two",
            kind="coding",
            intent="same request",
            dedupe_key="request-fingerprint",
        )

        self.assertNotEqual(first["id"], second["id"])
        self.assertEqual(len(self.registry.list_tasks()), 2)

    def test_transition_task_enforces_state_machine_and_versions(self) -> None:
        task = self.registry.create_task(
            conversation_key="conversation", kind="coding", intent="one"
        )
        running = self.registry.transition_task(
            task["id"], "running", expected_version=task["version"]
        )
        self.assertEqual(running["state"], "running")
        self.assertEqual(running["version"], task["version"] + 1)

        with self.assertRaises(InvalidTransitionError):
            self.registry.transition_task(
                task["id"], "queued", expected_version=running["version"]
            )

        with self.assertRaises(IncompleteEvidenceError):
            self.registry.transition_task(
                task["id"], "completed", expected_version=running["version"]
            )

        completed = self.registry.transition_task(
            task["id"],
            "completed",
            expected_version=running["version"],
            evidence=CompletionEvidence(executor_confirmed=True),
        )
        self.assertEqual(completed["state"], "completed")
        self.assertIsNotNone(completed["finished_at"])

    def test_max_parallel_is_enforced_by_database_at_insertion(self) -> None:
        first = self.registry.create_task(
            conversation_key="conversation", kind="coding", intent="one"
        )
        second = self.registry.create_task(
            conversation_key="conversation", kind="coding", intent="two"
        )
        third = self.registry.create_task(
            conversation_key="conversation", kind="coding", intent="three"
        )

        self.assertEqual(first["state"], "starting")
        self.assertEqual(second["state"], "starting")
        self.assertEqual(third["state"], "queued")
        with sqlite3.connect(self.db_path) as connection:
            with self.assertRaisesRegex(sqlite3.IntegrityError, "max parallel"):
                connection.execute(
                    """
                    INSERT INTO tasks (
                        id, kind, intent, state, created_at, updated_at
                    ) VALUES ('over-cap', 'coding', 'bypass', 'starting', 1, 1)
                    """
                )


if __name__ == "__main__":
    unittest.main()
