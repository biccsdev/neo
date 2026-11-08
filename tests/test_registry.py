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

from registry import (  # noqa: E402
    ActiveTaskError,
    CapacityExceededError,
    InvalidCursorError,
    Registry,
    ResourceBusyError,
    TaskNotFoundError,
    VersionConflictError,
)
from state_machine import (  # noqa: E402
    CompletionEvidence,
    IncompleteEvidenceError,
    InvalidTransitionError,
    UnknownStateError,
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
        task = self.registry.create_task(
            conversation_key="conversation", kind="coding", intent="survives"
        )
        reopened = Registry(path=self.db_path)
        reopened.migrate()

        with sqlite3.connect(self.db_path) as connection:
            tables = {
                row[0]
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table'"
                )
            }
            self.assertTrue(
                {
                    "tasks",
                    "task_events",
                    "workers",
                    "capability_consents",
                    "task_presentations",
                }.issubset(tables)
            )
            self.assertEqual(connection.execute("PRAGMA user_version").fetchone()[0], 5)
            self.assertEqual(connection.execute("PRAGMA journal_mode").fetchone()[0], "wal")
            indexes = {
                row[0]
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'index'"
                )
            }
            triggers = {
                row[0]
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'trigger'"
                )
            }
            self.assertIn("tasks_active_dedupe_v2", indexes)
            self.assertIn("task_events_task_cursor", indexes)
            self.assertIn("tasks_max_parallel_insert", triggers)
            self.assertIn("tasks_max_parallel_update", triggers)
        self.assertEqual(reopened.get_task(task["id"]), task)

    def test_reopening_without_an_override_preserves_the_parallel_limit(self) -> None:
        Registry(path=self.db_path, max_parallel=1)
        reopened = Registry(path=self.db_path)
        self.assertEqual(reopened.max_parallel, 1)

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
        doomed = self.registry.get_task(created_ids[0])
        doomed = self.registry.transition_task(
            doomed["id"], "running", expected_version=doomed["version"]
        )
        doomed = self.registry.transition_task(
            doomed["id"], "cancelling", expected_version=doomed["version"]
        )
        self.registry.transition_task(
            doomed["id"], "cancelled", expected_version=doomed["version"]
        )
        self.assertTrue(self.registry.delete_task(created_ids[0]))
        self.assertIsNone(self.registry.get_task(created_ids[0]))
        self.assertFalse(self.registry.delete_task(created_ids[0]))
        self.assertEqual(len(self.registry.list_tasks()), 7)

    def test_transition_requires_an_integer_expected_version(self) -> None:
        task = self.registry.create_task(
            conversation_key="conv", kind="coding", intent="typed version"
        )

        for version in (True, 1.0, "1", None):
            with self.subTest(version=version):
                with self.assertRaises(ValueError):
                    self.registry.transition_task(
                        task["id"], "running", expected_version=version
                    )
                self.assertEqual(self.registry.get_task(task["id"]), task)

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

    def test_max_parallel_is_enforced_when_a_queued_task_starts(self) -> None:
        first = self.registry.create_task(
            conversation_key="conversation", kind="coding", intent="one"
        )
        self.registry.create_task(
            conversation_key="conversation", kind="coding", intent="two"
        )
        queued = self.registry.create_task(
            conversation_key="conversation", kind="coding", intent="three"
        )

        with self.assertRaises(CapacityExceededError):
            self.registry.transition_task(
                queued["id"], "starting", expected_version=queued["version"]
            )

        self.assertEqual(self.registry.get_task(queued["id"])["state"], "queued")
        self.assertEqual(self.registry.get_task(first["id"])["state"], "starting")

    def test_resource_conflict_is_a_domain_error_on_transition(self) -> None:
        holder = self.registry.create_task(
            conversation_key="conversation",
            kind="blender",
            intent="holder",
            resource_key="blender:9876",
        )
        waiter = self.registry.create_task(
            conversation_key="conversation",
            kind="blender",
            intent="waiter",
            resource_key="blender:9876",
        )

        with self.assertRaises(ResourceBusyError):
            self.registry.transition_task(
                waiter["id"], "starting", expected_version=waiter["version"]
            )

        self.assertEqual(self.registry.get_task(holder["id"])["state"], "starting")
        self.assertEqual(self.registry.get_task(waiter["id"])["state"], "queued")

    def test_update_task_translates_resource_conflict(self) -> None:
        self.registry.create_task(
            conversation_key="conversation",
            kind="blender",
            intent="holder",
            resource_key="blender:9876",
        )
        other = self.registry.create_task(
            conversation_key="conversation", kind="browser", intent="other"
        )

        with self.assertRaises(ResourceBusyError):
            self.registry.update_task(
                other["id"],
                expected_version=other["version"],
                resource_key="blender:9876",
            )
        self.assertIsNone(self.registry.get_task(other["id"])["resource_key"])

    def test_active_task_cannot_be_deleted(self) -> None:
        task = self.registry.create_task(
            conversation_key="conversation", kind="coding", intent="one"
        )

        with self.assertRaises(ActiveTaskError):
            self.registry.delete_task(task["id"])

        self.assertEqual(self.registry.get_task(task["id"]), task)

    def test_queued_cancellation_does_not_claim_a_start_time(self) -> None:
        self.registry.create_task(
            conversation_key="conversation", kind="coding", intent="one"
        )
        self.registry.create_task(
            conversation_key="conversation", kind="coding", intent="two"
        )
        queued = self.registry.create_task(
            conversation_key="conversation", kind="coding", intent="three"
        )

        cancelled = self.registry.transition_task(
            queued["id"], "cancelled", expected_version=queued["version"]
        )

        self.assertIsNone(cancelled["started_at"])
        self.assertIsNotNone(cancelled["finished_at"])

    def test_completion_evidence_is_audited_on_the_task(self) -> None:
        task = self.registry.create_task(
            conversation_key="conversation", kind="coding", intent="one"
        )
        task = self.registry.update_task(
            task["id"],
            expected_version=task["version"],
            expected_json='["result.txt"]',
        )
        task = self.registry.transition_task(
            task["id"], "running", expected_version=task["version"]
        )

        with self.assertRaises(IncompleteEvidenceError):
            self.registry.transition_task(
                task["id"],
                "completed",
                expected_version=task["version"],
                evidence=CompletionEvidence(
                    executor_confirmed=True,
                    verified_artifacts=("different.txt",),
                ),
            )

        completed = self.registry.transition_task(
            task["id"],
            "completed",
            expected_version=task["version"],
            evidence=CompletionEvidence(
                executor_confirmed=True,
                verified_artifacts=("result.txt",),
            ),
        )
        self.assertEqual(completed["expected_json"], '["result.txt"]')
        self.assertEqual(completed["artifacts_json"], '["result.txt"]')

    def test_dedupe_key_is_released_after_terminal_state(self) -> None:
        first = self.registry.create_task(
            conversation_key="conversation",
            kind="coding",
            intent="same request",
            dedupe_key="request-fingerprint",
        )
        running = self.registry.transition_task(
            first["id"], "running", expected_version=first["version"]
        )
        self.registry.transition_task(
            running["id"], "failed", expected_version=running["version"]
        )

        second = self.registry.create_task(
            conversation_key="conversation",
            kind="coding",
            intent="same request",
            dedupe_key="request-fingerprint",
        )
        self.assertNotEqual(first["id"], second["id"])

    def test_invalid_event_cursor_is_rejected(self) -> None:
        with self.assertRaises(InvalidCursorError):
            self.registry.events_since("not-a-number")
        with self.assertRaises(InvalidCursorError):
            self.registry.events_since(-1)
        with self.assertRaises(InvalidCursorError):
            self.registry.events_since("²")
        with self.assertRaises(InvalidCursorError):
            self.registry.events_since("١٢")

    def test_unknown_transition_target_keeps_state_machine_error_type(self) -> None:
        task = self.registry.create_task(
            conversation_key="conversation", kind="coding", intent="one"
        )
        with self.assertRaises(UnknownStateError):
            self.registry.transition_task(
                task["id"], "not-a-state", expected_version=task["version"]
            )

    def test_event_for_unknown_task_is_a_domain_error(self) -> None:
        with self.assertRaises(TaskNotFoundError):
            self.registry.append_event("missing", "task.updated", {})

    def test_missing_parallel_setting_falls_back_to_four(self) -> None:
        with sqlite3.connect(self.db_path) as connection:
            connection.execute("DELETE FROM registry_settings")

        states = [
            self.registry.create_task(
                conversation_key="conversation", kind="coding", intent="task-%s" % index
            )["state"]
            for index in range(5)
        ]
        self.assertEqual(states, ["starting", "starting", "starting", "starting", "queued"])

    def test_migration_replaces_an_older_trigger_body(self) -> None:
        with sqlite3.connect(self.db_path) as connection:
            connection.execute("DROP TRIGGER tasks_max_parallel_insert")
            connection.execute(
                """
                CREATE TRIGGER tasks_max_parallel_insert
                BEFORE INSERT ON tasks
                WHEN NEW.state IN ('starting', 'running')
                BEGIN
                    SELECT RAISE(ABORT, 'max parallel tasks exceeded')
                    WHERE (SELECT COUNT(*) FROM tasks WHERE state IN ('starting', 'running'))
                          >= (SELECT value FROM registry_settings WHERE key = 'max_parallel');
                END
                """
            )
            connection.execute("PRAGMA user_version = 1")

        migrated = Registry(path=self.db_path)
        with sqlite3.connect(self.db_path) as connection:
            trigger_sql = connection.execute(
                "SELECT sql FROM sqlite_master "
                "WHERE type = 'trigger' AND name = 'tasks_max_parallel_insert'"
            ).fetchone()[0]
            self.assertIn("COALESCE", trigger_sql)
            connection.execute("DELETE FROM registry_settings")
            for index in range(4):
                connection.execute(
                    "INSERT INTO tasks (id, kind, intent, state, created_at, updated_at) "
                    "VALUES (?, 'coding', 'i', 'starting', 1, 1)",
                    ("probe-%s" % index,),
                )
            with self.assertRaisesRegex(sqlite3.IntegrityError, "max parallel"):
                connection.execute(
                    "INSERT INTO tasks (id, kind, intent, state, created_at, updated_at) "
                    "VALUES ('probe-4', 'coding', 'i', 'starting', 1, 1)"
                )
        self.assertEqual(migrated.max_parallel, 2)

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


class UnfinishedTaskQueryTests(unittest.TestCase):
    """The restart pass needs every row that has not reached a terminal state."""

    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        self.db_path = Path(self.tempdir.name) / "registry.db"

    def test_unfinished_tasks_exclude_terminal_rows_and_keep_creation_order(self) -> None:
        registry = Registry(path=self.db_path, max_parallel=4)
        first = registry.create_task(
            conversation_key="conversation", kind="general", intent="first"
        )
        second = registry.create_task(
            conversation_key="conversation", kind="general", intent="second"
        )
        done = registry.create_task(
            conversation_key="conversation", kind="general", intent="done"
        )
        running = registry.transition_task(
            done["id"], "running", expected_version=done["version"]
        )
        registry.transition_task(
            running["id"],
            "completed",
            expected_version=running["version"],
            evidence=CompletionEvidence(executor_confirmed=True),
        )

        rows = registry.unfinished_tasks()

        self.assertEqual([row["id"] for row in rows], [first["id"], second["id"]])
        self.assertEqual([row["state"] for row in rows], ["starting", "starting"])

    def test_unfinished_tasks_include_queued_rows(self) -> None:
        registry = Registry(path=self.db_path, max_parallel=1)
        holder = registry.create_task(
            conversation_key="conversation", kind="general", intent="holder"
        )
        queued = registry.create_task(
            conversation_key="conversation", kind="general", intent="queued"
        )
        self.assertEqual(queued["state"], "queued")

        rows = registry.unfinished_tasks()

        self.assertEqual([row["id"] for row in rows], [holder["id"], queued["id"]])


if __name__ == "__main__":
    unittest.main()
