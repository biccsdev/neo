"""Task 5 — the worker-token boundary in front of the task service.

A worker gets a random token once. Only its SHA-256 digest is stored, and a
transition needs the token to match before the service will move the task.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import tools  # noqa: E402
from events import TaskService  # noqa: E402
from registry import Registry, TaskNotFoundError  # noqa: E402
from state_machine import (  # noqa: E402
    CompletionEvidence,
    IncompleteEvidenceError,
    InvalidTransitionError,
)
from tools import (  # noqa: E402
    InvalidWorkerTokenError,
    WorkerTaskTools,
    WorkerTokenStore,
    WorkerToolError,
)


class WorkerTokenTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tempdir.name) / "registry.db"
        self.registry = Registry(path=self.db_path, max_parallel=4)
        self.service = TaskService(self.registry)
        self.tokens = WorkerTokenStore(self.registry)
        self.tools = WorkerTaskTools(self.service, self.tokens)
        self.task = self.service.create_task(
            conversation_key="conversation", kind="coding", intent="do the work"
        )

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    # -- helpers ---------------------------------------------------------

    def token_rows(self) -> list:
        with sqlite3.connect(self.db_path) as connection:
            return connection.execute(
                "SELECT task_id, token_hash, issued_at FROM task_worker_tokens"
            ).fetchall()

    def task_events(self, task_id: str) -> list:
        return [
            event
            for event in self.service.events_since("conversation")
            if event["task_id"] == task_id
        ]

    # -- storage ---------------------------------------------------------

    def test_worker_token_table_is_added_idempotently(self) -> None:
        self.registry.migrate()
        Registry(path=self.db_path).migrate()

        with sqlite3.connect(self.db_path) as connection:
            columns = {
                row[1]: row[2]
                for row in connection.execute(
                    "PRAGMA table_info(task_worker_tokens)"
                ).fetchall()
            }
            primary_key = [
                row[1]
                for row in connection.execute(
                    "PRAGMA table_info(task_worker_tokens)"
                ).fetchall()
                if row[5]
            ]

        self.assertEqual(set(columns), {"task_id", "token_hash", "issued_at"})
        self.assertEqual(primary_key, ["task_id"])
        self.assertTrue(columns["token_hash"])

    def test_an_existing_database_gains_the_token_table_on_migrate(self) -> None:
        with sqlite3.connect(self.db_path) as connection:
            connection.execute("DROP TABLE task_worker_tokens")
            connection.execute("PRAGMA user_version = 2")

        Registry(path=self.db_path)

        with sqlite3.connect(self.db_path) as connection:
            version = connection.execute("PRAGMA user_version").fetchone()[0]
        self.assertEqual(version, 5)
        token = self.tokens.issue(self.task["id"])
        self.assertEqual(len(self.token_rows()), 1)
        self.assertTrue(self.tokens.verify(self.task["id"], token))

    def test_issue_returns_a_raw_token_and_stores_only_its_digest(self) -> None:
        token = self.tokens.issue(self.task["id"])

        self.assertIsInstance(token, str)
        self.assertGreaterEqual(len(token), 32)

        rows = self.token_rows()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0][0], self.task["id"])
        self.assertEqual(rows[0][1], hashlib.sha256(token.encode("utf-8")).hexdigest())
        self.assertNotEqual(rows[0][1], token)
        self.assertIsNotNone(rows[0][2])

        for path in (self.db_path, self.db_path.with_name(self.db_path.name + "-wal")):
            if path.exists():
                self.assertNotIn(token.encode("utf-8"), path.read_bytes())

    def test_each_task_gets_its_own_token(self) -> None:
        other = self.service.create_task(
            conversation_key="conversation", kind="coding", intent="second"
        )

        first_token = self.tokens.issue(self.task["id"])
        second_token = self.tokens.issue(other["id"])

        self.assertNotEqual(first_token, second_token)
        self.assertTrue(self.tokens.verify(self.task["id"], first_token))
        self.assertTrue(self.tokens.verify(other["id"], second_token))
        self.assertFalse(self.tokens.verify(other["id"], first_token))

    def test_tokens_survive_a_registry_reopen(self) -> None:
        token = self.tokens.issue(self.task["id"])
        reopened = WorkerTokenStore(Registry(path=self.db_path))

        self.assertTrue(reopened.verify(self.task["id"], token))

    def test_deleting_a_finished_task_removes_its_token(self) -> None:
        token = self.tokens.issue(self.task["id"])
        running = self.service.transition(
            self.task["id"], "running", expected_version=self.task["version"]
        )
        self.service.transition(
            running["id"], "failed", expected_version=running["version"]
        )

        self.assertTrue(self.registry.delete_task(self.task["id"]))
        self.assertEqual(self.token_rows(), [])
        self.assertFalse(self.tokens.verify(self.task["id"], token))

    def test_issuing_for_an_unknown_task_is_a_domain_error(self) -> None:
        with self.assertRaises(TaskNotFoundError):
            self.tokens.issue("missing")

    def test_reissue_rotates_the_token(self) -> None:
        first = self.tokens.issue(self.task["id"])
        second = self.tokens.issue(self.task["id"])

        self.assertNotEqual(first, second)
        self.assertFalse(self.tokens.verify(self.task["id"], first))
        self.assertTrue(self.tokens.verify(self.task["id"], second))
        self.assertEqual(len(self.token_rows()), 1)

    def test_rotation_during_an_update_revokes_that_transition(self) -> None:
        token = self.tokens.issue(self.task["id"])
        original = self.registry.transition_task

        def rotate_then_transition(*args, **kwargs):
            self.tokens.issue(self.task["id"])
            return original(*args, **kwargs)

        self.registry.transition_task = rotate_then_transition
        with self.assertRaises(InvalidWorkerTokenError):
            self.tools.worker_task_update(
                self.task["id"], token, "running", self.task["version"]
            )

        self.assertEqual(self.registry.get_task(self.task["id"]), self.task)
        self.assertEqual(len(self.task_events(self.task["id"])), 1)

    # -- verification ----------------------------------------------------

    def test_verify_accepts_the_issued_token(self) -> None:
        token = self.tokens.issue(self.task["id"])

        self.assertTrue(self.tokens.verify(self.task["id"], token))

    def test_verify_rejects_wrong_missing_and_unknown_tokens(self) -> None:
        token = self.tokens.issue(self.task["id"])

        for candidate in ("", None, "wrong", token + "x", token[:-1]):
            with self.subTest(candidate=candidate):
                self.assertFalse(self.tokens.verify(self.task["id"], candidate))

        self.assertFalse(self.tokens.verify("missing", token))
        self.assertFalse(self.tokens.verify(self.task["id"], 12345))

    def test_verify_compares_in_constant_time(self) -> None:
        token = self.tokens.issue(self.task["id"])
        recorded = []
        original = tools.hmac.compare_digest

        def recorder(left, right):
            recorded.append((left, right))
            return original(left, right)

        with mock.patch.object(tools.hmac, "compare_digest", recorder):
            self.assertTrue(self.tokens.verify(self.task["id"], token))
            self.assertFalse(self.tokens.verify(self.task["id"], "wrong"))

        self.assertEqual(len(recorded), 2)

    # -- worker_task_update ----------------------------------------------

    def test_stale_version_does_not_bypass_token_authentication(self) -> None:
        self.tokens.issue(self.task["id"])
        running = self.service.transition(
            self.task["id"], "running", expected_version=self.task["version"]
        )

        with self.assertRaises(InvalidWorkerTokenError):
            self.tools.neo_task_update(
                self.task["id"],
                "wrong-token",
                state="waiting",
                expected_version=self.task["version"],
            )

        self.assertEqual(self.registry.get_task(self.task["id"]), running)
        self.assertEqual(len(self.task_events(self.task["id"])), 2)

    def test_invalid_worker_payload_is_a_domain_error_without_changes(self) -> None:
        token = self.tokens.issue(self.task["id"])
        invalid_payloads = (
            {"progress": object()},
            {"progress": float("nan")},
            {"progress": True},
            {"detail": object()},
            {"detail": "x" * 4001},
            {"summary": object()},
        )

        for payload in invalid_payloads:
            with self.subTest(payload=payload):
                with self.assertRaises(WorkerToolError):
                    self.tools.neo_task_update(
                        self.task["id"],
                        token,
                        state="running",
                        expected_version=self.task["version"],
                        **payload,
                    )
                self.assertEqual(self.registry.get_task(self.task["id"]), self.task)
                self.assertEqual(len(self.task_events(self.task["id"])), 1)

    def test_public_update_tool_reports_progress_and_detail(self) -> None:
        token = self.tokens.issue(self.task["id"])

        running = self.tools.neo_task_update(
            self.task["id"],
            token,
            state="running",
            expected_version=self.task["version"],
            progress=25,
            detail="One quarter complete",
        )

        self.assertEqual(running["state"], "running")
        payload = self.task_events(self.task["id"])[-1]["payload"]
        self.assertEqual(payload["progress"], 25)
        self.assertEqual(payload["detail"], "One quarter complete")

    def test_worker_update_transitions_and_emits_one_event(self) -> None:
        token = self.tokens.issue(self.task["id"])

        running = self.tools.worker_task_update(
            self.task["id"],
            token,
            "running",
            self.task["version"],
            summary="Worker is running",
        )

        self.assertEqual(running["state"], "running")
        self.assertEqual(running["version"], self.task["version"] + 1)
        events = self.task_events(self.task["id"])
        self.assertEqual([event["kind"] for event in events], ["task.created", "task.updated"])
        self.assertEqual(events[-1]["payload"]["summary"], "Worker is running")

    def test_wrong_token_changes_nothing(self) -> None:
        self.tokens.issue(self.task["id"])

        with self.assertRaises(InvalidWorkerTokenError):
            self.tools.worker_task_update(
                self.task["id"], "wrong-token", "running", self.task["version"]
            )

        self.assertEqual(self.registry.get_task(self.task["id"]), self.task)
        self.assertEqual(len(self.task_events(self.task["id"])), 1)

    def test_missing_token_changes_nothing(self) -> None:
        self.tokens.issue(self.task["id"])

        for candidate in (None, ""):
            with self.subTest(candidate=candidate):
                with self.assertRaises(InvalidWorkerTokenError):
                    self.tools.worker_task_update(
                        self.task["id"], candidate, "running", self.task["version"]
                    )

        self.assertEqual(self.registry.get_task(self.task["id"]), self.task)
        self.assertEqual(len(self.task_events(self.task["id"])), 1)

    def test_unknown_task_is_a_token_error_not_an_enumeration(self) -> None:
        token = self.tokens.issue(self.task["id"])

        with self.assertRaises(InvalidWorkerTokenError):
            self.tools.worker_task_update("missing", token, "running", 1)

    def test_worker_update_still_respects_the_state_machine(self) -> None:
        token = self.tokens.issue(self.task["id"])

        with self.assertRaises(InvalidTransitionError):
            self.tools.worker_task_update(
                self.task["id"], token, "waiting", self.task["version"]
            )

        self.assertEqual(self.registry.get_task(self.task["id"]), self.task)

    def test_worker_update_to_completed_needs_evidence(self) -> None:
        token = self.tokens.issue(self.task["id"])
        running = self.tools.worker_task_update(
            self.task["id"], token, "running", self.task["version"]
        )

        with self.assertRaises(IncompleteEvidenceError):
            self.tools.worker_task_update(
                running["id"], token, "completed", running["version"]
            )

        self.assertEqual(self.registry.get_task(self.task["id"]), running)

    def test_worker_update_to_completed_with_evidence_emits_task_completed(self) -> None:
        token = self.tokens.issue(self.task["id"])
        running = self.tools.worker_task_update(
            self.task["id"], token, "running", self.task["version"]
        )

        completed = self.tools.worker_task_update(
            running["id"],
            token,
            "completed",
            running["version"],
            summary="Done",
            evidence=CompletionEvidence(executor_confirmed=True),
        )

        self.assertEqual(completed["state"], "completed")
        events = self.task_events(self.task["id"])
        self.assertEqual(events[-1]["kind"], "task.completed")
        self.assertEqual(json.loads(events[-1]["payload_json"])["state"], "completed")


if __name__ == "__main__":
    unittest.main()
