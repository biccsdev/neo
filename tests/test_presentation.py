"""Task 17 — durable, idempotent automatic presentation.

Opening a finished result is a user-visible side effect, so NEO records the
decision durably before it acts. This suite pins that contract:

* the registry migrates v4 databases to v5, keeps every existing row and adds
  ``task_presentations`` with a cascade delete;
* one claim per ``(task_id, version)`` survives two service instances and a
  restart;
* a failed opener releases its claim, so an honest retry can present;
* a new task version may present again;
* incomplete, missing or unverified work never claims and never opens;
* ``TaskPresenter`` reads the task itself, binds it to a conversation and
  reports ``presented`` / ``skipped`` / ``reason`` as JSON-safe fields;
* the production opener is an argv-only, no-shell ``open -- <path>`` call with
  a timeout, and it never accepts a relative path.

Every test uses a temporary database and a fake opener; the real macOS ``open``
command is never invoked.
"""

from __future__ import annotations

import json
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from artifacts import completion_evidence  # noqa: E402
from presentation import (  # noqa: E402
    MacOSOpener,
    PresentationOpenError,
    PresentationService,
    TaskPresenter,
)
from registry import Registry, TaskNotFoundError  # noqa: E402


class PresentationTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tempdir.cleanup)
        self.root = Path(self._tempdir.name)
        self.db_path = self.root / "neo.db"
        self.workdir = self.root / "work"
        self.workdir.mkdir()
        self.registry = Registry(path=self.db_path, max_parallel=8)
        self.opened = []

    # -- fixtures --------------------------------------------------------

    def opener(self, path: str) -> None:
        self.opened.append(path)

    def service(self, opener=None, *, registry=None, **kwargs) -> PresentationService:
        return PresentationService(
            opener or self.opener, ledger=registry or self.registry, **kwargs
        )

    def write(self, name: str = "report.md", text: str = "# done") -> Path:
        path = self.workdir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    def completed(self, *, registry=None, conversation_key: str = "conv-a") -> dict:
        """Drive a real task to ``completed`` with persisted evidence."""
        registry = registry or self.registry
        self.write("report.md")
        task = registry.create_task(
            conversation_key=conversation_key,
            kind="coding",
            intent="write the report",
            workdir=str(self.workdir),
        )
        if task["state"] == "starting":
            task = registry.transition_task(
                task["id"], "running", expected_version=task["version"]
            )
        evidence = completion_evidence(self.workdir, ["report.md"])
        return registry.transition_task(
            task["id"],
            "completed",
            expected_version=task["version"],
            evidence=evidence,
        )

    def ledger_rows(self) -> list:
        with sqlite3.connect(self.db_path) as connection:
            return connection.execute(
                "SELECT task_id, version, claimed_at, presented_at, path "
                "FROM task_presentations ORDER BY task_id, version"
            ).fetchall()


class MigrationAndClaimTests(PresentationTestCase):
    def test_migration_from_v4_keeps_rows_and_adds_the_ledger(self) -> None:
        task = self.completed()
        with sqlite3.connect(self.db_path) as connection:
            connection.execute("DROP TABLE IF EXISTS task_presentations")
            connection.execute("PRAGMA user_version = 4")

        migrated = Registry(path=self.db_path)
        self.assertEqual(migrated.get_task(task["id"])["state"], "completed")
        with sqlite3.connect(self.db_path) as connection:
            version = connection.execute("PRAGMA user_version").fetchone()[0]
            tables = {
                row[0]
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table'"
                )
            }
        self.assertEqual(version, 5)
        self.assertIn("task_presentations", tables)

    def test_one_claim_per_task_version_across_instances_and_restart(self) -> None:
        task = self.completed()
        path = str((self.workdir / "report.md").resolve())
        self.assertTrue(
            self.registry.claim_presentation(task["id"], task["version"], path)
        )
        self.assertFalse(
            self.registry.claim_presentation(task["id"], task["version"], path)
        )
        reopened = Registry(path=self.db_path)
        self.assertFalse(
            reopened.claim_presentation(task["id"], task["version"], path)
        )
        self.assertEqual(len(self.ledger_rows()), 1)

    def test_release_frees_only_an_unfinished_claim(self) -> None:
        task = self.completed()
        path = str((self.workdir / "report.md").resolve())
        self.assertTrue(
            self.registry.claim_presentation(task["id"], task["version"], path)
        )
        self.assertTrue(self.registry.release_presentation(task["id"], task["version"]))
        self.assertTrue(
            self.registry.claim_presentation(task["id"], task["version"], path)
        )
        self.assertTrue(self.registry.finish_presentation(task["id"], task["version"]))
        self.assertFalse(self.registry.finish_presentation(task["id"], task["version"]))
        self.assertFalse(self.registry.release_presentation(task["id"], task["version"]))
        self.assertFalse(
            self.registry.claim_presentation(task["id"], task["version"], path)
        )
        row = self.ledger_rows()[0]
        self.assertIsNotNone(row[3])

    def test_claim_on_a_missing_task_is_a_domain_error(self) -> None:
        with self.assertRaises(TaskNotFoundError):
            self.registry.claim_presentation("missing", 1, "/tmp/x.md")
        self.assertEqual(self.ledger_rows(), [])

    def test_deleting_a_task_cascades_its_presentation_row(self) -> None:
        task = self.completed()
        path = str((self.workdir / "report.md").resolve())
        self.assertTrue(
            self.registry.claim_presentation(task["id"], task["version"], path)
        )
        self.assertTrue(self.registry.finish_presentation(task["id"], task["version"]))
        self.assertTrue(self.registry.delete_task(task["id"]))
        self.assertEqual(self.ledger_rows(), [])


class DurableServiceTests(PresentationTestCase):
    def test_durable_claim_presents_once_across_services_and_restart(self) -> None:
        task = self.completed()
        first = self.service().present(task)
        second = self.service().present(task)
        restarted = Registry(path=self.db_path)
        third = self.service(registry=restarted).present(task)
        self.assertTrue(first.presented)
        self.assertFalse(second.presented)
        self.assertIn("already presented", second.reason)
        self.assertFalse(third.presented)
        self.assertEqual(self.opened, [str((self.workdir / "report.md").resolve())])
        self.assertEqual(len(self.ledger_rows()), 1)
        self.assertIsNotNone(self.ledger_rows()[0][3])

    def test_failed_opener_releases_the_durable_claim(self) -> None:
        task = self.completed()
        attempts = []
        failures = [OSError("no handler")]

        def flaky(path: str) -> None:
            attempts.append(path)
            if failures:
                raise failures.pop()

        first = self.service(flaky).present(task)
        self.assertFalse(first.presented)
        self.assertIn("opener failed", first.reason)
        self.assertEqual(self.ledger_rows(), [])
        second = self.service(flaky).present(task)
        self.assertTrue(second.presented)
        self.assertEqual(len(attempts), 2)
        self.assertEqual(len(self.ledger_rows()), 1)

    def test_a_new_version_may_present_again(self) -> None:
        task = self.completed()
        service = self.service()
        self.assertTrue(service.present(task).presented)
        bumped = self.registry.update_task(
            task["id"], expected_version=task["version"], result_summary="amended"
        )
        self.assertNotEqual(bumped["version"], task["version"])
        again = service.present(bumped)
        self.assertTrue(again.presented)
        self.assertEqual(len(self.opened), 2)
        self.assertEqual(len(self.ledger_rows()), 2)

    def test_incomplete_missing_and_unverified_never_claim_or_open(self) -> None:
        queued = self.registry.create_task(
            conversation_key="conv-a", kind="coding", intent="still running"
        )
        result = self.service().present(queued)
        self.assertFalse(result.presented)
        self.assertEqual(self.ledger_rows(), [])

        self.write("report.md")
        completed = self.completed()
        missing = dict(completed)
        (self.workdir / "report.md").unlink()
        self.assertFalse(self.service().present(missing).presented)
        self.assertEqual(self.ledger_rows(), [])

        unverified = dict(completed)
        unverified["artifacts_json"] = json.dumps([])
        self.assertFalse(self.service().present(unverified).presented)
        self.write("report.md")
        self.assertEqual(self.ledger_rows(), [])
        self.assertEqual(self.opened, [])


class PresenterTests(PresentationTestCase):
    def presenter(self, **kwargs) -> TaskPresenter:
        return TaskPresenter(self.registry, self.service(**kwargs))

    def test_presenter_presents_a_completed_task_as_json(self) -> None:
        task = self.completed()
        outcome = self.presenter().present_task(task["id"])
        self.assertTrue(outcome["presented"])
        self.assertFalse(outcome["skipped"])
        self.assertEqual(outcome["reason"], "presented")
        self.assertEqual(outcome["task_id"], task["id"])
        self.assertIn("path", outcome)
        json.dumps(outcome)

    def test_presenter_enforces_conversation_binding(self) -> None:
        task = self.completed(conversation_key="conv-a")
        with self.assertRaises(TaskNotFoundError):
            self.presenter().present_task(task["id"], "conv-b")
        self.assertEqual(self.opened, [])
        outcome = self.presenter().present_task(task["id"], "conv-a")
        self.assertTrue(outcome["presented"])

    def test_presenter_refuses_a_missing_task(self) -> None:
        with self.assertRaises(TaskNotFoundError):
            self.presenter().present_task("does-not-exist")
        self.assertEqual(self.opened, [])

    def test_presenter_reports_an_incomplete_task_as_skipped(self) -> None:
        queued = self.registry.create_task(
            conversation_key="conv-a", kind="coding", intent="waiting"
        )
        outcome = self.presenter().present_task(queued["id"])
        self.assertFalse(outcome["presented"])
        self.assertTrue(outcome["skipped"])
        self.assertIn("not completed", outcome["reason"])
        self.assertEqual(self.opened, [])


class OpenerTests(PresentationTestCase):
    def test_macos_opener_uses_an_argv_only_no_shell_command(self) -> None:
        calls = []

        def runner(argv, **kwargs):
            calls.append((list(argv), kwargs))
            return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")

        target = str(self.write("report.md").resolve())
        opener = MacOSOpener(timeout=5, run=runner)
        opener(target)
        self.assertEqual(len(calls), 1)
        argv, kwargs = calls[0]
        self.assertEqual(argv, ["open", "--", target])
        self.assertFalse(kwargs.get("shell", False))
        self.assertEqual(kwargs.get("timeout"), 5)
        self.assertIsInstance(argv[0], str)
        for token in argv:
            self.assertNotIn(" ", token)

    def test_macos_opener_reports_a_non_zero_exit_as_a_failure(self) -> None:
        def runner(argv, **kwargs):
            return SimpleNamespace(returncode=1, stdout=b"", stderr=b"no handler")

        opener = MacOSOpener(timeout=5, run=runner)
        with self.assertRaises(PresentationOpenError):
            opener(str(self.write("report.md").resolve()))

    def test_macos_opener_requires_an_absolute_path(self) -> None:
        called = []

        def runner(argv, **kwargs):
            called.append(list(argv))
            return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")

        opener = MacOSOpener(run=runner)
        with self.assertRaises(ValueError):
            opener("relative/report.md")
        self.assertEqual(called, [])


if __name__ == "__main__":
    unittest.main()
