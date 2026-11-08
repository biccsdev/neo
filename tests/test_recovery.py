"""Task 22 — restart recovery and failure honesty.

A worker can outlive the viewer. After a restart the new process inherits rows
it did not create: some carry a live PID, some are dead, some never had a
process at all, and one may be corrupt. ``ProcessSupervisor.reconcile()`` must
answer one question per row — is the work still running? — and record an
honest answer without ever guessing completion.

This suite pins:

* queued work is not a process and stays queued;
* a dead or missing worker becomes ``failed`` (or ``unknown`` when a
  cancellation cannot be confirmed), never ``completed``;
* a live PID is adopted by a polling monitor — no signal, no kill, and no
  ``Popen.wait`` on a child this process does not own;
* a receipt proves a process ended, never that its work succeeded;
* recovering the last worker holding an exclusive resource releases that lock
  and records the reclaim on the public event;
* a corrupt row is reported and never aborts the rest of the pass or rolls
  back unrelated tasks;
* terminal rows, session ids and conversation keys survive untouched;
* a durable presentation claim still opens a result exactly once across a
  restart;
* ``NeoRuntime.build`` reconciles once and launches nothing.

Every test uses a temporary database. Processes are short-lived real children
or an injected ``alive_fn``; nothing is installed and no network is used.
"""

from __future__ import annotations

import json
import os
import signal
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock
from uuid import uuid4

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import workers  # noqa: E402
from artifacts import completion_evidence  # noqa: E402
from events import TaskService  # noqa: E402
from neo_tools import WorkerTokenStore  # noqa: E402
from presentation import PresentationService  # noqa: E402
from registry import Registry  # noqa: E402
from workers import ProcessSupervisor, process_alive  # noqa: E402


def dead_process(pid: int) -> bool:
    """An injected liveness probe that reports every process as gone."""
    return False


DEAD = dead_process


class RecoveryTestCase(unittest.TestCase):
    """Shared fixtures: a temporary database, a supervisor factory, receipts."""

    max_parallel = 4

    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        self.root = Path(self.tempdir.name)
        self.db_path = self.root / "neo.db"
        self.log_dir = self.root / "logs"
        self.registry = Registry(path=self.db_path, max_parallel=self.max_parallel)
        self.service = TaskService(self.registry)
        self.tokens = WorkerTokenStore(self.registry)

    # -- process fixtures ------------------------------------------------

    def spawn_sleep(self, seconds: float = 30.0) -> subprocess.Popen:
        process = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(%r)" % seconds],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        self.addCleanup(self._reap, process)
        return process

    @staticmethod
    def _reap(process: subprocess.Popen) -> None:
        if process.poll() is None:
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass

    def dead_pid(self) -> int:
        process = subprocess.Popen(
            [sys.executable, "-c", "pass"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        process.wait(timeout=10)
        return process.pid

    # -- task fixtures ---------------------------------------------------

    def supervisor(self, *, alive=None, service=None, tokens=None, log_dir=None):
        supervisor = ProcessSupervisor(
            service or self.service,
            tokens or self.tokens,
            log_dir=log_dir or self.log_dir,
            alive_fn=alive,
        )
        self.addCleanup(supervisor.close)
        return supervisor

    def restart(self, *, alive=None):
        """Build a fresh graph over the same database, as a restart would."""
        registry = Registry(path=self.db_path, max_parallel=self.max_parallel)
        service = TaskService(registry)
        tokens = WorkerTokenStore(registry)
        supervisor = self.supervisor(
            alive=alive, service=service, tokens=tokens
        )
        return supervisor, registry, service

    def create(self, *, resource_key=None, conversation="conv-a", intent="recover"):
        return self.service.create_task(
            conversation_key=conversation,
            kind="general",
            intent=intent,
            resource_key=resource_key,
        )

    def running(self, task=None, **kwargs):
        task = task or self.create(**kwargs)
        if task["state"] == "starting":
            task = self.service.transition(
                task["id"], "running", expected_version=task["version"]
            )
        return task

    def record_process(self, task, pid, *, log_path=None, pgid=None):
        """Record the durable identity a previous run would have left behind."""
        path = Path(log_path) if log_path is not None else self.log_dir / (
            "%s.log" % task["id"]
        )
        updated = self.registry.update_task(
            task["id"],
            expected_version=task["version"],
            pid=pid,
            pgid=pid if pgid is None else pgid,
            log_path=str(path),
        )
        with sqlite3.connect(self.db_path) as connection:
            connection.execute(
                "INSERT INTO workers (id, task_id, pid, pgid, state, started_at, "
                "heartbeat_at, log_path) VALUES (?, ?, ?, ?, 'running', ?, ?, ?)",
                (
                    uuid4().hex,
                    task["id"],
                    pid,
                    pid if pgid is None else pgid,
                    time.time(),
                    time.time(),
                    str(path),
                ),
            )
        return updated

    def write_receipt(self, log_path, *, task_id, pid, exit_code) -> Path:
        log_path = Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        receipt = log_path.with_suffix(".receipt.json")
        receipt.write_text(
            json.dumps(
                {
                    "task_id": task_id,
                    "pid": pid,
                    "pgid": pid,
                    "exit_code": exit_code,
                    "finished_at": time.time(),
                    "term_sent": False,
                    "kill_sent": False,
                }
            )
        )
        return receipt

    def event_payloads(self, task_id, service=None, conversation="conv-a"):
        service = service or self.service
        return [event["payload"] for event in service.task_events(conversation, task_id)]

    def wait_for(self, predicate, timeout: float = 10.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            value = predicate()
            if value:
                return value
            time.sleep(0.02)
        self.fail("condition was not met within %.1fs" % timeout)

    def worker_rows(self) -> list:
        with sqlite3.connect(self.db_path) as connection:
            return connection.execute(
                "SELECT task_id, pid, state FROM workers ORDER BY rowid"
            ).fetchall()


class QueuedRecoveryTests(RecoveryTestCase):
    max_parallel = 1

    def test_queued_task_stays_queued_because_no_process_is_expected(self) -> None:
        self.create(intent="holder")
        queued = self.create(intent="waiting")
        self.assertEqual(queued["state"], "queued")

        supervisor, registry, service = self.restart(alive=DEAD)
        summary = supervisor.reconcile()

        self.assertEqual(registry.get_task(queued["id"])["state"], "queued")
        self.assertIn(queued["id"], summary["queued"])
        self.assertNotIn(queued["id"], summary["failed"])
        # The holder never had a process, so it is honestly failed instead.
        self.assertEqual(len(summary["failed"]), 1)
        self.assertEqual([row for row in self.worker_rows() if row[0] == queued["id"]], [])

    def test_recovery_of_the_capacity_holder_drains_a_queued_task(self) -> None:
        holder = self.running(intent="holder")
        holder = self.record_process(holder, 999_999_999)
        queued = self.create(intent="queued")
        self.assertEqual(queued["state"], "queued")

        marker = self.root / "queued-ran.txt"
        supervisor, registry, service = self.restart(alive=DEAD)
        supervisor.submit(
            queued["id"],
            [sys.executable, "-c", "import sys; open(sys.argv[1], 'w').write('ran')", str(marker)],
        )
        self.assertEqual(registry.get_task(queued["id"])["state"], "queued")

        summary = supervisor.reconcile()

        self.assertEqual(registry.get_task(holder["id"])["state"], "failed")
        self.assertIn(holder["id"], summary["failed"])
        self.wait_for(marker.exists)
        self.assertIn(
            registry.get_task(queued["id"])["state"], {"running", "failed"}
        )


class MissingProcessTests(RecoveryTestCase):
    def test_stale_starting_row_without_a_pid_is_failed_with_a_summary(self) -> None:
        task = self.create()
        self.assertEqual(task["state"], "starting")

        supervisor, registry, service = self.restart(alive=DEAD)
        summary = supervisor.reconcile()

        current = registry.get_task(task["id"])
        self.assertEqual(current["state"], "failed")
        self.assertEqual(current["error_code"], "worker_missing")
        self.assertIn(task["id"], summary["failed"])
        payload = self.event_payloads(task["id"], service)[-1]
        self.assertEqual(payload["state"], "failed")
        self.assertIn("never reported completion", payload["summary"])
        self.assertTrue(payload["recovered"])

    def test_active_row_without_a_pid_is_failed_and_never_completed(self) -> None:
        task = self.running()

        supervisor, registry, service = self.restart(alive=DEAD)
        supervisor.reconcile()

        current = registry.get_task(task["id"])
        self.assertEqual(current["state"], "failed")
        self.assertNotEqual(current["state"], "completed")
        self.assertEqual(current["error_code"], "worker_missing")
        self.assertEqual(
            [row for row in self.worker_rows() if row[0] == task["id"]], []
        )

    def test_reconcile_launches_nothing_and_writes_no_log(self) -> None:
        task = self.create()
        supervisor, registry, service = self.restart(alive=DEAD)

        supervisor.reconcile()

        self.assertEqual(self.worker_rows(), [])
        self.assertFalse(self.log_dir.exists())
        self.assertEqual(registry.get_task(task["id"])["state"], "failed")


class DeadProcessTests(RecoveryTestCase):
    def test_dead_pid_with_a_zero_exit_receipt_is_failed_not_completed(self) -> None:
        task = self.running()
        log = self.log_dir / "zero.log"
        self.write_receipt(log, task_id=task["id"], pid=4242, exit_code=0)
        self.record_process(task, 4242, log_path=log)

        supervisor, registry, service = self.restart(alive=DEAD)
        summary = supervisor.reconcile()

        current = registry.get_task(task["id"])
        self.assertEqual(current["state"], "failed")
        self.assertNotEqual(current["state"], "completed")
        self.assertEqual(current["error_code"], "worker_exit_zero")
        self.assertIn(task["id"], summary["failed"])
        payload = self.event_payloads(task["id"], service)[-1]
        self.assertEqual(payload["state"], "failed")
        self.assertIn("without reporting completion", payload["summary"])
        self.assertEqual(payload["recovery"]["exit_code"], 0)
        self.assertTrue(payload["recovery"]["receipt"])
        self.assertTrue(payload["recovered"])

    def test_dead_pid_with_a_nonzero_receipt_is_failed_with_its_status(self) -> None:
        task = self.running()
        log = self.log_dir / "seven.log"
        self.write_receipt(log, task_id=task["id"], pid=4243, exit_code=7)
        self.record_process(task, 4243, log_path=log)

        supervisor, registry, service = self.restart(alive=DEAD)
        supervisor.reconcile()

        current = registry.get_task(task["id"])
        self.assertEqual(current["state"], "failed")
        self.assertEqual(current["error_code"], "worker_exit_nonzero")
        self.assertIn("7", current["error_detail"])
        payload = self.event_payloads(task["id"], service)[-1]
        self.assertIn("status 7", payload["summary"])
        self.assertEqual(payload["recovery"]["exit_code"], 7)

    def test_dead_pid_without_a_receipt_is_failed_as_lost(self) -> None:
        task = self.running()
        self.record_process(task, 4244)

        supervisor, registry, service = self.restart(alive=DEAD)
        supervisor.reconcile()

        current = registry.get_task(task["id"])
        self.assertEqual(current["state"], "failed")
        self.assertEqual(current["error_code"], "worker_lost")
        self.assertIn("no receipt", current["error_detail"])
        self.assertFalse(self.event_payloads(task["id"], service)[-1]["recovery"]["receipt"])

    def test_dead_worker_holding_needs_approval_becomes_failed(self) -> None:
        task = self.running()
        task = self.service.transition(
            task["id"], "needs_approval", expected_version=task["version"]
        )
        self.record_process(task, 4245)

        supervisor, registry, service = self.restart(alive=DEAD)
        summary = supervisor.reconcile()

        current = registry.get_task(task["id"])
        self.assertEqual(current["state"], "failed")
        self.assertIn(task["id"], summary["failed"])
        self.assertEqual(
            service.task_events("conv-a", task["id"])[-1]["kind"], "task.failed"
        )

    def test_dead_worker_holding_needs_input_becomes_failed(self) -> None:
        task = self.running()
        task = self.service.transition(
            task["id"], "needs_input", expected_version=task["version"]
        )
        self.record_process(task, 4246)

        supervisor, registry, service = self.restart(alive=DEAD)
        supervisor.reconcile()

        self.assertEqual(registry.get_task(task["id"])["state"], "failed")

    def test_kill_and_restart_integration_marks_the_task_failed(self) -> None:
        child = self.spawn_sleep(30)
        task = self.running()
        task = self.record_process(task, child.pid, log_path=self.log_dir / "killed.log")

        os.killpg(os.getpgid(child.pid), signal.SIGKILL)
        child.wait(timeout=10)

        supervisor, registry, service = self.restart()
        summary = supervisor.reconcile()

        current = registry.get_task(task["id"])
        self.assertEqual(current["state"], "failed")
        self.assertEqual(current["error_code"], "worker_lost")
        self.assertIn(task["id"], summary["failed"])
        self.assertEqual(
            [row[2] for row in self.worker_rows() if row[0] == task["id"]], ["failed"]
        )

    def test_terminal_rows_are_untouched(self) -> None:
        task = self.running()
        completed = self.service.transition(
            task["id"],
            "completed",
            expected_version=task["version"],
            evidence=completion_evidence(self.root, [], allow_empty=True),
        )
        before_events = len(self.event_payloads(task["id"]))

        supervisor, registry, service = self.restart(alive=DEAD)
        summary = supervisor.reconcile()

        current = registry.get_task(task["id"])
        self.assertEqual(current["state"], "completed")
        self.assertEqual(current["version"], completed["version"])
        self.assertEqual(
            len(self.event_payloads(task["id"], service)), before_events
        )
        self.assertEqual(summary["scanned"], 0)
        self.assertEqual(self.worker_rows(), [])


class AdoptedProcessTests(RecoveryTestCase):
    def test_live_pid_is_adopted_and_monitored_until_it_exits(self) -> None:
        child = self.spawn_sleep(0.4)
        task = self.running()
        task = self.record_process(task, child.pid, log_path=self.log_dir / "adopted.log")

        supervisor, registry, service = self.restart()

        summary = supervisor.reconcile()
        self.assertIn(task["id"], summary["adopted"])
        self.assertIsNone(child.poll(), "reconcile must not signal or reap the child")
        self.assertEqual(registry.get_task(task["id"])["state"], "running")

        self.wait_for(lambda: registry.get_task(task["id"])["state"] == "failed")
        child.wait(timeout=10)
        current = registry.get_task(task["id"])
        self.assertEqual(current["error_code"], "worker_lost")
        self.assertEqual(
            [row[2] for row in self.worker_rows() if row[0] == task["id"]], ["failed"]
        )

    def test_adopted_worker_applies_the_receipt_its_successor_writes(self) -> None:
        child = self.spawn_sleep(0.3)
        log = self.log_dir / "receipt-on-exit.log"
        task = self.running()
        task = self.record_process(task, child.pid, log_path=log)

        supervisor, registry, service = self.restart()
        supervisor.reconcile()
        self.write_receipt(log, task_id=task["id"], pid=child.pid, exit_code=0)

        self.wait_for(lambda: registry.get_task(task["id"])["state"] == "failed")
        self.assertEqual(registry.get_task(task["id"])["error_code"], "worker_exit_zero")
        child.wait(timeout=10)

    def test_a_second_reconcile_does_not_add_a_duplicate_watcher(self) -> None:
        child = self.spawn_sleep(5)
        task = self.running()
        task = self.record_process(task, child.pid, log_path=self.log_dir / "twice.log")

        supervisor, registry, service = self.restart()
        first = supervisor.reconcile()
        second = supervisor.reconcile()

        self.assertIn(task["id"], first["adopted"])
        self.assertEqual(second["adopted"], [])
        self.assertIn(task["id"], second["watching"])
        watchers = [
            thread
            for thread in threading.enumerate()
            if thread.name == "neo-adopt-%s" % task["id"]
        ]
        self.assertEqual(len(watchers), 1)
        self.assertEqual(registry.get_task(task["id"])["state"], "running")

    def test_reconcile_never_signals_a_live_process(self) -> None:
        task = self.running()
        self.record_process(task, 4247)
        supervisor, registry, service = self.restart(alive=lambda pid: True)

        with mock.patch.object(
            workers.os, "killpg", side_effect=AssertionError("no signal allowed")
        ), mock.patch.object(
            ProcessSupervisor,
            "_kill_group",
            side_effect=AssertionError("no signal allowed"),
        ):
            summary = supervisor.reconcile()

        self.assertIn(task["id"], summary["adopted"])
        self.assertEqual(registry.get_task(task["id"])["state"], "running")

    def test_close_stops_the_monitor_without_killing_the_child(self) -> None:
        child = self.spawn_sleep(5)
        task = self.running()
        self.record_process(task, child.pid, log_path=self.log_dir / "close.log")
        supervisor, registry, service = self.restart()
        supervisor.reconcile()

        supervisor.close()

        self.assertIsNone(child.poll())
        self.assertEqual(registry.get_task(task["id"])["state"], "running")


class ProcessIdentityTests(RecoveryTestCase):
    def test_a_live_process_is_alive_and_a_reaped_one_is_not(self) -> None:
        child = self.spawn_sleep(5)
        self.assertTrue(process_alive(child.pid))
        os.killpg(os.getpgid(child.pid), signal.SIGKILL)
        child.wait(timeout=10)
        self.assertFalse(process_alive(child.pid))

    def test_a_zombie_process_counts_as_dead(self) -> None:
        child = subprocess.Popen(
            [sys.executable, "-c", "pass"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            deadline = time.monotonic() + 5.0
            state = ""
            while time.monotonic() < deadline:
                state = subprocess.run(
                    ["ps", "-o", "stat=", "-p", str(child.pid)],
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    text=True,
                    check=False,
                ).stdout.strip()
                if state.startswith("Z"):
                    break
                time.sleep(0.02)
            else:
                self.skipTest("could not observe a zombie process on this host")
            self.assertFalse(process_alive(child.pid))
        finally:
            child.wait(timeout=10)

    def test_invalid_pids_are_refused(self) -> None:
        for value in (0, -1, True, "4242", 1.5):
            with self.subTest(pid=value):
                with self.assertRaises(ValueError):
                    process_alive(value)  # type: ignore[arg-type]


class ResourceAuditTests(RecoveryTestCase):
    def test_recovering_a_dead_holder_reclaims_the_lock_with_an_audit_event(self) -> None:
        holder = self.running(resource_key="blender:9876")
        holder = self.record_process(holder, 4248)

        blocked = self.create(resource_key="blender:9876", intent="blocked")
        self.assertEqual(blocked["state"], "queued")

        supervisor, registry, service = self.restart(alive=DEAD)
        summary = supervisor.reconcile()

        self.assertIn(holder["id"], summary["failed"])
        payload = self.event_payloads(holder["id"], service)[-1]
        self.assertTrue(payload["recovered"])
        self.assertEqual(payload["recovery"]["reason"], "worker_lost")
        self.assertEqual(payload["recovery"]["resource_key"], "blender:9876")
        self.assertTrue(payload["recovery"]["resource_released"])

        following = self.create(resource_key="blender:9876", intent="now free")
        self.assertEqual(following["state"], "starting")


class CorruptRowTests(RecoveryTestCase):
    def test_a_corrupt_row_is_reported_and_other_tasks_still_recover(self) -> None:
        corrupt = self.running(intent="corrupt")
        healthy = self.running(intent="healthy")
        healthy = self.record_process(healthy, 4249)
        with sqlite3.connect(self.db_path) as connection:
            connection.execute(
                "UPDATE tasks SET pid = 'not-a-pid' WHERE id = ?", (corrupt["id"],)
            )

        supervisor, registry, service = self.restart(alive=DEAD)
        summary = supervisor.reconcile()

        self.assertEqual(registry.get_task(healthy["id"])["state"], "failed")
        self.assertEqual(registry.get_task(corrupt["id"])["state"], "running")
        self.assertEqual(
            [entry["task_id"] for entry in summary["errors"]], [corrupt["id"]]
        )
        self.assertEqual(summary["failed"], [healthy["id"]])

    def test_corrupt_expected_json_does_not_crash_reconcile(self) -> None:
        task = self.running()
        task = self.record_process(task, 4250)
        with sqlite3.connect(self.db_path) as connection:
            connection.execute(
                "UPDATE tasks SET expected_json = '{oops' WHERE id = ?", (task["id"],)
            )

        supervisor, registry, service = self.restart(alive=DEAD)
        summary = supervisor.reconcile()

        self.assertEqual(summary["errors"], [])
        self.assertEqual(registry.get_task(task["id"])["state"], "failed")

    def test_a_corrupt_row_never_rolls_back_an_unrelated_recovery(self) -> None:
        corrupt = self.running(intent="corrupt")
        healthy = self.running(intent="healthy")
        healthy = self.record_process(healthy, 4251)
        with sqlite3.connect(self.db_path) as connection:
            connection.execute(
                "UPDATE tasks SET pid = 'not-a-pid' WHERE id = ?", (corrupt["id"],)
            )

        supervisor, registry, service = self.restart(alive=DEAD)
        supervisor.reconcile()

        healthy_events = self.event_payloads(healthy["id"], service)
        self.assertEqual(healthy_events[-1]["state"], "failed")
        self.assertTrue(healthy_events[-1]["recovered"])
        self.assertEqual(
            self.event_payloads(corrupt["id"], service)[-1]["state"], "running"
        )


class IdempotenceTests(RecoveryTestCase):
    def test_a_second_reconcile_finds_nothing_left_to_recover(self) -> None:
        task = self.running()
        self.record_process(task, 4252)
        supervisor, registry, service = self.restart(alive=DEAD)

        first = supervisor.reconcile()
        events_after_first = len(self.event_payloads(task["id"], service))
        second = supervisor.reconcile()

        self.assertEqual(first["failed"], [task["id"]])
        self.assertEqual(second["scanned"], 0)
        self.assertEqual(second["failed"], [])
        self.assertEqual(second["errors"], [])
        self.assertEqual(
            len(self.event_payloads(task["id"], service)), events_after_first
        )
        self.assertEqual(registry.get_task(task["id"])["state"], "failed")


class SurvivesRestartTests(RecoveryTestCase):
    def test_session_ids_and_conversation_keys_survive_close_and_reopen(self) -> None:
        task = self.running(conversation="conv-survives")
        task = self.service.transition(
            task["id"], "completed", expected_version=task["version"],
            evidence=completion_evidence(self.root, [], allow_empty=True),
        )
        task = self.registry.update_task(
            task["id"],
            expected_version=task["version"],
            executor_session_id="0199c0de-session",
            executor="codex",
        )

        reopened = Registry(path=self.db_path, max_parallel=self.max_parallel)
        service = TaskService(reopened)
        supervisor = self.supervisor(alive=DEAD, service=service)
        supervisor.reconcile()

        survived = reopened.get_task(task["id"])
        self.assertEqual(survived["executor_session_id"], "0199c0de-session")
        self.assertEqual(survived["conversation_key"], "conv-survives")
        self.assertEqual(survived["state"], "completed")
        self.assertEqual(survived["version"], task["version"])

    def test_a_completed_result_is_presented_once_across_two_services_and_a_restart(self) -> None:
        workdir = self.root / "work"
        workdir.mkdir()
        (workdir / "report.md").write_text("# done", encoding="utf-8")
        task = self.create()
        task = self.registry.update_task(
            task["id"],
            expected_version=task["version"],
            workdir=str(workdir),
        )
        task = self.service.transition(
            task["id"], "running", expected_version=task["version"]
        )
        task = self.service.transition(
            task["id"],
            "completed",
            expected_version=task["version"],
            evidence=completion_evidence(workdir, ["report.md"]),
        )

        opened = []

        def opener(path):
            opened.append(path)

        first = PresentationService(opener, ledger=self.registry).present(task)
        restarted = Registry(path=self.db_path, max_parallel=self.max_parallel)
        second = PresentationService(opener, ledger=restarted).present(task)
        supervisor, registry, service = self.restart(alive=DEAD)
        supervisor.reconcile()
        third = PresentationService(opener, ledger=registry).present(task)

        self.assertTrue(first.presented)
        self.assertFalse(second.presented)
        self.assertFalse(third.presented)
        self.assertEqual(len(opened), 1)
        self.assertEqual(registry.get_task(task["id"])["state"], "completed")


class RuntimeBuildRecoveryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        self.root = Path(self.tempdir.name)

    def test_build_calls_reconcile_once_and_launches_nothing(self) -> None:
        from conversation import NeoRuntime

        class Recorder:
            def __init__(self):
                self.calls = 0

            def reconcile(self):
                self.calls += 1
                return {}

            def close(self):
                return None

        supervisor = Recorder()
        launches = []
        runtime = NeoRuntime.build(
            data_dir=self.root / "plugin-data",
            supervisor=supervisor,
            launch=lambda *args, **kwargs: launches.append(args),
            max_parallel=4,
        )
        self.addCleanup(runtime.supervisor.close)
        self.assertEqual(supervisor.calls, 1)
        self.assertEqual(launches, [])

    def test_build_recovers_a_dead_row_from_a_previous_run(self) -> None:
        from conversation import NeoRuntime

        data = self.root / "plugin-data"
        data.mkdir(parents=True)
        registry = Registry(path=data / "neo.db", max_parallel=4)
        service = TaskService(registry)
        task = service.create_task(
            conversation_key="conv-a", kind="general", intent="crashed"
        )
        task = service.transition(task["id"], "running", expected_version=task["version"])

        process = subprocess.Popen([sys.executable, "-c", "pass"])
        process.wait(timeout=10)
        task = registry.update_task(
            task["id"],
            expected_version=task["version"],
            pid=process.pid,
            pgid=process.pid,
            log_path=str(data / "logs" / "crashed.log"),
        )

        started = time.monotonic()
        runtime = NeoRuntime.build(data_dir=data, max_parallel=4)
        self.addCleanup(runtime.supervisor.close)
        elapsed = time.monotonic() - started

        recovered = runtime.registry.get_task(task["id"])
        self.assertEqual(recovered["state"], "failed")
        self.assertIsNotNone(recovered["error_code"])
        self.assertLess(elapsed, 5.0, "startup reconciliation must stay bounded")
        with sqlite3.connect(data / "neo.db") as connection:
            self.assertEqual(
                connection.execute("SELECT COUNT(*) FROM workers").fetchone()[0], 0
            )

    def test_build_tolerates_a_supervisor_without_reconcile(self) -> None:
        from conversation import NeoRuntime

        supervisor = object()
        runtime = NeoRuntime.build(
            data_dir=self.root / "plugin-data", supervisor=supervisor, max_parallel=4
        )
        self.assertIs(runtime.supervisor, supervisor)


if __name__ == "__main__":
    unittest.main()
