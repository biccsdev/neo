"""Task 7 — detached, durable process supervision."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from events import TaskService  # noqa: E402
from registry import Registry  # noqa: E402
from state_machine import CompletionEvidence  # noqa: E402
from tools import WorkerTokenStore  # noqa: E402
from workers import ProcessSupervisor, RecursionRefusedError  # noqa: E402


class ProcessSupervisorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.registry = Registry(path=self.root / "registry.db", max_parallel=1)
        self.service = TaskService(self.registry)
        self.tokens = WorkerTokenStore(self.registry)
        self.supervisor = ProcessSupervisor(
            self.service, self.tokens, log_dir=self.root / "logs"
        )

    def tearDown(self) -> None:
        self.supervisor.close()
        self.tempdir.cleanup()

    def create_task(self, name: str = "task") -> dict:
        return self.service.create_task(
            conversation_key="conversation", kind="general", intent=name
        )

    def wait_for(self, predicate, timeout: float = 5.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            value = predicate()
            if value:
                return value
            time.sleep(0.02)
        self.fail("condition was not met within %.1fs" % timeout)

    def test_submit_requires_a_nonempty_argv_array(self) -> None:
        task = self.create_task()
        for command in ("echo unsafe", "", [], None):
            with self.subTest(command=command):
                with self.assertRaises(ValueError):
                    self.supervisor.submit(task["id"], command)
        self.assertEqual(self.registry.get_task(task["id"]), task)
        self.assertIsNone(self.supervisor.get_worker(task["id"]))

    def test_recursion_guard_refuses_a_nested_worker(self) -> None:
        task = self.create_task()
        with mock.patch.dict(os.environ, {"NEO_WORKER": "1"}):
            with self.assertRaises(RecursionRefusedError):
                self.supervisor.submit(task["id"], [sys.executable, "-c", "pass"])
        self.assertEqual(self.registry.get_task(task["id"])["state"], "starting")

    def test_spawn_persists_detached_identity_log_and_receipt(self) -> None:
        task = self.create_task()
        env_path = self.root / "child-env.json"
        code = (
            "import json, os, sys; "
            "open(sys.argv[1], 'w').write(json.dumps({key: os.environ.get(key) "
            "for key in ('NEO_WORKER', 'NEO_TASK_ID', 'NEO_TASK_TOKEN')})); "
            "print('worker-stdout', flush=True); import time; time.sleep(0.2)"
        )

        submitted = self.supervisor.submit(
            task["id"], [sys.executable, "-c", code, str(env_path)]
        )
        self.assertEqual(submitted["state"], "running")
        worker = self.supervisor.get_worker(task["id"])
        recorded = self.registry.get_task(task["id"])
        self.assertEqual(recorded["pid"], worker["pid"])
        self.assertEqual(recorded["pgid"], worker["pgid"])
        self.assertEqual(recorded["log_path"], worker["log_path"])
        self.assertEqual(recorded["heartbeat_at"], worker["heartbeat_at"])
        self.assertEqual(worker["pid"], worker["pgid"])
        self.assertEqual(os.getpgid(worker["pid"]), worker["pgid"])
        self.assertEqual(os.getsid(worker["pid"]), worker["pid"])
        self.assertIsNotNone(worker["heartbeat_at"])
        self.assertTrue(Path(worker["log_path"]).is_file())

        receipt_path = Path(worker["log_path"]).with_suffix(".receipt.json")
        self.wait_for(receipt_path.exists)
        receipt = json.loads(receipt_path.read_text())
        self.assertEqual(receipt["pid"], worker["pid"])
        self.assertEqual(receipt["exit_code"], 0)
        environment = json.loads(env_path.read_text())
        self.assertEqual(environment["NEO_WORKER"], "1")
        self.assertEqual(environment["NEO_TASK_ID"], task["id"])
        self.assertTrue(self.tokens.verify(task["id"], environment["NEO_TASK_TOKEN"]))
        self.assertNotIn(environment["NEO_TASK_TOKEN"], Path(worker["log_path"]).read_text())
        self.assertIn("worker-stdout", Path(worker["log_path"]).read_text())
        self.wait_for(lambda: self._task_in_state(task["id"], "failed"))

    def test_queued_argv_drains_when_capacity_is_released(self) -> None:
        first = self.create_task("first")
        second = self.create_task("second")
        marker = self.root / "queued-ran.txt"
        self.assertEqual(first["state"], "starting")
        self.assertEqual(second["state"], "queued")

        self.supervisor.submit(
            first["id"], [sys.executable, "-c", "import time; time.sleep(0.15)"]
        )
        queued = self.supervisor.submit(
            second["id"],
            [sys.executable, "-c", "import sys; open(sys.argv[1], 'w').write('ran')", str(marker)],
        )
        self.assertEqual(queued["state"], "queued")
        self.assertIsNone(self.supervisor.get_worker(second["id"]))
        self.assertFalse(marker.exists())

        def second_receipt():
            worker = self.supervisor.get_worker(second["id"])
            if worker is None:
                return False
            return Path(worker["log_path"]).with_suffix(".receipt.json").exists()

        self.wait_for(second_receipt)
        self.assertEqual(marker.read_text(), "ran")
        self.wait_for(lambda: self._task_in_state(second["id"], "failed"))

    def test_external_capacity_release_drains_a_pending_task(self) -> None:
        holder = self.create_task("holder")
        waiter = self.create_task("waiter")
        marker = self.root / "externally-released.txt"
        self.supervisor.submit(
            waiter["id"],
            [sys.executable, "-c", "import sys; open(sys.argv[1], 'w').write('ran')", str(marker)],
        )
        self.assertEqual(self.registry.get_task(waiter["id"])["state"], "queued")

        running = self.registry.get_task(holder["id"])
        running = self.service.transition(
            holder["id"], "running", expected_version=running["version"]
        )
        self.service.transition(
            holder["id"],
            "completed",
            expected_version=running["version"],
            evidence=CompletionEvidence(executor_confirmed=True),
        )

        self.wait_for(
            lambda: self.supervisor.get_worker(waiter["id"]) is not None
        )
        self.wait_for(marker.exists)
        self.assertIn(
            self.registry.get_task(waiter["id"])["state"], {"running", "failed"}
        )
        self.assertEqual(marker.read_text(), "ran")

    def test_spawn_failure_releases_capacity_for_the_next_task(self) -> None:
        failed_task = self.create_task("will-fail")
        next_task = self.create_task("next")

        failed = self.supervisor.submit(
            failed_task["id"], [str(self.root / "missing-binary")]
        )
        self.assertEqual(failed["state"], "failed")
        queued = self.supervisor.submit(
            next_task["id"], [sys.executable, "-c", "pass"]
        )
        self.assertIn(queued["state"], {"running", "failed"})
        self.wait_for(
            lambda: self.supervisor.get_worker(next_task["id"]) is not None
        )

    def test_invalid_log_directory_fails_task_and_releases_capacity(self) -> None:
        blocker = self.root / "not-a-directory"
        blocker.write_text("file")
        supervisor = ProcessSupervisor(
            self.service, self.tokens, log_dir=blocker
        )
        self.addCleanup(supervisor.close)
        task = self.create_task("bad-log")

        failed = supervisor.submit(task["id"], [sys.executable, "-c", "pass"])

        self.assertEqual(failed["state"], "failed")
        self.assertEqual(
            self.registry.get_task(task["id"])["state"], "failed"
        )
        following = self.create_task("following")
        self.assertEqual(following["state"], "starting")

    def test_embedded_nul_argv_is_failed_not_stranded(self) -> None:
        task = self.create_task()

        failed = self.supervisor.submit(
            task["id"], [sys.executable, "-c", "pass\0hidden"]
        )

        self.assertEqual(failed["state"], "failed")
        self.assertIsNone(self.supervisor.get_worker(task["id"])["pid"])

    def test_watcher_survives_a_running_transition_fault(self) -> None:
        task = self.create_task()
        original = self.service.transition

        def faulty(task_id, target, **kwargs):
            if target == "running":
                raise RuntimeError("simulated post-commit fault")
            return original(task_id, target, **kwargs)

        with mock.patch.object(self.service, "transition", faulty):
            self.supervisor.submit(
                task["id"], [sys.executable, "-c", "import time; time.sleep(0.1)"]
            )

        self.wait_for(lambda: self._task_in_state(task["id"], "failed"))
        worker = self.supervisor.get_worker(task["id"])
        receipt = Path(worker["log_path"]).with_suffix(".receipt.json")
        self.assertTrue(receipt.exists())

    def test_cancelling_a_completed_task_is_idempotent(self) -> None:
        task = self.create_task()
        self.supervisor.submit(
            task["id"], [sys.executable, "-c", "import time; time.sleep(0.2)"]
        )
        running = self.registry.get_task(task["id"])
        completed = self.service.transition(
            task["id"],
            "completed",
            expected_version=running["version"],
            evidence=CompletionEvidence(executor_confirmed=True),
        )

        self.assertEqual(self.supervisor.cancel(task["id"]), completed)
        self.wait_for(
            lambda: self.supervisor.get_worker(task["id"])["state"] == "completed"
        )

    def test_close_leaves_the_detached_worker_and_watcher_running(self) -> None:
        task = self.create_task()
        marker = self.root / "detached-finished.txt"
        self.supervisor.submit(
            task["id"],
            [sys.executable, "-c", "import sys,time; time.sleep(0.2); open(sys.argv[1], 'w').write('done')", str(marker)],
        )
        worker = self.supervisor.get_worker(task["id"])
        self.supervisor.close()

        self.wait_for(marker.exists)
        receipt = Path(worker["log_path"]).with_suffix(".receipt.json")
        self.wait_for(receipt.exists)
        self.assertEqual(json.loads(receipt.read_text())["exit_code"], 0)

    def test_queued_cancellation_never_claims_process_execution(self) -> None:
        self.create_task("capacity-holder")
        queued = self.create_task("queued")
        marker = self.root / "must-not-run.txt"
        self.supervisor.submit(
            queued["id"],
            [sys.executable, "-c", "import sys; open(sys.argv[1], 'w').write('ran')", str(marker)],
        )

        cancelled = self.supervisor.cancel(queued["id"])

        self.assertEqual(cancelled["state"], "cancelled")
        self.assertIsNone(cancelled["started_at"])
        self.assertIsNone(self.supervisor.get_worker(queued["id"]))
        self.supervisor.drain_pending()
        self.assertFalse(marker.exists())

    def test_active_cancellation_escalates_to_sigkill_after_process_group_term(self) -> None:
        task = self.create_task()
        ready = self.root / "ready.txt"
        termed = self.root / "termed.txt"
        code = (
            "import os, signal, sys, time; "
            "signal.signal(signal.SIGTERM, lambda signum, frame: open(sys.argv[2], 'w').write('term')); "
            "open(sys.argv[1], 'w').write(str(os.getpid())); "
            "exec('while True:\\n time.sleep(1)')"
        )
        self.supervisor.submit(
            task["id"], [sys.executable, "-c", code, str(ready), str(termed)]
        )
        self.wait_for(ready.exists)
        worker = self.supervisor.get_worker(task["id"])
        original_transition = self.service.transition

        def checked_transition(task_id, target, **kwargs):
            if target == "cancelled":
                with self.assertRaises(ProcessLookupError):
                    os.kill(worker["pid"], 0)
            return original_transition(task_id, target, **kwargs)

        with mock.patch.object(self.service, "transition", checked_transition):
            cancelled = self.supervisor.cancel(task["id"], grace=0.05)

        self.assertEqual(cancelled["state"], "cancelled")
        self.assertTrue(termed.exists())
        receipt_path = Path(worker["log_path"]).with_suffix(".receipt.json")
        self.wait_for(receipt_path.exists)
        receipt = json.loads(receipt_path.read_text())
        self.assertTrue(receipt["term_sent"])
        self.assertTrue(receipt["kill_sent"])
        self.assertEqual(receipt["exit_code"], -signal.SIGKILL)
        self.assertEqual(self.supervisor.get_worker(task["id"])["state"], "cancelled")

    def test_spawn_failure_is_failed_not_queued_or_running(self) -> None:
        task = self.create_task()

        failed = self.supervisor.submit(
            task["id"], [str(self.root / "does-not-exist"), "unsafe; text"]
        )

        self.assertEqual(failed["state"], "failed")
        worker = self.supervisor.get_worker(task["id"])
        self.assertEqual(worker["state"], "failed")
        self.assertIsNone(worker["pid"])
        receipt_path = Path(worker["log_path"]).with_suffix(".receipt.json")
        receipt = json.loads(receipt_path.read_text())
        self.assertIsNone(receipt["exit_code"])
        self.assertEqual(receipt["error"], "spawn_failed")
        self.assertIn("spawn failed", Path(worker["log_path"]).read_text())

    def test_nonzero_worker_exit_marks_task_failed(self) -> None:
        task = self.create_task()
        self.supervisor.submit(
            task["id"], [sys.executable, "-c", "import sys; sys.exit(7)"]
        )

        failed = self.wait_for(
            lambda: self._task_in_state(task["id"], "failed")
        )
        self.assertEqual(failed["state"], "failed")
        worker = self.supervisor.get_worker(task["id"])
        self.assertEqual(worker["state"], "failed")
        receipt = json.loads(
            Path(worker["log_path"]).with_suffix(".receipt.json").read_text()
        )
        self.assertEqual(receipt["exit_code"], 7)

    def _task_in_state(self, task_id: str, state: str):
        task = self.registry.get_task(task_id)
        return task if task["state"] == state else None

    def test_launcher_execs_exact_argv_without_shell_interpolation(self) -> None:
        launcher = REPO_ROOT / "resources" / "neo-launch-macos.sh"
        output = self.root / "argv.json"
        marker = self.root / "shell-injected.txt"
        hostile = "plain; touch %s; $(touch %s) # arguments" % (marker, marker)
        result = subprocess.run(
            [
                str(launcher),
                sys.executable,
                "-c",
                "import json, sys; open(sys.argv[2], 'w').write(json.dumps(sys.argv[1]))",
                hostile,
                str(output),
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=5,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(output.read_text()), hostile)
        self.assertFalse(marker.exists())


if __name__ == "__main__":
    unittest.main()
