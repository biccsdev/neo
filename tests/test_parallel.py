"""Task 21 — parallel task integration.

One registry, one supervisor and one runtime carry several kinds of work at
once. These tests pin the integration rules that no single-slice suite can
see:

* A coding task and a Blender task (or a browser task) start concurrently,
  each with its own task id, process record and log file.
* Two Blender tasks share ``blender:9876``: the second queues even when
  capacity is free and only starts once the first is terminal.
* Cancelling one task neither touches its neighbour nor blocks the drain of a
  queued task; the survivor finishes on its own evidence.
* Conversation reads and events stay inside their conversation.
* The dashboard list and its filters carry every kind and state.

The coding executor and both slices are fakes injected into the real runtime;
every launched argv is a tiny Python recorder or gate-waiter, so no Codex,
Claude, Hermes, Blender or browser process is ever started. Nothing is
installed and no network is touched.
"""

from __future__ import annotations

import json
import os
import signal
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DASHBOARD_DIR = REPO_ROOT / "dashboard"
if str(DASHBOARD_DIR) not in sys.path:
    sys.path.insert(0, str(DASHBOARD_DIR))

try:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
except ModuleNotFoundError:  # pragma: no cover - system python lacks fastapi
    FastAPI = None
    TestClient = None

from conversation import ConversationTools, NeoRuntime  # noqa: E402
from events import TaskService  # noqa: E402
from registry import Registry  # noqa: E402
from state_machine import CompletionEvidence  # noqa: E402

PREFIX = "/api/plugins/neo"
ACTIVE_STATES = frozenset(
    {
        "queued",
        "starting",
        "running",
        "waiting",
        "needs_approval",
        "needs_input",
        "cancelling",
        "unknown",
    }
)
TERMINAL_STATES = frozenset({"completed", "failed", "cancelled"})

# A recorder, gate-waiter and exit-code carrier in one: it proves which task
# and argv reached the process boundary, then holds the task alive until the
# test releases it. ``NEO_TASK_ID`` is set by the supervisor.
WORKER_STUB = r"""
import json, os, sys, time

record_path, gate, exit_code = sys.argv[1], sys.argv[2], sys.argv[3]
with open(record_path, "w", encoding="utf-8") as handle:
    json.dump(
        {
            "task_id": os.environ.get("NEO_TASK_ID"),
            "pid": os.getpid(),
            "argv": sys.argv[4:],
        },
        handle,
    )

deadline = time.time() + 30
while not os.path.exists(gate):
    if time.time() > deadline:
        break
    time.sleep(0.01)
sys.exit(int(exit_code))
"""


class FakeSlice:
    """A prepared-task slice that returns a recorder argv, never a real CLI."""

    def __init__(self, kind, resource_key, expected) -> None:
        self.kind = kind
        self.resource_key = resource_key
        self.expected = tuple(expected)
        self.calls = []

    def record_path(self, workdir, task_id) -> Path:
        return Path(workdir) / ("%s.launch.json" % task_id)

    def gate_path(self, workdir, task_id) -> Path:
        return Path(workdir) / ("%s.gate" % task_id)

    def prepare_task(self, intent, conversation_key, workdir=None, *, task_id=None):
        self.calls.append(
            {
                "intent": intent,
                "conversation": conversation_key,
                "workdir": workdir,
                "task_id": task_id,
            }
        )
        if not isinstance(workdir, str) or not Path(workdir).is_dir():
            raise ValueError("a slice task needs an existing workdir")
        return {
            "kind": self.kind,
            "capability": self.kind,
            "executor": "hermes_worker",
            "resource_key": self.resource_key,
            "expected_artifacts": self.expected,
            "prompt": "fake %s work order for %s" % (self.kind, intent),
            "argv": (
                sys.executable,
                "-c",
                WORKER_STUB,
                str(self.record_path(workdir, task_id)),
                str(self.gate_path(workdir, task_id)),
                "0",
                "fake-%s" % self.kind,
                intent,
            ),
        }


class FakeCodingExecutor:
    """Builds the same recorder argv for a coding task; no Codex is present."""

    def __init__(self) -> None:
        self.calls = []

    @staticmethod
    def record_path(workdir) -> Path:
        return Path(workdir) / "coding.record.json"

    @staticmethod
    def gate_path(workdir) -> Path:
        return Path(workdir) / "coding.gate"

    def build_initial_argv(self, intent, *, workdir=None, session_id=None):
        self.calls.append({"intent": intent, "workdir": workdir})
        return [
            sys.executable,
            "-c",
            WORKER_STUB,
            str(self.record_path(workdir)),
            str(self.gate_path(workdir)),
            "0",
            intent,
        ]


class ParallelCase(unittest.TestCase):
    """A real registry, task service and supervisor under a temp directory."""

    max_parallel = 4

    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        self.root = Path(self.tempdir.name)
        self.blender = FakeSlice("blender", "blender:9876", ("scene.blend",))
        self.browser = FakeSlice(
            "browser", None, ("browser-state.json", "navigation.png")
        )
        self.runtime = NeoRuntime.build(
            data_dir=self.root / "plugin-data",
            max_parallel=self.max_parallel,
            blender=self.blender,
            browser=self.browser,
        )
        self.codex = FakeCodingExecutor()
        self.runtime.codex = self.codex
        self.runtime.coding_executor = "codex"
        # The fake browser represents an explicitly authorised host capability.
        self.runtime.browser_authorized = True
        self.tools = ConversationTools(lambda: self.runtime)
        self.addCleanup(self._cleanup_processes)

    # -- submission and polling ------------------------------------------

    def submit(self, intent, kind, conversation="conv-a"):
        payload = self.tools.neo_submit_task(
            {"intent": intent, "kind": kind}, session_id=conversation
        )
        body = json.loads(payload)
        self.assertIsInstance(body, dict)
        self.assertTrue(body.get("ok"), body)
        return body

    def task(self, task_id: str) -> dict:
        row = self.runtime.registry.get_task(task_id)
        self.assertIsNotNone(row, "task %s disappeared" % task_id)
        return row

    def wait_for(self, predicate, message="condition", timeout=10.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            value = predicate()
            if value:
                return value
            time.sleep(0.02)
        self.fail("%s was not met within %.1fs" % (message, timeout))

    def wait_for_state(self, task_id: str, state: str) -> dict:
        return self.wait_for(
            lambda: (
                self.task(task_id) if self.task(task_id)["state"] == state else None
            ),
            message="task %s in %s (was %s)"
            % (task_id, state, self.task(task_id)["state"]),
        )

    def wait_for_worker(self, task_id: str) -> dict:
        return self.wait_for(
            lambda: self.runtime.supervisor.get_worker(task_id),
            message="worker record for %s" % task_id,
        )

    def wait_for_record(self, path) -> dict:
        def load():
            try:
                return json.loads(Path(path).read_text(encoding="utf-8"))
            except (OSError, ValueError):
                return None

        return self.wait_for(lambda: load(), message="argv record at %s" % path)

    def gate_path(self, task_id: str) -> Path:
        row = self.task(task_id)
        workdir = Path(row["workdir"])
        if row["kind"] == "coding":
            return FakeCodingExecutor.gate_path(workdir)
        return Path(workdir) / ("%s.gate" % task_id)

    def record_path(self, task_id: str) -> Path:
        row = self.task(task_id)
        workdir = Path(row["workdir"])
        if row["kind"] == "coding":
            return FakeCodingExecutor.record_path(workdir)
        return Path(workdir) / ("%s.launch.json" % task_id)

    def release(self, task_id: str) -> None:
        self.gate_path(task_id).write_text("released", encoding="utf-8")

    # -- cleanup ---------------------------------------------------------

    @staticmethod
    def _alive(pid: int) -> bool:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:  # pragma: no cover - not our own process
            return True
        return True

    def _live_worker_pids(self) -> set:
        pids = set()
        for row in self.runtime.registry.list_tasks():
            worker = self.runtime.supervisor.get_worker(row["id"])
            if worker is not None and worker.get("pid"):
                pids.add(int(worker["pid"]))
        return pids

    def _cleanup_processes(self) -> None:
        """Release every gate, cancel live tasks, then kill any survivor."""
        tasks_dir = self.root / "plugin-data" / "tasks"
        if tasks_dir.is_dir():
            for gate in tasks_dir.rglob("*.gate"):
                if not gate.exists():
                    gate.write_text("released", encoding="utf-8")
        supervisor = self.runtime.supervisor
        for row in self.runtime.registry.list_tasks():
            if row["state"] in ACTIVE_STATES:
                try:
                    supervisor.cancel(row["id"], grace=0.5)
                except Exception:
                    pass
        supervisor.close()
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            live = {pid for pid in self._live_worker_pids() if self._alive(pid)}
            if not live:
                break
            for pid in live:
                try:
                    os.killpg(pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
            time.sleep(0.02)


class CodingAndBlenderTests(ParallelCase):
    def test_coding_and_blender_start_concurrently_and_stay_isolated(self) -> None:
        self.assertGreaterEqual(self.runtime.registry.max_parallel, 2)

        coding = self.submit("write a small parser", "coding")
        blender = self.submit("make a low-poly rocket", "blender")

        self.assertEqual(coding["state"], "running", coding)
        self.assertEqual(blender["state"], "running", blender)
        self.assertNotEqual(coding["task_id"], blender["task_id"])
        self.assertEqual(coding["kind"], "coding")
        self.assertEqual(blender["kind"], "blender")
        self.assertEqual(blender["resource_key"], "blender:9876")

        coding_worker = self.wait_for_worker(coding["task_id"])
        blender_worker = self.wait_for_worker(blender["task_id"])
        self.assertNotEqual(coding_worker["id"], blender_worker["id"])
        self.assertNotEqual(coding_worker["pid"], blender_worker["pid"])
        self.assertNotEqual(coding_worker["log_path"], blender_worker["log_path"])
        self.assertEqual(
            Path(coding_worker["log_path"]).name, "%s.log" % coding["task_id"]
        )
        self.assertEqual(
            Path(blender_worker["log_path"]).name, "%s.log" % blender["task_id"]
        )
        self.assertTrue(Path(coding_worker["log_path"]).is_file())
        self.assertTrue(Path(blender_worker["log_path"]).is_file())

        coding_record = self.wait_for_record(self.record_path(coding["task_id"]))
        blender_record = self.wait_for_record(self.record_path(blender["task_id"]))
        self.assertEqual(coding_record["task_id"], coding["task_id"])
        self.assertEqual(blender_record["task_id"], blender["task_id"])
        self.assertIn("fake-blender", blender_record["argv"])
        self.assertIn("write a small parser", coding_record["argv"])

        rows = {row["id"]: row for row in self.runtime.registry.list_tasks()}
        self.assertEqual(set(rows), {coding["task_id"], blender["task_id"]})
        self.assertEqual(rows[coding["task_id"]]["state"], "running")
        self.assertEqual(rows[blender["task_id"]]["state"], "running")


class CodingAndBrowserTests(ParallelCase):
    def test_coding_and_browser_start_concurrently_and_stay_isolated(self) -> None:
        self.assertGreaterEqual(self.runtime.registry.max_parallel, 2)

        coding = self.submit("summarise the build log", "coding")
        browser = self.submit("open the status page and read it", "browser")

        self.assertEqual(coding["state"], "running", coding)
        self.assertEqual(browser["state"], "running", browser)
        self.assertNotEqual(coding["task_id"], browser["task_id"])

        browser_row = self.task(browser["task_id"])
        self.assertEqual(browser_row["kind"], "browser")
        self.assertIsNone(browser_row["resource_key"])

        coding_worker = self.wait_for_worker(coding["task_id"])
        browser_worker = self.wait_for_worker(browser["task_id"])
        self.assertNotEqual(coding_worker["pid"], browser_worker["pid"])
        self.assertNotEqual(coding_worker["log_path"], browser_worker["log_path"])

        browser_record = self.wait_for_record(self.record_path(browser["task_id"]))
        coding_record = self.wait_for_record(self.record_path(coding["task_id"]))
        self.assertEqual(browser_record["task_id"], browser["task_id"])
        self.assertEqual(coding_record["task_id"], coding["task_id"])
        self.assertIn("fake-browser", browser_record["argv"])

        states = {
            row["kind"]: row["state"] for row in self.runtime.registry.list_tasks()
        }
        self.assertEqual(states, {"coding": "running", "browser": "running"})


class BlenderQueueTests(ParallelCase):
    def test_second_blender_queues_on_the_port_then_starts_after_terminal(self) -> None:
        self.assertGreaterEqual(self.runtime.registry.max_parallel, 2)

        first = self.submit("first rocket", "blender")
        self.assertEqual(first["state"], "running", first)
        active = [
            row
            for row in self.runtime.registry.list_tasks()
            if row["state"] in ACTIVE_STATES
        ]
        self.assertEqual(len(active), 1, "capacity is free, the port is not")

        second = self.submit("second rocket", "blender")
        self.assertEqual(second["state"], "queued", second)
        self.assertEqual(self.task(second["task_id"])["resource_key"], "blender:9876")
        self.assertIsNone(self.runtime.supervisor.get_worker(second["task_id"]))
        self.assertFalse(self.record_path(second["task_id"]).exists())

        first_worker = self.wait_for_worker(first["task_id"])
        first_preview = self.task(first["task_id"])["workdir"] + "/preview.png"
        first_row = self.task(first["task_id"])
        self.runtime.registry.update_task(
            first["task_id"],
            expected_version=first_row["version"],
            preview_path=first_preview,
        )

        running_first = self.task(first["task_id"])
        self.assertEqual(running_first["state"], "running")
        completed = self.runtime.service.transition(
            first["task_id"],
            "completed",
            expected_version=running_first["version"],
            evidence=CompletionEvidence(
                executor_confirmed=True,
                expected_artifacts=("scene.blend",),
                verified_artifacts=("scene.blend",),
            ),
        )
        self.assertEqual(completed["state"], "completed")

        second_worker = self.wait_for_worker(second["task_id"])
        started = self.wait_for_state(second["task_id"], "running")
        self.assertIsNotNone(started["started_at"])
        self.assertGreaterEqual(started["started_at"], completed["finished_at"])

        self.assertNotEqual(first_worker["pid"], second_worker["pid"])
        self.assertNotEqual(first_worker["log_path"], second_worker["log_path"])
        self.assertNotEqual(first["task_id"], second["task_id"])
        self.assertNotEqual(
            self.task(first["task_id"])["workdir"],
            self.task(second["task_id"])["workdir"],
        )

        second_preview = Path(self.task(second["task_id"])["workdir"]) / "preview.png"
        second_row = self.task(second["task_id"])
        self.runtime.registry.update_task(
            second["task_id"],
            expected_version=second_row["version"],
            preview_path=str(second_preview),
        )
        self.assertNotEqual(first_preview, str(second_preview))
        self.assertIn("first rocket", json.dumps(self.calls_for(first["task_id"])))
        self.assertIn(
            second["task_id"], Path(second_worker["log_path"]).name
        )

    def calls_for(self, task_id: str) -> dict:
        """The argv record a task's own worker wrote."""
        return self.wait_for_record(self.record_path(task_id))


class CancellationTests(ParallelCase):
    max_parallel = 2

    def test_cancel_one_task_leaves_the_other_running_and_drains_the_queue(
        self,
    ) -> None:
        self.assertEqual(self.runtime.registry.max_parallel, 2)
        first = self.submit("task one", "coding")
        second = self.submit("task two", "coding")
        third = self.submit("task three", "coding")

        self.assertEqual(first["state"], "running", first)
        self.assertEqual(second["state"], "running", second)
        self.assertEqual(third["state"], "queued", third)
        self.assertIsNone(self.runtime.supervisor.get_worker(third["task_id"]))

        first_worker = self.wait_for_worker(first["task_id"])
        second_worker = self.wait_for_worker(second["task_id"])

        cancelled = self.runtime.supervisor.cancel(second["task_id"], grace=1.0)
        self.assertEqual(cancelled["state"], "cancelled")
        self.assertEqual(
            self.runtime.supervisor.get_worker(second["task_id"])["state"], "cancelled"
        )
        with self.assertRaises(ProcessLookupError):
            os.kill(second_worker["pid"], 0)

        survivor = self.task(first["task_id"])
        self.assertEqual(survivor["state"], "running")
        self.assertEqual(survivor["pid"], first_worker["pid"])
        self.assertEqual(survivor["log_path"], first_worker["log_path"])
        self.assertEqual(self.runtime.supervisor.get_worker(first["task_id"])["state"], "running")
        self.assertFalse(self.gate_path(first["task_id"]).exists())

        # The queued task was not blocked by the cancellation.
        third_worker = self.wait_for_worker(third["task_id"])
        third_row = self.wait_for_state(third["task_id"], "running")
        self.assertNotEqual(third_worker["pid"], first_worker["pid"])
        self.assertNotEqual(third_worker["log_path"], first_worker["log_path"])
        self.assertEqual(third_worker["log_path"], third_row["log_path"])

        # The survivor completes on its own evidence while the third still runs.
        survivor = self.task(first["task_id"])
        completed = self.runtime.service.transition(
            first["task_id"],
            "completed",
            expected_version=survivor["version"],
            evidence=CompletionEvidence(executor_confirmed=True),
        )
        self.assertEqual(completed["state"], "completed")
        self.assertEqual(self.task(third["task_id"])["state"], "running")

        # The third fails honestly and independently when its process ends
        # without reporting completion.
        self.release(third["task_id"])
        failed = self.wait_for_state(third["task_id"], "failed")
        self.assertEqual(failed["state"], "failed")
        terminal = self.runtime.service.task_events("conv-a", third["task_id"])[-1]
        self.assertEqual(terminal["kind"], "task.failed")
        self.assertEqual(
            terminal["payload"]["summary"],
            "worker exited without reporting completion",
        )
        self.assertEqual(self.task(second["task_id"])["state"], "cancelled")


class ListingTests(unittest.TestCase):
    """Conversation filters and the dashboard list over every kind and state."""

    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        self.root = Path(self.tempdir.name)
        self.registry = Registry(path=self.root / "neo.db", max_parallel=8)
        self.service = TaskService(self.registry)
        self.tools = ConversationTools(
            lambda: SimpleNamespace(registry=self.registry, clock=time.time)
        )

        self.ids = {}
        self.ids["coding"] = self.service.create_task(
            conversation_key="conv-a", kind="coding", intent="code the thing"
        )
        self.ids["blender"] = self.service.create_task(
            conversation_key="conv-a", kind="blender", intent="model the thing"
        )
        self.ids["browser"] = self.service.create_task(
            conversation_key="conv-a", kind="browser", intent="browse the thing"
        )
        self.ids["general"] = self.service.create_task(
            conversation_key="conv-a", kind="general", intent="answer the thing"
        )
        self.drive("coding", "running")
        self.drive("browser", "running", "waiting")
        self.drive("blender", "queued")
        # general stays in ``starting``.

    def drive(self, key: str, *states: str) -> None:
        current = self.ids[key]
        for state in states:
            current = self.service.transition(
                current["id"], state, expected_version=current["version"]
            )
        self.ids[key] = current

    def listed(self, **filters) -> list:
        payload = json.loads(self.tools.neo_task_list(filters, session_id="conv-a"))
        self.assertTrue(payload.get("ok"), payload)
        return payload["tasks"]

    def test_conversation_list_filters_cover_every_kind_and_state(self) -> None:
        everything = self.listed()
        self.assertEqual({row["kind"] for row in everything}, set(self.ids))
        self.assertEqual(
            {row["state"] for row in everything}, {"running", "waiting", "queued"}
            | {"starting"}
        )

        for kind in ("coding", "blender", "browser", "general"):
            with self.subTest(kind=kind):
                rows = self.listed(kind=kind)
                self.assertEqual(len(rows), 1)
                self.assertEqual(rows[0]["kind"], kind)

        for state in ("starting", "running", "waiting", "queued"):
            with self.subTest(state=state):
                rows = self.listed(state=state)
                self.assertTrue(rows)
                self.assertTrue(all(row["state"] == state for row in rows))

        self.assertEqual(
            [row["task_id"] for row in self.listed(kind="browser", state="waiting")],
            [self.ids["browser"]["id"]],
        )

    @unittest.skipUnless(FastAPI is not None, "FastAPI/httpx are required for the API check")
    def test_api_tasks_returns_every_kind_and_honours_filters(self) -> None:
        import plugin_api

        services = SimpleNamespace(
            registry=self.registry,
            task_service=self.service,
            preview_service=None,
            supervisor=None,
            retry=None,
            capabilities=None,
            installer=None,
            log_root=self.root / "logs",
            artifact_max_bytes=1024 * 1024,
            log_max_bytes=64 * 1024,
        )
        app = FastAPI()
        app.include_router(plugin_api.build_router(lambda: services), prefix=PREFIX)
        client = TestClient(app, raise_server_exceptions=False)

        body = client.get(PREFIX + "/tasks").json()
        self.assertEqual(body["count"], 4)
        self.assertEqual(
            {row["id"] for row in body["tasks"]},
            {row["id"] for row in self.ids.values()},
        )
        self.assertEqual(
            {row["kind"] for row in body["tasks"]}, {"coding", "blender", "browser", "general"}
        )

        blender = client.get(PREFIX + "/tasks", params={"kind": "blender"}).json()
        self.assertEqual(blender["count"], 1)
        self.assertEqual(blender["tasks"][0]["kind"], "blender")

        running = client.get(PREFIX + "/tasks", params={"state": "running"}).json()
        self.assertEqual({row["kind"] for row in running["tasks"]}, {"coding"})

        waiting = client.get(
            PREFIX + "/tasks", params={"kind": "browser", "state": "waiting"}
        ).json()
        self.assertEqual(waiting["count"], 1)
        self.assertEqual(waiting["tasks"][0]["id"], self.ids["browser"]["id"])


class ConversationIsolationTests(ParallelCase):
    def test_events_and_reads_stay_inside_their_conversation(self) -> None:
        first = self.submit("task for conversation a", "coding", conversation="conv-a")
        second = self.submit("task for conversation b", "coding", conversation="conv-b")

        events_a = self.runtime.service.events_since("conv-a")
        events_b = self.runtime.service.events_since("conv-b")
        self.assertTrue(events_a)
        self.assertTrue(events_b)
        self.assertEqual({event["task_id"] for event in events_a}, {first["task_id"]})
        self.assertEqual({event["task_id"] for event in events_b}, {second["task_id"]})

        ids = [event["id"] for event in events_a + events_b]
        self.assertEqual(len(ids), len(set(ids)), "event ids must be unique")

        self.assertEqual(
            self.runtime.service.task_events("conv-a", second["task_id"]), []
        )
        self.assertEqual(
            self.runtime.service.task_events("conv-b", first["task_id"]), []
        )
        self.assertEqual(
            {
                row["task_id"]
                for row in self.runtime.registry.events_for_conversation("conv-a")
            },
            {first["task_id"]},
        )

        status = json.loads(
            self.tools.neo_task_status(
                {"task_id": second["task_id"]}, session_id="conv-a"
            )
        )
        self.assertFalse(status["ok"])
        self.assertEqual(status["error"]["code"], "task_not_found")

        listed = json.loads(self.tools.neo_task_list({}, session_id="conv-a"))
        self.assertEqual(
            [row["task_id"] for row in listed["tasks"]], [first["task_id"]]
        )


if __name__ == "__main__":
    unittest.main()
