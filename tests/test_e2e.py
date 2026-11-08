"""Task 24 — automated end-to-end acceptance for the NEO operator flow.

One conversation walks the whole operator contract against the real NEO
components: the SQLite registry, the task service, the process supervisor, the
conversational tools and the browser slice. Executors are controlled: coding
work goes to an argv recorder that starts nothing, and the only process ever
spawned is a short-lived local Python child used to prove real cancellation and
real restart honesty.

Covered here:

* status and list are accurate and non-mutating (no row, event or version
  changes);
* a duplicate request is deduplicated to one task and one launch;
* an active follow-up is queued into the executor's exact session without
  creating a duplicate task;
* a completed follow-up creates exactly one child bound to the same session
  and never substitutes a fresh run;
* cancellation goes through the real supervisor, kills the process group and
  records ``cancelled`` plus a receipt;
* automatic presentation opens a verified artifact once per task version and
  refuses a task whose artifact no longer verifies;
* browser approval records the exact action, approves without publishing
  anything, and is idempotent;
* a restart over the same database records a lost worker as ``failed`` and
  never infers completion.

What is deliberately **not** covered: live Codex/Claude runs, a real Blender
session, a real browser window, the Desktop app and the microphone. Those are
live-only gates recorded in ``docs/demo.md`` and are never claimed here.

No network, no install and no coding CLI is used. Every spawned process group
is killed in cleanup.
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

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from browser import publication_fingerprint  # noqa: E402
from conversation import ConversationTools, NeoRuntime  # noqa: E402
from events import TaskService  # noqa: E402
from neo_tools import WorkerBridge, WorkerTokenStore  # noqa: E402
from presentation import PresentationService, TaskPresenter  # noqa: E402
from registry import Registry  # noqa: E402
from workers import ProcessSupervisor, process_alive  # noqa: E402

CONVERSATION = "neo-acceptance"
OTHER_CONVERSATION = "someone-else"
SLEEP_COMMAND = [sys.executable, "-u", "-c", "import time; time.sleep(30)"]


class LaunchRecorder:
    """Controlled executor: records every argv and starts nothing."""

    def __init__(self) -> None:
        self.calls = []

    def __call__(self, task_id, argv, *, role="task"):
        self.calls.append({"task_id": task_id, "argv": list(argv), "role": role})
        return None

    def last(self) -> dict:
        if not self.calls:
            raise AssertionError("nothing was launched")
        return self.calls[-1]

    def reset_count(self) -> int:
        return len(self.calls)


class OpenerRecorder:
    """Controlled presentation opener: records paths instead of calling `open`."""

    def __init__(self) -> None:
        self.paths = []

    def __call__(self, path):
        self.paths.append(path)
        return None


class AcceptanceTestCase(unittest.TestCase):
    """A real runtime under a temporary directory with controllable executors."""

    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        self.root = Path(self.tempdir.name)
        self.data_dir = self.root / "plugin-data"
        self.db_path = self.data_dir / "neo.db"
        self.launch = LaunchRecorder()
        self.opener = OpenerRecorder()
        self.runtime = NeoRuntime.build(
            data_dir=self.data_dir,
            launch=self.launch,
            max_parallel=4,
        )
        # Real supervisor cleanup runs before the temporary directory goes away.
        self.addCleanup(self.runtime.supervisor.close)
        # This fixture represents a host where browser work is enabled.
        self.runtime.browser_authorized = True
        # Presentation is the same policy production uses, with a recorder opener.
        self.runtime.presenter = TaskPresenter(
            self.runtime.registry,
            PresentationService(self.opener, ledger=self.runtime.registry),
        )
        self.tools = ConversationTools(lambda: self.runtime)
        self.tokens = WorkerTokenStore(self.runtime.registry)
        self.bridge = WorkerBridge(
            self.runtime.service,
            self.tokens,
            preview_root=self.data_dir / "previews",
            preview_max_bytes=65536,
        )

    # -- tool calls ------------------------------------------------------

    def call(self, name: str, args=None, **kwargs):
        payload = getattr(self.tools, name)(args if args is not None else {}, **kwargs)
        self.assertIsInstance(payload, str)
        body = json.loads(payload)
        self.assertIsInstance(body, dict)
        return body

    def submit(self, intent: str, **args):
        payload = {"intent": intent}
        payload.update(args)
        return self.call("neo_submit_task", payload, session_id=CONVERSATION)

    def task_row(self, task_id: str) -> dict:
        row = self.runtime.registry.get_task(task_id)
        self.assertIsNotNone(row)
        return row

    def conversation_tasks(self, key: str = CONVERSATION) -> list:
        return [
            row
            for row in self.runtime.registry.list_tasks()
            if row.get("conversation_key") == key
        ]

    # -- controlled worker -------------------------------------------------

    def start_worker(self, task_id: str) -> str:
        """Issue a real per-task token and move the task to running through the bridge."""
        token = self.tokens.issue(task_id)
        row = self.task_row(task_id)
        self.bridge.neo_task_update(
            task_id,
            token,
            state="running",
            expected_version=row["version"],
            progress=0.4,
            detail="controlled executor is working",
        )
        return token

    def complete_with(self, task_id: str, token: str, name: str) -> dict:
        path = Path(self.task_row(task_id)["workdir"]) / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("acceptance evidence\n", encoding="utf-8")
        return self.bridge.neo_task_complete(
            task_id, token, summary="controlled executor reported completion"
        )

    # -- real processes ----------------------------------------------------

    def spawn_supervised(self, task_id: str) -> dict:
        """Start one real short-lived child through the real supervisor."""
        started = self.runtime.supervisor.submit(task_id, SLEEP_COMMAND)
        self.assertEqual(started["state"], "running", started)
        worker = self.runtime.supervisor.get_worker(task_id)
        self.assertIsNotNone(worker)
        self.addCleanup(self._kill_group, worker["pgid"])
        return worker

    @staticmethod
    def _kill_group(pgid: int) -> None:
        if not isinstance(pgid, int):
            return
        try:
            os.killpg(pgid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass

    def wait_for(self, predicate, timeout: float = 5.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            value = predicate()
            if value:
                return value
            time.sleep(0.02)
        self.fail("condition was not met within %.1fs" % timeout)


class StatusAcceptanceTests(AcceptanceTestCase):
    def test_status_and_list_are_accurate_and_never_mutate(self) -> None:
        submitted = self.submit(
            "implement the acceptance report writer",
            kind="coding",
            expected_artifacts=["report.md"],
        )
        self.assertTrue(submitted["ok"], submitted)
        task_id = submitted["task_id"]
        self.runtime.registry.update_task(
            task_id,
            expected_version=self.task_row(task_id)["version"],
            executor_session_id="thread-acceptance-1",
        )
        self.start_worker(task_id)

        before_row = self.task_row(task_id)
        before_tasks = [row["id"] for row in self.runtime.registry.list_tasks()]
        before_events = len(self.runtime.registry.events_since(0))

        status = self.call("neo_task_status", {"task_id": task_id}, session_id=CONVERSATION)
        status_again = self.call(
            "neo_task_status", {"intent": "acceptance report"}, session_id=CONVERSATION
        )
        listing = self.call("neo_task_list", {}, session_id=CONVERSATION)

        self.assertTrue(status["ok"], status)
        self.assertEqual(status["task_id"], task_id)
        self.assertEqual(status["kind"], "coding")
        self.assertEqual(status["executor"], "codex")
        self.assertEqual(status["intent"], "implement the acceptance report writer")
        self.assertEqual(status["state"], "running")
        self.assertEqual(status["progress"], 0.4)
        self.assertEqual(status["needs"], {"approval": False, "input": False})
        self.assertIsNone(status["error"])
        self.assertEqual(status["expected_artifacts"], ["report.md"])
        self.assertEqual(status["artifacts"], [])
        self.assertEqual(status["preview"], {"available": False, "rev": 0})

        self.assertTrue(status_again["ok"], status_again)
        self.assertEqual(status_again["task_id"], task_id)
        self.assertTrue(listing["ok"], listing)
        self.assertEqual(listing["count"], 1)
        self.assertEqual(listing["total"], 1)
        self.assertEqual(listing["tasks"][0]["task_id"], task_id)
        self.assertEqual(listing["tasks"][0]["state"], "running")
        self.assertEqual(listing["tasks"][0]["needs"], {"approval": False, "input": False})

        after_row = self.task_row(task_id)
        self.assertEqual(after_row, before_row)
        self.assertEqual(after_row["version"], before_row["version"])
        self.assertEqual(
            [row["id"] for row in self.runtime.registry.list_tasks()], before_tasks
        )
        self.assertEqual(len(self.runtime.registry.events_since(0)), before_events)
        self.assertEqual(self.conversation_tasks(OTHER_CONVERSATION), [])


class DuplicatePreventionTests(AcceptanceTestCase):
    def test_a_repeated_request_keeps_one_task_and_one_launch(self) -> None:
        first = self.submit("implement the idempotent acceptance parser", kind="coding")
        second = self.submit(
            "  implement   the idempotent acceptance parser  ", kind="coding"
        )

        self.assertTrue(first["ok"], first)
        self.assertTrue(second["ok"], second)
        self.assertFalse(first["deduplicated"])
        self.assertTrue(second["deduplicated"])
        self.assertEqual(first["task_id"], second["task_id"])
        self.assertEqual(len(self.launch.calls), 1)
        self.assertEqual(self.launch.last()["task_id"], first["task_id"])
        self.assertEqual(len(self.conversation_tasks()), 1)

    def test_the_same_request_in_another_conversation_is_a_new_task(self) -> None:
        first = self.submit("implement the isolated acceptance parser", kind="coding")
        second = self.call(
            "neo_submit_task",
            {"intent": "implement the isolated acceptance parser", "kind": "coding"},
            session_id=OTHER_CONVERSATION,
        )

        self.assertTrue(second["ok"], second)
        self.assertNotEqual(first["task_id"], second["task_id"])
        self.assertFalse(second["deduplicated"])
        self.assertEqual(len(self.conversation_tasks()), 1)
        self.assertEqual(len(self.conversation_tasks(OTHER_CONVERSATION)), 1)


class ExactSessionFollowUpTests(AcceptanceTestCase):
    def test_active_follow_up_queues_into_the_exact_session_without_a_new_task(self) -> None:
        submitted = self.submit(
            "implement the follow-up acceptance parser", kind="coding"
        )
        task_id = submitted["task_id"]
        self.runtime.registry.update_task(
            task_id,
            expected_version=self.task_row(task_id)["version"],
            executor_session_id="thread-acceptance-active",
        )
        self.start_worker(task_id)
        launches_before = self.launch.reset_count()
        tasks_before = len(self.conversation_tasks())

        body = self.call(
            "neo_task_refine",
            {"task_id": task_id, "follow_up": "tighten the error handling"},
            session_id=CONVERSATION,
        )

        self.assertTrue(body["ok"], body)
        self.assertTrue(body["queued"])
        self.assertEqual(body["state"], "queued")
        self.assertEqual(body["session_id"], "thread-acceptance-active")
        self.assertEqual(body["task_state"], "running")
        self.assertEqual(len(self.conversation_tasks()), tasks_before)
        self.assertEqual(self.task_row(task_id)["state"], "running")

        self.assertEqual(len(self.launch.calls), launches_before + 1)
        record = self.launch.last()
        self.assertEqual(record["task_id"], task_id)
        self.assertEqual(record["role"], "follow_up")
        self.assertEqual(
            record["argv"],
            [
                "codex",
                "queue",
                "--thread",
                "thread-acceptance-active",
                "--message",
                "tighten the error handling",
            ],
        )
        self.assertNotIn("--last", record["argv"])

    def test_completed_follow_up_creates_one_child_on_the_exact_session(self) -> None:
        submitted = self.submit(
            "implement the resumable acceptance parser",
            kind="coding",
            expected_artifacts=["report.md"],
        )
        parent_id = submitted["task_id"]
        self.runtime.registry.update_task(
            parent_id,
            expected_version=self.task_row(parent_id)["version"],
            executor_session_id="thread-acceptance-resume",
        )
        token = self.start_worker(parent_id)
        completed = self.complete_with(parent_id, token, "report.md")
        self.assertEqual(completed["state"], "completed")

        body = self.call(
            "neo_task_refine",
            {"task_id": parent_id, "follow_up": "add a summary section"},
            session_id=CONVERSATION,
        )

        self.assertTrue(body["ok"], body)
        self.assertTrue(body["resumed"])
        self.assertEqual(body["executor"], "codex")
        self.assertEqual(body["session_id"], "thread-acceptance-resume")
        self.assertNotEqual(body["task_id"], parent_id)

        child = self.task_row(body["task_id"])
        self.assertEqual(child["parent_id"], parent_id)
        self.assertEqual(child["executor_session_id"], "thread-acceptance-resume")
        self.assertEqual(child["state"], "starting")
        self.assertEqual(len(self.conversation_tasks()), 2)
        self.assertEqual(
            sum(1 for row in self.conversation_tasks() if row["parent_id"] == parent_id),
            1,
        )

        record = self.launch.last()
        self.assertEqual(record["task_id"], child["id"])
        self.assertEqual(record["role"], "task")
        self.assertIn("resume", record["argv"])
        self.assertEqual(
            record["argv"][record["argv"].index("resume") + 1],
            "thread-acceptance-resume",
        )
        self.assertNotIn("--last", record["argv"])


class CancellationTests(AcceptanceTestCase):
    def test_cancel_kills_the_supervised_group_and_records_the_decision(self) -> None:
        task = self.runtime.service.create_task(
            conversation_key=CONVERSATION,
            kind="general",
            intent="run the cancellable acceptance probe",
        )
        worker = self.spawn_supervised(task["id"])
        pid = worker["pid"]

        body = self.call(
            "neo_task_cancel", {"task_id": task["id"]}, session_id=CONVERSATION
        )

        self.assertTrue(body["ok"], body)
        self.assertTrue(body["cancelled"], body)
        self.assertEqual(body["state"], "cancelled")
        row = self.task_row(task["id"])
        self.assertEqual(row["state"], "cancelled")
        worker = self.runtime.supervisor.get_worker(task["id"])
        self.assertEqual(worker["state"], "cancelled")
        self.wait_for(lambda: not process_alive(pid))
        receipt_path = Path(worker["log_path"]).with_suffix(".receipt.json")
        self.wait_for(receipt_path.exists)
        receipt = json.loads(receipt_path.read_text())
        self.assertTrue(receipt["term_sent"] or receipt["kill_sent"], receipt)
        self.assertNotEqual(receipt["exit_code"], 0)

        again = self.call(
            "neo_task_cancel", {"task_id": task["id"]}, session_id=CONVERSATION
        )
        self.assertTrue(again["ok"], again)
        self.assertTrue(again["already_terminal"])
        self.assertEqual(again["version"], body["version"])
        self.assertEqual(len(self.launch.calls), 0)


class PresentationTests(AcceptanceTestCase):
    def completed_task(self, intent: str, name: str):
        submitted = self.submit(intent, kind="coding", expected_artifacts=[name])
        task_id = submitted["task_id"]
        self.runtime.registry.update_task(
            task_id,
            expected_version=self.task_row(task_id)["version"],
            executor_session_id="thread-acceptance-present",
        )
        token = self.start_worker(task_id)
        return task_id, self.complete_with(task_id, token, name)

    def test_verified_artifact_is_presented_exactly_once(self) -> None:
        task_id, completed = self.completed_task(
            "implement the presentable acceptance report", "report.md"
        )
        self.assertEqual(completed["state"], "completed")
        self.assertEqual(json.loads(completed["artifacts_json"]), ["report.md"])

        first = self.call(
            "neo_task_open", {"task_id": task_id}, session_id=CONVERSATION
        )
        second = self.call(
            "neo_task_open", {"task_id": task_id}, session_id=CONVERSATION
        )

        self.assertTrue(first["ok"], first)
        self.assertTrue(first["presented"])
        self.assertFalse(first["skipped"])
        self.assertEqual(first["reason"], "presented")
        self.assertEqual(len(self.opener.paths), 1)
        opened = Path(self.opener.paths[0])
        self.assertTrue(opened.is_absolute())
        self.assertEqual(opened.name, "report.md")
        self.assertTrue(opened.is_file())
        self.assertNotIn("path", first)

        self.assertFalse(second["ok"], second)
        self.assertFalse(second["presented"])
        self.assertTrue(second["skipped"])
        self.assertIn("already presented", second["reason"])
        self.assertEqual(len(self.opener.paths), 1)
        self.assertEqual(self.task_row(task_id)["state"], "completed")

    def test_an_artifact_that_no_longer_verifies_is_not_opened(self) -> None:
        task_id, completed = self.completed_task(
            "implement the vanishing acceptance report", "vanishing.md"
        )
        self.assertEqual(completed["state"], "completed")
        Path(self.task_row(task_id)["workdir"], "vanishing.md").unlink()

        body = self.call("neo_task_open", {"task_id": task_id}, session_id=CONVERSATION)

        self.assertFalse(body["ok"], body)
        self.assertFalse(body["presented"])
        self.assertTrue(body["skipped"])
        self.assertIn("artifact verification failed", body["reason"])
        self.assertEqual(self.opener.paths, [])


class BrowserApprovalTests(AcceptanceTestCase):
    def test_approval_records_the_exact_action_and_publishes_nothing(self) -> None:
        submitted = self.submit(
            "navigate to https://example.com and report the page title",
            kind="browser",
        )
        self.assertTrue(submitted["ok"], submitted)
        self.assertEqual(submitted["executor"], "hermes_worker")
        argv = self.launch.last()["argv"]
        self.assertEqual(argv[0], "hermes")
        self.assertEqual(argv[argv.index("--skills") + 1], "browser,neo-worker")
        task_id = submitted["task_id"]

        task = self.runtime.service.transition(
            task_id, "running", expected_version=self.task_row(task_id)["version"]
        )
        launches_before = len(self.launch.calls)

        pending = self.runtime.browser.request_approval(
            task,
            reason="the next step would publish the draft",
            action="publish the draft post to the blog",
            account="demo-account",
            audience="public",
            conversation_key=CONVERSATION,
        )
        self.assertEqual(pending["state"], "needs_approval")

        stored = self.task_row(task_id)
        approval = json.loads(stored["approval_json"])
        self.assertIsNone(approval["decision"])
        self.assertTrue(approval["publication"])
        self.assertFalse(approval["publication_executed"])
        self.assertEqual(approval["action"], "publish the draft post to the blog")
        self.assertEqual(approval["account"], "demo-account")
        self.assertEqual(approval["audience"], "public")
        self.assertEqual(
            approval["fingerprint"],
            publication_fingerprint(
                "publish the draft post to the blog", "demo-account", "public"
            ),
        )
        self.assertEqual(approval["task_version"], stored["version"])

        status = self.call(
            "neo_task_status", {"task_id": task_id}, session_id=CONVERSATION
        )
        self.assertEqual(status["state"], "needs_approval")
        self.assertEqual(status["needs"], {"approval": True, "input": False})

        decision = self.runtime.browser.record_approval(
            task_id, "approve", reason="approved for the offline acceptance demo"
        )
        self.assertEqual(decision["decision"], "approve")
        self.assertFalse(decision["idempotent"])
        self.assertFalse(decision["publication_executed"])

        stored = self.task_row(task_id)
        approval = json.loads(stored["approval_json"])
        self.assertEqual(stored["state"], "running")
        self.assertEqual(approval["decision"], "approve")
        self.assertFalse(approval["publication_executed"])
        self.assertEqual(approval["outcome"], "publication_not_executed")
        # Approving is a decision, never a launch or a publication.
        self.assertEqual(len(self.launch.calls), launches_before)
        self.assertEqual(len(self.conversation_tasks()), 1)

        repeat = self.runtime.browser.record_approval(task_id, "approve")
        self.assertTrue(repeat["idempotent"])
        self.assertFalse(repeat["publication_executed"])
        self.assertEqual(len(self.launch.calls), launches_before)


class RestartHonestyTests(AcceptanceTestCase):
    def test_restart_fails_a_lost_worker_and_never_infers_completion(self) -> None:
        task = self.runtime.service.create_task(
            conversation_key=CONVERSATION,
            kind="general",
            intent="run the recoverable acceptance probe",
        )
        worker = self.spawn_supervised(task["id"])
        self.assertEqual(worker["state"], "running")

        # A fresh graph over the same database, as the next process would build.
        registry = Registry(path=self.db_path, max_parallel=4)
        service = TaskService(registry)
        supervisor = ProcessSupervisor(
            service,
            WorkerTokenStore(registry),
            log_dir=self.data_dir / "logs",
            alive_fn=lambda pid: False,
            poll_interval=0.02,
        )
        self.addCleanup(supervisor.close)

        summary = supervisor.reconcile()

        self.assertEqual(summary["scanned"], 1)
        self.assertEqual(summary["failed"], [task["id"]])
        self.assertEqual(summary["adopted"], [])
        recovered = registry.get_task(task["id"])
        self.assertEqual(recovered["state"], "failed")
        self.assertEqual(recovered["error_code"], "worker_lost")
        self.assertIn("never reported completion", recovered["error_detail"])
        self.assertNotEqual(recovered["state"], "completed")

        # The database, not the process table, is authoritative across the restart.
        self.assertEqual(self.task_row(task["id"])["state"], "failed")


if __name__ == "__main__":
    unittest.main()
