"""Task 18 — conversational tools and the NEO operator middleware.

These tests fix the conversational contract: submit, status, list, refine,
frame follow-up, cancel and open all answer in JSON and never raise; a status
or list question creates nothing; a follow-up goes to the exact native session
or is refused honestly; and the ``pre_llm_call`` hook adds a compact, bounded
summary of this conversation's active tasks without touching anything else.

No test spawns a process or touches the network. Every runtime is built inside a
temporary directory, and ``launch`` is a recorder.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from conversation import (  # noqa: E402
    ConversationTools,
    NeoRuntime,
    conversation_key_from,
    make_pre_llm_call,
    register_conversation_tools,
)
from presentation import PresentationService, TaskPresenter  # noqa: E402
from state_machine import CompletionEvidence  # noqa: E402

TOOL_NAMES = (
    "neo_submit_task",
    "neo_task_status",
    "neo_task_list",
    "neo_task_refine",
    "neo_task_cancel",
    "neo_task_frame_follow_up",
    "neo_task_open",
)
TERMINAL = ("completed", "failed", "cancelled")
ACTIVE = (
    "queued",
    "starting",
    "running",
    "waiting",
    "needs_approval",
    "needs_input",
    "cancelling",
    "unknown",
)


class LaunchRecorder:
    """Stand-in for the process supervisor: records argv, spawns nothing."""

    def __init__(self) -> None:
        self.calls = []

    def __call__(self, task_id, argv, *, role="task"):
        self.calls.append({"task_id": task_id, "argv": list(argv), "role": role})
        return None

    def last(self) -> dict:
        if not self.calls:
            raise AssertionError("nothing was launched")
        return self.calls[-1]


class FakeSupervisor:
    """Cancel through the real state machine, without any process."""

    def __init__(self) -> None:
        self.calls = []
        self.service = None

    def cancel(self, task_id, **_kwargs):
        self.calls.append(task_id)
        task = self.service.get_task(task_id)
        if task is None:
            raise KeyError(task_id)
        if task["state"] in TERMINAL:
            return task
        if task["state"] == "queued":
            return self.service.transition(
                task_id, "cancelled", expected_version=task["version"]
            )
        if task["state"] == "starting":
            task = self.service.transition(
                task_id, "running", expected_version=task["version"]
            )
        if task["state"] == "running":
            task = self.service.transition(
                task_id, "cancelling", expected_version=task["version"]
            )
        if task["state"] in ("cancelling", "waiting", "needs_approval", "needs_input", "unknown"):
            task = self.service.transition(
                task_id, "cancelled", expected_version=task["version"]
            )
        return task


class OpenerRecorder:
    def __init__(self) -> None:
        self.paths = []

    def __call__(self, path):
        self.paths.append(path)
        return None


class ConversationTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.data_dir = self.root / "plugin-data"
        self.launch = LaunchRecorder()
        self.supervisor = FakeSupervisor()
        self.runtime = NeoRuntime.build(
            data_dir=self.data_dir,
            launch=self.launch,
            supervisor=self.supervisor,
            max_parallel=16,
        )
        self.supervisor.service = self.runtime.service
        # Coding/general tests are independent of host capability discovery.
        # Browser routing tests opt in explicitly; security tests exercise the
        # fail-closed default separately.
        self.runtime.browser_authorized = True
        self.tools = ConversationTools(lambda: self.runtime)

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    # -- helpers ---------------------------------------------------------

    def call(self, name: str, args=None, **kwargs):
        payload = getattr(self.tools, name)(args if args is not None else {}, **kwargs)
        self.assertIsInstance(payload, str)
        body = json.loads(payload)
        self.assertIsInstance(body, dict)
        return body

    def create(
        self,
        *,
        session: str = "session-1",
        kind: str = "coding",
        intent: str = "do the work",
        state: str = "starting",
        executor: str = "codex",
        session_id=None,
        workdir=None,
    ) -> dict:
        task = self.runtime.service.create_task(
            conversation_key=session,
            kind=kind,
            intent=intent,
            workdir=workdir or self.runtime.workspace("seed"),
        )
        changes = {}
        if executor is not None:
            changes["executor"] = executor
        if session_id is not None:
            changes["executor_session_id"] = session_id
        if changes:
            task = self.runtime.registry.update_task(
                task["id"], expected_version=task["version"], **changes
            )
        if state != task["state"]:
            task = self.advance(task, state)
        return task

    def advance(self, task: dict, target: str, *, evidence=None) -> dict:
        if task["state"] == "queued":
            task = self.runtime.service.transition(
                task["id"], "starting", expected_version=task["version"]
            )
        if task["state"] == "starting" and target != "queued":
            task = self.runtime.service.transition(
                task["id"], "running", expected_version=task["version"]
            )
        if task["state"] == target:
            return task
        if target == "completed" and evidence is None:
            evidence = CompletionEvidence(
                executor_confirmed=True,
                expected_artifacts=(),
                verified_artifacts=(),
            )
        return self.runtime.service.transition(
            task["id"], target, expected_version=task["version"], evidence=evidence
        )

    def complete_with_artifact(self, name: str, *, session: str = "session-1") -> dict:
        workdir = self.root / "work"
        workdir.mkdir(exist_ok=True)
        (workdir / name).write_text("result body", encoding="utf-8")
        task = self.runtime.service.create_task(
            conversation_key=session,
            kind="coding",
            intent="produce the report",
            workdir=str(workdir),
        )
        task = self.runtime.registry.update_task(
            task["id"],
            expected_version=task["version"],
            executor="codex",
            executor_session_id="thread-open",
        )
        task = self.runtime.service.transition(
            task["id"], "running", expected_version=task["version"]
        )
        evidence = CompletionEvidence(
            executor_confirmed=True,
            expected_artifacts=(name,),
            verified_artifacts=(name,),
        )
        return self.runtime.service.transition(
            task["id"], "completed", expected_version=task["version"], evidence=evidence
        )

    def attach_presenter(self) -> OpenerRecorder:
        opener = OpenerRecorder()
        self.runtime.presenter = TaskPresenter(
            self.runtime.registry,
            PresentationService(opener, ledger=self.runtime.registry),
        )
        return opener

    def conversations_tasks(self, session: str) -> list:
        return [
            row
            for row in self.runtime.registry.list_tasks()
            if row.get("conversation_key") == session
        ]


class SubmitTaskTests(ConversationTestCase):
    def test_coding_submission_launches_codex_in_plugin_data_workspace(self) -> None:
        body = self.call(
            "neo_submit_task",
            {"intent": "implement a parser", "kind": "coding"},
            session_id="session-1",
        )

        self.assertTrue(body["ok"], body)
        self.assertEqual(body["executor"], "codex")
        self.assertEqual(body["kind"], "coding")
        self.assertIn(body["state"], ("starting", "queued"))

        rec = self.launch.last()
        self.assertEqual(rec["role"], "task")
        self.assertEqual(rec["argv"][0], "codex")
        self.assertEqual(rec["argv"][-2:], ["--", "implement a parser"])
        cd = rec["argv"].index("--cd")
        workdir = rec["argv"][cd + 1]
        self.assertTrue(Path(workdir).is_dir())
        self.assertTrue(workdir.startswith(str(self.runtime.data_dir)), workdir)

        task = self.runtime.registry.get_task(body["task_id"])
        self.assertEqual(task["workdir"], workdir)
        self.assertEqual(task["executor"], "codex")
        self.assertEqual(task["conversation_key"], "session-1")

    def test_browser_submission_uses_a_hermes_worker_prompt_file(self) -> None:
        body = self.call(
            "neo_submit_task",
            {"intent": "open the local docs page in a browser", "kind": "browser"},
            session_id="session-1",
        )

        self.assertTrue(body["ok"], body)
        self.assertEqual(body["executor"], "hermes_worker")
        rec = self.launch.last()
        argv = rec["argv"]
        self.assertEqual(argv[0], "hermes")
        self.assertIn("--query-file", argv)
        prompt_path = Path(argv[argv.index("--query-file") + 1])
        self.assertTrue(prompt_path.is_file())
        self.assertIn("open the local docs page", prompt_path.read_text(encoding="utf-8"))
        self.assertEqual(argv[argv.index("--skills") + 1], "browser,neo-worker")

    def test_blender_submission_waits_for_the_pinned_plugin_approval(self) -> None:
        body = self.call(
            "neo_submit_task",
            {"intent": "make a low-poly rocket in Blender", "kind": "blender"},
            session_id="session-1",
        )

        self.assertFalse(body["ok"], body)
        self.assertEqual(body["state"], "needs_approval")
        self.assertEqual(body["capability"], "blender")
        self.assertEqual(self.launch.calls, [])
        tasks = [
            row for row in self.runtime.registry.list_tasks() if row["kind"] == "blender"
        ]
        self.assertEqual(tasks, [])

    def test_auto_kind_infers_coding_from_the_intent(self) -> None:
        body = self.call(
            "neo_submit_task",
            {"intent": "refactor the parser tests"},
            session_id="session-1",
        )

        self.assertTrue(body["ok"], body)
        self.assertEqual(body["kind"], "coding")
        self.assertEqual(body["executor"], "codex")

    def test_auto_kind_falls_back_to_general_worker(self) -> None:
        body = self.call(
            "neo_submit_task",
            {"intent": "summarise the meeting notes on my desk"},
            session_id="session-1",
        )

        self.assertTrue(body["ok"], body)
        self.assertEqual(body["kind"], "general")
        self.assertEqual(body["executor"], "hermes_worker")

    def test_unknown_kind_is_normalized_not_trusted(self) -> None:
        body = self.call(
            "neo_submit_task",
            {"intent": "write a helper function", "kind": "shell-magic"},
            session_id="session-1",
        )

        self.assertTrue(body["ok"], body)
        self.assertIn(body["kind"], ("coding", "blender", "browser", "general"))

    def test_unsafe_workdir_falls_back_to_the_plugin_data_workspace(self) -> None:
        body = self.call(
            "neo_submit_task",
            {"intent": "implement a parser", "workdir": "relative/path"},
            session_id="session-1",
        )

        self.assertTrue(body["ok"], body)
        task = self.runtime.registry.get_task(body["task_id"])
        self.assertTrue(task["workdir"].startswith(str(self.runtime.data_dir)))
        self.assertTrue(Path(task["workdir"]).is_dir())

    def test_missing_workdir_defaults_to_task_workspace(self) -> None:
        body = self.call(
            "neo_submit_task", {"intent": "implement a parser"}, session_id="session-1"
        )

        task = self.runtime.registry.get_task(body["task_id"])
        self.assertIn(str(Path("plugin-data") / "tasks"), task["workdir"])

    def test_duplicate_equivalent_submission_is_deduplicated(self) -> None:
        first = self.call(
            "neo_submit_task", {"intent": "implement a parser"}, session_id="session-1"
        )
        second = self.call(
            "neo_submit_task", {"intent": "  implement   a parser "}, session_id="session-1"
        )

        self.assertEqual(first["task_id"], second["task_id"])
        self.assertTrue(second["deduplicated"])
        self.assertEqual(len(self.conversations_tasks("session-1")), 1)
        self.assertEqual(len(self.launch.calls), 1)

    def test_the_same_intent_in_another_conversation_is_a_new_task(self) -> None:
        first = self.call(
            "neo_submit_task", {"intent": "implement a parser"}, session_id="session-1"
        )
        second = self.call(
            "neo_submit_task", {"intent": "implement a parser"}, session_id="session-2"
        )

        self.assertNotEqual(first["task_id"], second["task_id"])
        self.assertFalse(second["deduplicated"])

    def test_expected_artifacts_are_persisted_for_completion_checks(self) -> None:
        body = self.call(
            "neo_submit_task",
            {"intent": "write report.md", "expected_artifacts": ["report.md"]},
            session_id="session-1",
        )

        task = self.runtime.registry.get_task(body["task_id"])
        self.assertEqual(json.loads(task["expected_json"]), ["report.md"])

    def test_recursion_guard_refuses_submit_from_a_worker(self) -> None:
        with mock.patch.dict(os.environ, {"NEO_WORKER": "1"}):
            body = self.call(
                "neo_submit_task", {"intent": "implement a parser"}, session_id="session-1"
            )

        self.assertFalse(body["ok"])
        self.assertEqual(body["error"]["code"], "recursion_refused")
        self.assertEqual(self.runtime.registry.list_tasks(), [])
        self.assertEqual(self.launch.calls, [])

    def test_missing_intent_is_refused_without_creating_anything(self) -> None:
        body = self.call("neo_submit_task", {}, session_id="session-1")

        self.assertFalse(body["ok"])
        self.assertEqual(self.runtime.registry.list_tasks(), [])

    def test_missing_conversation_context_is_refused(self) -> None:
        body = self.call("neo_submit_task", {"intent": "implement a parser"})

        self.assertFalse(body["ok"])
        self.assertEqual(body["error"]["code"], "no_conversation")
        self.assertEqual(self.runtime.registry.list_tasks(), [])


class StatusAndListTests(ConversationTestCase):
    def test_status_by_task_id_creates_no_task(self) -> None:
        task = self.create(session="session-1")
        before = len(self.runtime.registry.list_tasks())

        body = self.call(
            "neo_task_status", {"task_id": task["id"]}, session_id="session-1"
        )

        self.assertTrue(body["ok"], body)
        self.assertEqual(body["task_id"], task["id"])
        self.assertEqual(len(self.runtime.registry.list_tasks()), before)

    def test_status_with_no_arguments_creates_no_task(self) -> None:
        before = len(self.runtime.registry.list_tasks())

        body = self.call("neo_task_status", {}, session_id="session-1")

        self.assertFalse(body["ok"])
        self.assertEqual(len(self.runtime.registry.list_tasks()), before)

    def test_status_reports_state_version_progress_and_no_secrets(self) -> None:
        task = self.create(session="session-1", state="running")
        self.runtime.service.transition(
            task["id"],
            "waiting",
            expected_version=task["version"],
            event_extra={"progress": 0.25, "detail": "waiting on build"},
        )

        body = self.call(
            "neo_task_status", {"task_id": task["id"]}, session_id="session-1"
        )

        self.assertEqual(body["state"], "waiting")
        self.assertGreaterEqual(body["version"], 3)
        self.assertEqual(body["progress"], 0.25)
        self.assertEqual(body["intent"], "do the work")
        self.assertEqual(body["preview"], {"available": False, "rev": 0})
        self.assertEqual(body["artifacts"], [])
        self.assertIsNone(body["error"])
        rendered = json.dumps(body)
        self.assertNotIn("conversation_key", rendered)
        self.assertNotIn("workdir", rendered)
        self.assertNotIn("token", rendered.lower())

    def test_status_resolves_an_unambiguous_partial_intent(self) -> None:
        rocket = self.create(session="session-1", kind="blender", intent="render the rocket")
        self.create(session="session-1", kind="coding", intent="fix the API bug")

        body = self.call("neo_task_status", {"intent": "rocket"}, session_id="session-1")

        self.assertTrue(body["ok"], body)
        self.assertEqual(body["task_id"], rocket["id"])

    def test_status_ambiguous_reference_asks_exactly_one_question(self) -> None:
        first = self.create(session="session-1", intent="fix the parser tests")
        second = self.create(session="session-1", intent="fix the parser docs")

        body = self.call("neo_task_status", {"intent": "parser"}, session_id="session-1")

        self.assertFalse(body["ok"])
        self.assertEqual(body["status"], "ambiguous")
        self.assertEqual(body["question"].count("?"), 1)
        candidate_ids = {candidate["task_id"] for candidate in body["candidates"]}
        self.assertEqual(candidate_ids, {first["id"], second["id"]})

    def test_status_unknown_task_is_not_found_without_creating_a_task(self) -> None:
        self.create(session="session-1")
        before = len(self.runtime.registry.list_tasks())

        body = self.call(
            "neo_task_status", {"task_id": "0" * 32}, session_id="session-1"
        )

        self.assertFalse(body["ok"])
        self.assertEqual(body["error"]["code"], "task_not_found")
        self.assertEqual(len(self.runtime.registry.list_tasks()), before)

    def test_status_never_reads_another_conversation(self) -> None:
        other = self.create(session="session-2")

        body = self.call(
            "neo_task_status", {"task_id": other["id"]}, session_id="session-1"
        )

        self.assertFalse(body["ok"])
        self.assertEqual(body["error"]["code"], "task_not_found")

    def test_list_is_conversation_scoped_and_filters(self) -> None:
        keep = self.create(session="session-1", kind="coding", intent="fix the parser")
        self.create(session="session-1", kind="blender", intent="render the rocket")
        self.create(session="session-2", kind="coding", intent="someone else's work")

        body = self.call("neo_task_list", {}, session_id="session-1")
        self.assertTrue(body["ok"], body)
        self.assertEqual(body["count"], 2)

        coding = self.call("neo_task_list", {"kind": "coding"}, session_id="session-1")
        self.assertEqual([task["task_id"] for task in coding["tasks"]], [keep["id"]])

        missing = self.call(
            "neo_task_list", {"state": "needs_approval"}, session_id="session-1"
        )
        self.assertEqual(missing["count"], 0)

        limited = self.call("neo_task_list", {"limit": 1}, session_id="session-1")
        self.assertEqual(len(limited["tasks"]), 1)

    def test_list_creates_no_task(self) -> None:
        before = len(self.runtime.registry.list_tasks())

        body = self.call("neo_task_list", {}, session_id="session-1")

        self.assertTrue(body["ok"])
        self.assertEqual(body["count"], 0)
        self.assertEqual(len(self.runtime.registry.list_tasks()), before)


class RefineTests(ConversationTestCase):
    def test_completed_codex_task_resumes_the_exact_thread(self) -> None:
        parent = self.create(
            session="session-1",
            executor="codex",
            session_id="thread-abc",
            state="completed",
        )

        body = self.call(
            "neo_task_refine",
            {"task_id": parent["id"], "follow_up": "make it darker"},
            session_id="session-1",
        )

        self.assertTrue(body["ok"], body)
        self.assertNotEqual(body["task_id"], parent["id"])
        self.assertEqual(body["executor"], "codex")
        self.assertTrue(body["resumed"])

        child = self.runtime.registry.get_task(body["task_id"])
        self.assertEqual(child["parent_id"], parent["id"])
        self.assertEqual(child["executor_session_id"], "thread-abc")

        rec = self.launch.last()
        self.assertEqual(rec["task_id"], child["id"])
        self.assertEqual(rec["role"], "task")
        self.assertEqual(
            rec["argv"],
            [
                "codex",
                "exec",
                "--json",
                "--sandbox",
                "workspace-write",
                "--skip-git-repo-check",
                "--cd",
                parent["workdir"],
                "resume",
                "thread-abc",
                "make it darker",
            ],
        )
        self.assertNotIn("--last", rec["argv"])

    def test_active_codex_task_queues_into_the_same_thread(self) -> None:
        parent = self.create(
            session="session-1",
            executor="codex",
            session_id="thread-live",
            state="running",
        )

        body = self.call(
            "neo_task_refine",
            {"task_id": parent["id"], "follow_up": "make it darker"},
            session_id="session-1",
        )

        self.assertTrue(body["ok"], body)
        self.assertTrue(body["queued"])
        self.assertEqual(body["state"], "queued")
        self.assertEqual(body["task_state"], "running")

        rec = self.launch.last()
        self.assertEqual(rec["role"], "follow_up")
        self.assertEqual(
            rec["argv"],
            ["codex", "queue", "--thread", "thread-live", "--message", "make it darker"],
        )
        # A queue is a follow-up, not a new task, and it does not move the task.
        self.assertEqual(len(self.runtime.registry.list_tasks()), 1)
        self.assertEqual(self.runtime.service.get_task(parent["id"])["state"], "running")

    def test_completed_claude_task_resumes_the_exact_uuid(self) -> None:
        session = str(uuid.uuid4())
        parent = self.create(
            session="session-1",
            executor="claude",
            session_id=session,
            state="completed",
        )

        body = self.call(
            "neo_task_refine",
            {"task_id": parent["id"], "follow_up": "make it darker"},
            session_id="session-1",
        )

        self.assertTrue(body["ok"], body)
        self.assertEqual(body["executor"], "claude")
        rec = self.launch.last()
        self.assertEqual(
            rec["argv"],
            [
                "claude",
                "-p",
                "--resume",
                session,
                "--output-format",
                "stream-json",
                "--verbose",
                "--",
                "make it darker",
            ],
        )
        self.assertNotIn("--fork-session", rec["argv"])

    def test_active_claude_task_is_refused_rather_than_forked(self) -> None:
        session = str(uuid.uuid4())
        parent = self.create(
            session="session-1",
            executor="claude",
            session_id=session,
            state="running",
        )

        body = self.call(
            "neo_task_refine",
            {"task_id": parent["id"], "follow_up": "make it darker"},
            session_id="session-1",
        )

        self.assertFalse(body["ok"])
        self.assertEqual(body["error"]["code"], "active_session_refused")
        self.assertEqual(len(self.runtime.registry.list_tasks()), 1)
        self.assertEqual(self.launch.calls, [])

    def test_completed_hermes_worker_resumes_the_exact_result_session(self) -> None:
        parent = self.create(
            session="session-1",
            kind="browser",
            executor="hermes_worker",
            session_id="worker-session-7",
            state="completed",
        )

        body = self.call(
            "neo_task_refine",
            {"task_id": parent["id"], "follow_up": "go one page further"},
            session_id="session-1",
        )

        self.assertTrue(body["ok"], body)
        self.assertEqual(body["executor"], "hermes_worker")
        argv = self.launch.last()["argv"]
        self.assertEqual(argv[0], "hermes")
        self.assertEqual(argv[argv.index("--resume") + 1], "worker-session-7")
        prompt = Path(argv[argv.index("--query-file") + 1]).read_text(encoding="utf-8")
        self.assertIn("go one page further", prompt)

    def test_missing_session_id_refuses_instead_of_starting_a_fresh_run(self) -> None:
        parent = self.create(
            session="session-1", executor="codex", session_id=None, state="completed"
        )

        body = self.call(
            "neo_task_refine",
            {"task_id": parent["id"], "follow_up": "make it darker"},
            session_id="session-1",
        )

        self.assertFalse(body["ok"])
        self.assertEqual(body["error"]["code"], "missing_session_id")
        self.assertEqual(len(self.runtime.registry.list_tasks()), 1)
        self.assertEqual(self.launch.calls, [])

    def test_refine_refuses_a_task_from_another_conversation(self) -> None:
        parent = self.create(
            session="session-2",
            executor="codex",
            session_id="thread-other",
            state="completed",
        )

        body = self.call(
            "neo_task_refine",
            {"task_id": parent["id"], "follow_up": "make it darker"},
            session_id="session-1",
        )

        self.assertFalse(body["ok"])
        self.assertEqual(body["error"]["code"], "task_not_found")
        self.assertEqual(self.launch.calls, [])

    def test_refine_recursion_guard_blocks_a_worker(self) -> None:
        parent = self.create(
            session="session-1",
            executor="codex",
            session_id="thread-abc",
            state="completed",
        )
        with mock.patch.dict(os.environ, {"NEO_WORKER": "1"}):
            body = self.call(
                "neo_task_refine",
                {"task_id": parent["id"], "follow_up": "make it darker"},
                session_id="session-1",
            )

        self.assertFalse(body["ok"])
        self.assertEqual(body["error"]["code"], "recursion_refused")
        self.assertEqual(self.launch.calls, [])


class FrameFollowUpTests(ConversationTestCase):
    def test_blender_follow_up_resumes_the_recorded_session_with_prior_context(self) -> None:
        parent = self.create(
            session="session-1",
            kind="blender",
            intent="model a low-poly rocket",
            executor="hermes_worker",
            session_id="blender-session-1",
            state="completed",
        )

        body = self.call(
            "neo_task_frame_follow_up",
            {"task_id": parent["id"], "follow_up": "add a red nose cone"},
            session_id="session-1",
        )

        self.assertTrue(body["ok"], body)
        self.assertEqual(body["kind"], "blender")
        self.assertEqual(body["parent_id"], parent["id"])
        self.assertTrue(body["resumed"])
        self.assertEqual(body["prior_session_id"], "blender-session-1")

        argv = self.launch.last()["argv"]
        self.assertEqual(argv[argv.index("--resume") + 1], "blender-session-1")
        prompt = Path(argv[argv.index("--query-file") + 1]).read_text(encoding="utf-8")
        self.assertIn("model a low-poly rocket", prompt)
        self.assertIn("add a red nose cone", prompt)

        child = self.runtime.registry.get_task(body["task_id"])
        self.assertEqual(child["kind"], "blender")
        self.assertEqual(child["parent_id"], parent["id"])

    def test_blender_follow_up_without_a_session_reports_honestly(self) -> None:
        parent = self.create(
            session="session-1",
            kind="blender",
            intent="model a low-poly rocket",
            executor="hermes_worker",
            session_id=None,
            state="failed",
        )

        body = self.call(
            "neo_task_frame_follow_up",
            {"task_id": parent["id"], "follow_up": "add a red nose cone"},
            session_id="session-1",
        )

        self.assertTrue(body["ok"], body)
        self.assertFalse(body["resumed"])
        self.assertIsNone(body["prior_session_id"])
        self.assertIn("no prior", body["note"].lower())

        argv = self.launch.last()["argv"]
        self.assertNotIn("--resume", argv)
        prompt = Path(argv[argv.index("--query-file") + 1]).read_text(encoding="utf-8")
        self.assertIn("model a low-poly rocket", prompt)
        self.assertIn("add a red nose cone", prompt)

    def test_frame_follow_up_refuses_a_non_blender_task(self) -> None:
        parent = self.create(session="session-1", kind="coding", state="completed")

        body = self.call(
            "neo_task_frame_follow_up",
            {"task_id": parent["id"], "follow_up": "make it darker"},
            session_id="session-1",
        )

        self.assertFalse(body["ok"])
        self.assertEqual(body["error"]["code"], "not_blender")
        self.assertEqual(self.launch.calls, [])

    def test_frame_follow_up_recursion_guard_blocks_a_worker(self) -> None:
        parent = self.create(
            session="session-1",
            kind="blender",
            executor="hermes_worker",
            session_id="blender-session-1",
            state="completed",
        )
        with mock.patch.dict(os.environ, {"NEO_WORKER": "1"}):
            body = self.call(
                "neo_task_frame_follow_up",
                {"task_id": parent["id"], "follow_up": "add a red nose cone"},
                session_id="session-1",
            )

        self.assertFalse(body["ok"])
        self.assertEqual(body["error"]["code"], "recursion_refused")
        self.assertEqual(self.launch.calls, [])


class CancelAndOpenTests(ConversationTestCase):
    def test_cancel_is_idempotent_and_does_not_rewrite_a_terminal_task(self) -> None:
        task = self.create(session="session-1", state="running")

        first = self.call(
            "neo_task_cancel", {"task_id": task["id"]}, session_id="session-1"
        )
        second = self.call(
            "neo_task_cancel", {"task_id": task["id"]}, session_id="session-1"
        )

        self.assertTrue(first["cancelled"], first)
        self.assertEqual(first["state"], "cancelled")
        self.assertTrue(second["already_terminal"])
        self.assertEqual(second["state"], "cancelled")
        self.assertEqual(first["version"], second["version"])
        self.assertEqual(self.supervisor.calls, [task["id"]])

    def test_cancel_refuses_another_conversations_task(self) -> None:
        task = self.create(session="session-2", state="running")

        body = self.call(
            "neo_task_cancel", {"task_id": task["id"]}, session_id="session-1"
        )

        self.assertFalse(body["ok"])
        self.assertEqual(body["error"]["code"], "task_not_found")
        self.assertEqual(self.supervisor.calls, [])

    def test_open_presents_a_completed_verified_task(self) -> None:
        opener = self.attach_presenter()
        task = self.complete_with_artifact("result.md")

        body = self.call("neo_task_open", {"task_id": task["id"]}, session_id="session-1")

        self.assertTrue(body["ok"], body)
        self.assertTrue(body["presented"])
        self.assertFalse(body["skipped"])
        self.assertEqual(len(opener.paths), 1)
        self.assertTrue(opener.paths[0].endswith("result.md"))
        self.assertNotIn("path", body)

    def test_open_skips_a_task_that_is_not_completed(self) -> None:
        opener = self.attach_presenter()
        task = self.create(session="session-1", state="running")

        body = self.call("neo_task_open", {"task_id": task["id"]}, session_id="session-1")

        self.assertFalse(body["ok"])
        self.assertTrue(body["skipped"])
        self.assertFalse(body["presented"])
        self.assertEqual(opener.paths, [])

    def test_open_refuses_another_conversations_task(self) -> None:
        opener = self.attach_presenter()
        task = self.complete_with_artifact("result.md", session="session-2")

        body = self.call("neo_task_open", {"task_id": task["id"]}, session_id="session-1")

        self.assertFalse(body["ok"])
        self.assertEqual(body["error"]["code"], "task_not_found")
        self.assertEqual(opener.paths, [])


class PreLlmSummaryTests(ConversationTestCase):
    def hook(self):
        return make_pre_llm_call(lambda: self.runtime)

    def test_summary_is_bounded_and_lists_at_most_five_tasks(self) -> None:
        tasks = [
            self.create(
                session="session-summary",
                intent=("task %d " % index) + "x" * 120,
                state="running",
            )
            for index in range(8)
        ]

        body = self.hook()(session_id="session-summary", user_message="how is it going?")

        context = body["context"]
        self.assertLessEqual(len(context), 2000)
        lines = [line for line in context.splitlines() if line.startswith("- ")]
        self.assertLessEqual(len(lines), 5)
        self.assertGreaterEqual(len(lines), 1)
        listed = [task for task in tasks if task["id"] in context]
        self.assertEqual(len(listed), len(lines))

    def test_summary_is_scoped_to_the_session(self) -> None:
        mine = self.create(session="session-summary")
        theirs = self.create(session="session-other")

        context = self.hook()(session_id="session-summary", user_message="status?")["context"]

        self.assertIn(mine["id"], context)
        self.assertNotIn(theirs["id"], context)
        self.assertNotIn("session-other", context)

    def test_summary_flags_a_task_that_needs_the_user(self) -> None:
        task = self.create(session="session-summary", state="running")
        self.runtime.service.transition(
            task["id"], "needs_approval", expected_version=task["version"]
        )

        context = self.hook()(session_id="session-summary", user_message="status?")["context"]

        self.assertIn(task["id"], context)
        self.assertIn("needs approval", context.lower())

    def test_summary_does_not_mutate_conversation_history(self) -> None:
        self.create(session="session-summary")
        history = [
            {"role": "system", "content": "core system prompt"},
            {"role": "user", "content": "earlier turn"},
        ]
        snapshot = [dict(message) for message in history]

        result = self.hook()(
            session_id="session-summary",
            user_message="status?",
            conversation_history=history,
            is_first_turn=False,
            model="m",
            platform="cli",
        )

        self.assertEqual(set(result), {"context"})
        self.assertEqual(history, snapshot)
        self.assertEqual(history[0]["content"], "core system prompt")

    def test_summary_is_silent_on_a_worker_session(self) -> None:
        def refuse():
            raise AssertionError("a worker session must not resolve the runtime")

        hook = make_pre_llm_call(refuse)
        with mock.patch.dict(os.environ, {"NEO_WORKER": "1"}):
            self.assertIsNone(hook(session_id="session-summary", user_message="status?"))

    def test_summary_is_silent_without_a_session(self) -> None:
        self.assertIsNone(self.hook()(session_id=None, user_message="status?"))

    def test_summary_never_raises_when_the_runtime_is_unavailable(self) -> None:
        def broken():
            raise RuntimeError("no profile")

        self.assertIsNone(make_pre_llm_call(broken)(session_id="s", user_message="hi"))


class RegistrationTests(ConversationTestCase):
    class FakeContext:
        def __init__(self) -> None:
            self.tools = {}
            self.hooks = []
            self.skills = []

        def register_tool(self, name=None, toolset=None, schema=None, handler=None, **kwargs):
            self.tools[str(name)] = {
                "toolset": toolset,
                "schema": schema,
                "handler": handler,
            }

        def register_hook(self, name=None, callback=None, **kwargs):
            self.hooks.append(str(name))

        def register_skill(self, name=None, path=None, **kwargs):
            self.skills.append((str(name), Path(path)))

    def test_register_declares_all_seven_tools_hook_and_skill(self) -> None:
        ctx = self.FakeContext()

        register_conversation_tools(ctx, lambda: self.runtime)

        self.assertEqual(set(ctx.tools), set(TOOL_NAMES))
        for entry in ctx.tools.values():
            self.assertEqual(entry["toolset"], "neo")
            self.assertIn("function", entry["schema"])
        self.assertEqual(ctx.hooks, [])
        self.assertEqual(ctx.skills, [])

    def test_package_register_wires_tools_hook_and_bundled_skill(self) -> None:
        spec = importlib.util.spec_from_file_location(
            "neo_package_conversation_under_test",
            REPO_ROOT / "__init__.py",
            submodule_search_locations=[str(REPO_ROOT)],
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        self.addCleanup(sys.modules.pop, spec.name, None)
        spec.loader.exec_module(module)

        ctx = self.FakeContext()
        self.assertIsNone(module.register(ctx))

        self.assertEqual(set(ctx.tools), set(TOOL_NAMES))
        self.assertEqual(ctx.hooks, ["pre_llm_call"])
        self.assertEqual([name for name, _path in ctx.skills], ["neo-operator", "neo-worker"])
        for _name, path in ctx.skills:
            self.assertTrue(path.is_file())

    def test_registration_does_not_touch_the_active_profile(self) -> None:
        import conversation

        ctx = self.FakeContext()
        register_conversation_tools(ctx, lambda: self.runtime)

        self.assertIsNone(conversation._RUNTIME)

    def test_every_registered_handler_returns_json_and_never_raises(self) -> None:
        ctx = self.FakeContext()
        register_conversation_tools(ctx, lambda: self.runtime)

        junk = [None, {}, {"task_id": "nope"}, {"intent": 5}, {"intent": "x" * 5000}]
        for name, entry in ctx.tools.items():
            for args in junk:
                payload = entry["handler"](args, session_id="session-1")
                self.assertIsInstance(payload, str, name)
                body = json.loads(payload)
                self.assertIsInstance(body, dict, name)
                self.assertIn("ok", body, name)


class SchemaAndHelperTests(unittest.TestCase):
    def test_seven_conversation_schemas_are_closed_and_shell_free(self) -> None:
        import schemas

        self.assertEqual(set(schemas.CONVERSATION_TOOL_NAMES), set(TOOL_NAMES))
        for schema in schemas.CONVERSATION_TOOL_SCHEMAS:
            function = schema["function"]
            parameters = function["parameters"]
            self.assertIs(parameters["additionalProperties"], False, function["name"])
            self.assertEqual(schema["type"], "function")
            for prop in parameters["properties"]:
                self.assertNotIn(
                    prop.lower(),
                    {"command", "argv", "shell", "script", "cmd"},
                    function["name"],
                )
            json.dumps(schema)

    def test_conversation_key_prefers_session_then_task(self) -> None:
        self.assertEqual(conversation_key_from({"session_id": " s1 "}), "s1")
        self.assertEqual(conversation_key_from({"task_id": "t1"}), "t1")
        self.assertEqual(conversation_key_from({}), "")
        self.assertEqual(conversation_key_from({"session_id": 5}), "")


if __name__ == "__main__":
    unittest.main()
