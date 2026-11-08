"""Task 20 — the browser vertical slice.

These tests fix the browser contract before any of it is trusted:

* A browser task navigates a harmless page and proves it with two artifacts
  inside its own workdir: ``browser-state.json`` (url, title, status) and a
  screenshot image. A claim in prose is not navigation evidence.
* Publication is never executed here. :func:`classify_publication_action`
  conservatively recognises publish/send/post/email/upload/purchase/deploy/
  comment/share/account actions, and anything it does not recognise as plain
  navigation defaults to requiring approval.
* Approval is an explicit, idempotent decision recorded in ``approval_json``
  with the action fingerprint and the task version. ``approve`` moves the task
  to ``running`` but executes nothing; ``deny`` cancels it. A timeout or an
  absent decision leaves the task in ``needs_approval``. A worker cannot use a
  completion or an update call to bypass that state, and another conversation
  cannot decide this task's approval.
* Completion needs a valid navigation state plus a task-scoped screenshot, and
  a publication whose fingerprint has no approved decision is refused.

Every test uses injected fakes. No test opens a browser, starts a process,
touches the network or publishes anything.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from browser import (  # noqa: E402
    SCREENSHOT_ARTIFACT,
    STATE_ARTIFACT,
    WORKER_SKILLS,
    BrowserApprovalError,
    BrowserCompletionError,
    BrowserSlice,
    build_worker_prompt,
    classify_publication_action,
    publication_fingerprint,
)
from conversation import ConversationTools, NeoRuntime  # noqa: E402

PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32


class LaunchRecorder:
    """Record every argv without starting a process."""

    def __init__(self) -> None:
        self.calls = []

    def __call__(self, task_id, argv, *, role="task"):
        self.calls.append({"task_id": task_id, "argv": list(argv), "role": role})
        return None

    def last(self) -> dict:
        if not self.calls:
            raise AssertionError("nothing was launched")
        return self.calls[-1]


class BrowserCase(unittest.TestCase):
    """A real runtime under a temporary directory; nothing is spawned."""

    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.launch = LaunchRecorder()
        self.approval_notices = []
        self.runtime = NeoRuntime.build(
            data_dir=self.root / "plugin-data",
            launch=self.launch,
            supervisor=object(),
            max_parallel=8,
        )
        self.slice = BrowserSlice(
            runtime=self.runtime,
            launcher=self.launch,
            approval=self.approval_notices.append,
        )

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    # -- helpers ---------------------------------------------------------

    def advance(self, task: dict, target: str, **kwargs) -> dict:
        return self.runtime.service.transition(
            task["id"], target, expected_version=task["version"], **kwargs
        )

    def make_task(
        self,
        *,
        conversation: str = "session-1",
        intent: str = "navigate to https://example.com and report the title",
        running: bool = True,
        workdir: str = None,
    ) -> dict:
        task = self.runtime.service.create_task(
            conversation_key=conversation,
            kind="browser",
            intent=intent,
            workdir=workdir or self.runtime.workspace("browser-task"),
        )
        if running:
            task = self.advance(task, "running")
        return task

    def write(self, task: dict, name: str, body: bytes) -> Path:
        path = Path(task["workdir"]) / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)
        return path

    def write_state(self, task: dict, **overrides) -> Path:
        state = {
            "url": "https://example.com/",
            "title": "Example Domain",
            "status": 200,
        }
        state.update(overrides)
        return self.write(task, STATE_ARTIFACT, json.dumps(state).encode("utf-8"))

    def write_screenshot(self, task: dict, name: str = SCREENSHOT_ARTIFACT) -> Path:
        return self.write(task, name, PNG_BYTES)

    def refresh(self, task: dict) -> dict:
        return self.runtime.registry.get_task(task["id"])


# -- publication classification ----------------------------------------------


class ClassificationTests(unittest.TestCase):
    def test_plain_navigation_does_not_require_approval(self) -> None:
        for text in (
            "navigate to https://example.com and read the page title",
            "open the page and take a screenshot",
            "scroll down and report the visible headings",
            "inspect the instructions on this local page",
        ):
            decision = classify_publication_action(text)
            self.assertFalse(decision["requires_approval"], text)
            self.assertEqual(decision["categories"], (), text)

    def test_publication_table_requires_approval(self) -> None:
        table = {
            "publish the finished article to the blog": "publish",
            "send an email to the team": "send",
            "post a comment on the thread": "post",
            "email the draft to the client": "send",
            "upload the screenshot to the gallery": "upload",
            "buy the cheapest ticket": "purchase",
            "deploy the build to production": "deploy",
            "share the link on social media": "post",
            "delete the account after exporting": "account",
            "sign in with the saved session": "account",
        }
        for text, category in table.items():
            decision = classify_publication_action(text)
            self.assertTrue(decision["requires_approval"], text)
            self.assertIn(category, decision["categories"], text)

    def test_unknown_external_action_defaults_to_approval(self) -> None:
        decision = classify_publication_action("frobnicate the widget remotely")
        self.assertTrue(decision["requires_approval"])
        self.assertEqual(decision["categories"], ("unknown_external_action",))
        self.assertTrue(decision["action"])

    def test_empty_or_non_text_defaults_to_approval(self) -> None:
        for value in ("", "   ", None, 42):
            self.assertTrue(classify_publication_action(value)["requires_approval"])

    def test_fingerprint_is_stable_and_binds_account_and_audience(self) -> None:
        first = publication_fingerprint("publish the draft", "dev@example.com", "public")
        again = publication_fingerprint("publish the draft", "dev@example.com", "public")
        other_account = publication_fingerprint(
            "publish the draft", "other@example.com", "public"
        )
        other_audience = publication_fingerprint(
            "publish the draft", "dev@example.com", "team"
        )
        self.assertEqual(first, again)
        self.assertNotEqual(first, other_account)
        self.assertNotEqual(first, other_audience)
        self.assertRegex(first, r"^[0-9a-f]{64}$")


# -- prompt and task preparation ---------------------------------------------


class PromptTests(unittest.TestCase):
    def test_prompt_carries_every_required_instruction(self) -> None:
        prompt = build_worker_prompt("navigate to a harmless page", "/tmp/browser-task")
        lowered = prompt.lower()
        self.assertIn("navigate to a harmless page", prompt)
        self.assertIn("/tmp/browser-task", prompt)
        self.assertIn("example.com", lowered)
        self.assertIn("visible browser", lowered)
        self.assertIn("not headless", lowered)
        self.assertIn("after each navigation", lowered)
        self.assertIn("browser-state.json", prompt)
        self.assertIn("browser-screenshot.png", prompt)
        self.assertIn("url", lowered)
        self.assertIn("title", lowered)
        self.assertIn("status", lowered)
        for boundary in ("publish", "send", "post", "upload", "purchase", "account"):
            self.assertIn(boundary, lowered)
        self.assertIn("needs_approval", lowered)
        self.assertIn("account", lowered)
        self.assertIn("audience", lowered)
        self.assertIn("timeout", lowered)
        self.assertIn("unpublished", lowered)
        self.assertIn("nested neo", lowered)
        self.assertIn("shell", lowered)

    def test_prompt_never_asks_for_a_login_or_carries_a_credential(self) -> None:
        prompt = build_worker_prompt("navigate to a harmless page", "/tmp/browser-task")
        lowered = prompt.lower()
        for forbidden in ("neo_task_token", "bearer ", "sk-", "password", "cookie ", "authorization:"):
            self.assertNotIn(forbidden, lowered)
        self.assertIn("never log in", lowered)
        self.assertIn("credentials", lowered)

    def test_prompt_is_bounded(self) -> None:
        prompt = build_worker_prompt("navigate", "/tmp/browser-task")
        self.assertLess(len(prompt), 6000)

    def test_prompt_refuses_empty_input(self) -> None:
        with self.assertRaises(ValueError):
            build_worker_prompt("   ", "/tmp/browser-task")


class PrepareTaskTests(BrowserCase):
    def test_prepare_returns_hermes_worker_fields_and_a_prompt_file_argv(self) -> None:
        task_id = "browser-task-1"
        workdir = self.runtime.workspace(task_id)
        fields = self.slice.prepare_task(
            "navigate to https://example.com and report the title",
            "session-1",
            workdir,
            task_id=task_id,
        )

        self.assertEqual(fields["kind"], "browser")
        self.assertEqual(fields["executor"], "hermes_worker")
        self.assertEqual(fields["skills"], WORKER_SKILLS)
        self.assertEqual(fields["skills"], ("browser", "neo-worker"))
        self.assertIn(STATE_ARTIFACT, fields["expected_artifacts"])
        self.assertIn(SCREENSHOT_ARTIFACT, fields["expected_artifacts"])
        self.assertFalse(fields["headless"])

        argv = list(fields["argv"])
        self.assertEqual(argv[0], "hermes")
        self.assertIn("--query-file", argv)
        self.assertEqual(argv[argv.index("--skills") + 1], "browser,neo-worker")
        self.assertNotIn("--", argv)
        prompt_path = Path(argv[argv.index("--query-file") + 1])
        self.assertTrue(prompt_path.is_file())
        self.assertEqual(os.stat(prompt_path).st_mode & 0o777, 0o600)
        self.assertIn("example.com", prompt_path.read_text(encoding="utf-8"))

    def test_prepare_builds_no_argv_without_a_task_id(self) -> None:
        fields = self.slice.prepare_task("navigate", "session-1", self.runtime.workspace("t"))
        self.assertIsNone(fields["prompt_path"])
        self.assertEqual(fields["argv"], ())

    def test_prepare_requires_an_absolute_existing_workdir(self) -> None:
        for workdir in ("relative/path", "/definitely/not/here", None):
            with self.assertRaises(ValueError):
                self.slice.prepare_task("navigate", "session-1", workdir, task_id="t")

    def test_prepare_requires_an_intent_and_a_conversation(self) -> None:
        workdir = self.runtime.workspace("t")
        with self.assertRaises(ValueError):
            self.slice.prepare_task("   ", "session-1", workdir, task_id="t")
        with self.assertRaises(ValueError):
            self.slice.prepare_task("navigate", "", workdir, task_id="t")

    def test_launcher_is_injected_and_receives_no_shell(self) -> None:
        fields = self.slice.prepare_task(
            "navigate to https://example.com",
            "session-1",
            self.runtime.workspace("t"),
            task_id="t",
        )
        self.slice.launch("t", fields)
        call = self.launch.last()
        self.assertEqual(call["task_id"], "t")
        self.assertIsInstance(call["argv"], list)
        self.assertFalse(any(token in ("-c", "--command", "sh", "bash") for token in call["argv"][1:]))


# -- approval ----------------------------------------------------------------


class ApprovalTests(BrowserCase):
    def test_request_approval_records_the_exact_action_account_and_audience(self) -> None:
        task = self.make_task()
        result = self.slice.request_approval(
            task,
            reason="the next step publishes the draft",
            action="publish the draft",
            account="dev@example.com",
            audience="public",
            payload={"url": "https://example.com/draft"},
        )
        self.assertEqual(result["state"], "needs_approval")
        stored = self.refresh(task)
        self.assertEqual(stored["state"], "needs_approval")
        approval = json.loads(stored["approval_json"])
        self.assertEqual(approval["decision"], None)
        self.assertEqual(approval["reason"], "the next step publishes the draft")
        self.assertEqual(approval["action"], "publish the draft")
        self.assertEqual(approval["account"], "dev@example.com")
        self.assertEqual(approval["audience"], "public")
        self.assertEqual(
            approval["fingerprint"],
            publication_fingerprint("publish the draft", "dev@example.com", "public"),
        )
        self.assertEqual(approval["task_version"], stored["version"])
        self.assertEqual(approval["payload"], {"url": "https://example.com/draft"})
        self.assertFalse(approval["publication_executed"])
        self.assertEqual(len(self.approval_notices), 1)
        self.assertEqual(self.approval_notices[0]["account"], "dev@example.com")
        self.assertEqual(self.launch.calls, [])

    def test_approve_moves_to_running_and_executes_no_publication(self) -> None:
        task = self.make_task()
        self.slice.request_approval(
            task, reason="publish", action="publish the draft", account="dev@example.com",
            audience="public",
        )
        result = self.slice.record_approval(task["id"], "approve", reason="ok")
        stored = self.refresh(task)

        self.assertEqual(result["decision"], "approve")
        self.assertFalse(result["idempotent"])
        self.assertFalse(result["publication_executed"])
        self.assertEqual(stored["state"], "running")
        approval = json.loads(stored["approval_json"])
        self.assertEqual(approval["decision"], "approve")
        self.assertEqual(approval["reason"], "ok")
        self.assertFalse(approval["publication_executed"])
        self.assertEqual(self.launch.calls, [], "approval must not publish or launch anything")

    def test_deny_cancels_the_task(self) -> None:
        task = self.make_task()
        self.slice.request_approval(task, reason="publish", action="publish the draft")
        result = self.slice.record_approval(task["id"], "deny", reason="no")
        stored = self.refresh(task)
        self.assertEqual(result["decision"], "deny")
        self.assertEqual(stored["state"], "cancelled")
        self.assertEqual(json.loads(stored["approval_json"])["decision"], "deny")
        self.assertEqual(self.launch.calls, [])

    def test_timeout_and_absent_decision_leave_the_task_needs_approval(self) -> None:
        slice_ = BrowserSlice(
            runtime=self.runtime,
            launcher=self.launch,
            approval=self.approval_notices.append,
            clock=lambda: 1000.0,
            timeout_seconds=10.0,
        )
        task = self.make_task()
        slice_.request_approval(task, reason="publish", action="publish the draft")
        stored = self.refresh(task)

        status = slice_.approval_status(stored, now=1005.0)
        self.assertTrue(status["pending"])
        self.assertFalse(status["timed_out"])
        expired = slice_.expire_approval(task["id"], now=1011.0)
        self.assertTrue(expired["timed_out"])
        self.assertIsNone(expired["decision"])
        self.assertEqual(self.refresh(task)["state"], "needs_approval")
        self.assertEqual(self.launch.calls, [])

    def test_repeated_identical_decisions_are_idempotent(self) -> None:
        task = self.make_task()
        self.slice.request_approval(task, reason="publish", action="publish the draft")
        first = self.slice.record_approval(task["id"], "approve")
        second = self.slice.record_approval(task["id"], "approve")
        self.assertFalse(first["idempotent"])
        self.assertTrue(second["idempotent"])
        self.assertEqual(first["task"]["version"], second["task"]["version"])
        self.assertEqual(first["task"]["state"], second["task"]["state"])
        events = self.runtime.service.task_events("session-1", task["id"])
        approvals = [event for event in events if event.get("kind") == "task.approval_required"]
        self.assertEqual(len(approvals), 1, "an idempotent decision emits no second event")

    def test_a_conflicting_decision_is_refused(self) -> None:
        task = self.make_task()
        self.slice.request_approval(task, reason="publish", action="publish the draft")
        self.slice.record_approval(task["id"], "approve")
        with self.assertRaises(BrowserApprovalError):
            self.slice.record_approval(task["id"], "deny")
        self.assertEqual(self.refresh(task)["state"], "running")

    def test_another_conversation_cannot_decide_the_task(self) -> None:
        task = self.make_task(conversation="session-1")
        self.slice.request_approval(task, reason="publish", action="publish the draft")
        with self.assertRaises(BrowserApprovalError):
            self.slice.record_approval(
                task["id"], "approve", conversation_key="session-2"
            )
        self.assertEqual(self.refresh(task)["state"], "needs_approval")

    def test_a_worker_cannot_update_or_complete_past_approval(self) -> None:
        task = self.make_task()
        self.slice.request_approval(task, reason="publish", action="publish the draft")
        stored = self.refresh(task)
        with self.assertRaises(BrowserApprovalError):
            self.slice.guard_worker_update(stored, "running")
        with self.assertRaises(BrowserApprovalError):
            self.slice.guard_worker_update(stored, "completed")
        # A cancellation request is not a bypass; it is the honest terminal answer.
        self.assertTrue(self.slice.guard_worker_update(stored, "cancelled"))


# -- navigation evidence and completion --------------------------------------


class NavigationTests(BrowserCase):
    def test_navigation_state_and_screenshot_verify(self) -> None:
        task = self.make_task()
        self.write_state(task, final_url="https://example.com/done")
        self.write_screenshot(task)
        navigation = self.slice.verify_navigation(task)
        self.assertEqual(navigation["url"], "https://example.com/")
        self.assertEqual(navigation["title"], "Example Domain")
        self.assertEqual(navigation["status"], 200)
        self.assertEqual(navigation["screenshot"], SCREENSHOT_ARTIFACT)
        self.assertEqual(
            set(navigation["artifacts"]), {STATE_ARTIFACT, SCREENSHOT_ARTIFACT}
        )

    def test_missing_or_invalid_state_refuses_navigation(self) -> None:
        task = self.make_task()
        self.write_screenshot(task)
        with self.assertRaises(BrowserCompletionError):
            self.slice.verify_navigation(task)

        self.write_state(task, url="file:///etc/passwd")
        with self.assertRaises(BrowserCompletionError):
            self.slice.verify_navigation(task)

        self.write_state(task, status="nonsense")
        with self.assertRaises(BrowserCompletionError):
            self.slice.verify_navigation(task)

        self.write_state(task, title=7)
        with self.assertRaises(BrowserCompletionError):
            self.slice.verify_navigation(task)

    def test_missing_screenshot_refuses_navigation(self) -> None:
        task = self.make_task()
        self.write_state(task)
        with self.assertRaises(BrowserCompletionError):
            self.slice.verify_navigation(task)

    def test_a_screenshot_outside_the_workdir_is_refused(self) -> None:
        task = self.make_task()
        self.write_state(task)
        outside = self.root / "outside.png"
        outside.write_bytes(PNG_BYTES)
        with self.assertRaises(BrowserCompletionError):
            self.slice.verify_navigation(task, screenshot="../outside.png")

    def test_completion_returns_verified_evidence_for_navigation_only(self) -> None:
        task = self.make_task()
        self.write_state(task)
        self.write_screenshot(task)
        evidence = self.slice.verify_completion(task)
        self.assertTrue(evidence.executor_confirmed)
        self.assertEqual(
            set(evidence.verified_artifacts), {STATE_ARTIFACT, SCREENSHOT_ARTIFACT}
        )
        record = self.slice.completion_record(task)
        self.assertFalse(record["publication_executed"])
        self.assertEqual(record["outcome"], "navigation_only")


class PublicationCompletionTests(BrowserCase):
    def _publication_task(self, approval: dict = None) -> dict:
        task = self.make_task()
        self.write_state(task, publication=True, action="publish the draft",
                         account="dev@example.com", audience="public")
        self.write_screenshot(task)
        if approval is not None:
            task = self.runtime.registry.update_task(
                task["id"],
                expected_version=task["version"],
                approval_json=json.dumps(approval, sort_keys=True),
            )
        return self.refresh(task)

    def test_completion_refuses_a_publication_without_an_approved_decision(self) -> None:
        task = self._publication_task()
        with self.assertRaises(BrowserCompletionError):
            self.slice.verify_completion(task)

    def test_completion_refuses_a_pending_approval_even_without_a_publication_claim(self) -> None:
        task = self.make_task()
        self.write_state(task)
        self.write_screenshot(task)
        self.slice.request_approval(task, reason="publish", action="publish the draft")
        with self.assertRaises(BrowserCompletionError):
            self.slice.verify_completion(self.refresh(task))

    def test_completion_refuses_an_approved_fingerprint_for_a_different_action(self) -> None:
        approved = {
            "decision": "approve",
            "reason": "ok",
            "action": "publish something else",
            "account": "dev@example.com",
            "audience": "public",
            "fingerprint": publication_fingerprint(
                "publish something else", "dev@example.com", "public"
            ),
            "task_version": 3,
            "publication_executed": False,
        }
        task = self._publication_task(approved)
        with self.assertRaises(BrowserCompletionError):
            self.slice.verify_completion(task)

    def test_an_approved_publication_completes_as_navigation_only(self) -> None:
        approved = {
            "decision": "approve",
            "reason": "ok",
            "action": "publish the draft",
            "account": "dev@example.com",
            "audience": "public",
            "fingerprint": publication_fingerprint(
                "publish the draft", "dev@example.com", "public"
            ),
            "task_version": 3,
            "publication_executed": False,
        }
        task = self._publication_task(approved)
        evidence = self.slice.verify_completion(task)
        self.assertTrue(evidence.executor_confirmed)
        record = self.slice.completion_record(task)
        self.assertFalse(record["publication_executed"])
        self.assertEqual(record["outcome"], "navigation_only")
        self.assertEqual(self.launch.calls, [], "completion publishes nothing")


# -- conversation routing ----------------------------------------------------


class ConversationRoutingTests(BrowserCase):
    def setUp(self) -> None:
        super().setUp()
        self.tools = ConversationTools(lambda: self.runtime)

    def submit(self, intent: str, kind: str = "browser", **kwargs) -> dict:
        payload = self.tools.neo_submit_task(
            {"intent": intent, "kind": kind}, session_id="session-1", **kwargs
        )
        body = json.loads(payload)
        self.assertIsInstance(body, dict)
        return body

    def test_browser_submission_uses_the_slice_when_present(self) -> None:
        self.runtime.browser = self.slice
        body = self.submit("navigate to https://example.com and report the title")
        self.assertTrue(body["ok"], body)
        self.assertEqual(body["kind"], "browser")
        self.assertEqual(body["executor"], "hermes_worker")

        task = self.runtime.registry.get_task(body["task_id"])
        expected = json.loads(task["expected_json"])
        self.assertIn(STATE_ARTIFACT, expected)
        self.assertIn(SCREENSHOT_ARTIFACT, expected)

        argv = self.launch.last()["argv"]
        self.assertEqual(argv[argv.index("--skills") + 1], "browser,neo-worker")
        prompt_path = Path(argv[argv.index("--query-file") + 1])
        self.assertIn("example.com", prompt_path.read_text(encoding="utf-8"))

    def test_without_the_slice_the_legacy_worker_route_keeps_running(self) -> None:
        body = self.submit("open the local docs page in a browser")
        self.assertTrue(body["ok"], body)
        argv = self.launch.last()["argv"]
        self.assertEqual(argv[argv.index("--skills") + 1], "neo-worker")

    def test_browser_slice_refusal_creates_nothing(self) -> None:
        self.runtime.browser = BrowserSlice(runtime=None)
        body = self.submit("navigate to https://example.com")
        self.assertFalse(body["ok"])
        self.assertEqual(body["error"]["code"], "browser_unavailable")
        self.assertEqual(self.launch.calls, [])
        self.assertEqual(
            [row for row in self.runtime.registry.list_tasks() if row["kind"] == "browser"],
            [],
        )


if __name__ == "__main__":
    unittest.main()
