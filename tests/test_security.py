"""Task 23 — security and privacy hardening.

These tests pin the boundaries NEO promises, not the shape of its internals:

* credentials, bearer tokens, API-key shapes, private-key blocks and the user's
  home path never leave the log tail route, and the tail is capped before and
  after redaction;
* a task can only read its own log, never another task's file, even when the
  stored path or a link points at one;
* the dashboard task body carries a safe workspace label instead of an absolute
  workdir, and never an arbitrary approval payload or an internal column;
* conversational JSON (submit, status, list, open) is path-free and
  spoken-safe;
* the ``pre_llm_call`` hook returns context only and mutates nothing;
* capability truth comes from the host's public lookups captured at
  ``register()`` time, never from configuration files, and blender/browser fail
  closed when those lookups are absent or disagree;
* preview and artifact containment, symlink refusal and byte caps still hold
  under adversarial names;
* workers get argv-only spawns and an optional environment allowlist, and
  capability installs stay pinned and consent-bound.

The API half skips when FastAPI is not installed (system interpreter), exactly
like the rest of the backend suite.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DASHBOARD_DIR = REPO_ROOT / "dashboard"
if str(DASHBOARD_DIR) not in sys.path:
    sys.path.insert(0, str(DASHBOARD_DIR))

try:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    FASTAPI_AVAILABLE = True
except ModuleNotFoundError:  # pragma: no cover - system python lacks fastapi
    FASTAPI_AVAILABLE = False

from conversation import (  # noqa: E402
    CapabilityBridge,
    ConversationTools,
    NeoRuntime,
    current_capability_context,
    make_pre_llm_call,
    register_capability_context,
    reset_capability_context,
    reset_runtime,
)
from neo_tools import (  # noqa: E402
    PATH_PLACEHOLDER,
    REDACTED,
    WorkerBridge,
    WorkerTokenStore,
    cap_text_bytes,
    redact_text,
    scrub_paths,
    workspace_label,
)
from registry import Registry  # noqa: E402
from events import TaskService  # noqa: E402
from state_machine import CompletionEvidence  # noqa: E402
from workers import worker_environment  # noqa: E402

HOME = str(Path.home())
USERNAME = Path.home().name

PNG = b"\x89PNG\r\n\x1a\n" + b"pixels" * 4


class LaunchRecorder:
    """Record argv; never start a process."""

    def __init__(self):
        self.calls = []

    def __call__(self, task_id, argv, *, role="task"):
        self.calls.append({"task_id": task_id, "argv": list(argv), "role": role})
        return None


    def last(self):
        if not self.calls:
            raise AssertionError("nothing was launched")
        return self.calls[-1]


class FakeSupervisor:
    def __init__(self):
        self.service = None
        self.calls = []

    def cancel(self, task_id, **_kwargs):
        self.calls.append(task_id)
        task = self.service.get_task(task_id)
        if task is None:
            raise KeyError(task_id)
        if task["state"] in ("completed", "failed", "cancelled"):
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
        return self.service.transition(
            task_id, "cancelled", expected_version=task["version"]
        )


# -- redaction primitives -----------------------------------------------------


class RedactionTests(unittest.TestCase):
    def test_redacts_task_token_assignments(self) -> None:
        text = "env NEO_TASK_TOKEN=abc123secretvalue other=1"

        cleaned = redact_text(text)

        self.assertNotIn("abc123secretvalue", cleaned)
        self.assertIn("NEO_TASK_TOKEN=", cleaned)
        self.assertIn(REDACTED, cleaned)

    def test_redacts_quoted_and_bearer_credentials(self) -> None:
        text = (
            '{"token": "json-secret-value"} Authorization: Bearer bearer-secret-value '
            "api_key='quoted-secret-value'"
        )

        cleaned = redact_text(text)

        for secret in ("json-secret-value", "bearer-secret-value", "quoted-secret-value"):
            self.assertNotIn(secret, cleaned)

    def test_redacts_known_api_key_shapes(self) -> None:
        keys = (
            "sk-proj-abcdefghijklmnopqrstuvwxyz012345",
            "ghp_abcdefghijklmnopqrstuvwxyz0123456789",
            "xoxb-1234567890-abcdefghijkl",
            "AKIAIOSFODNN7EXAMPLE",
        )
        text = "keys: " + " ".join(keys)

        cleaned = redact_text(text)

        for key in keys:
            self.assertNotIn(key, cleaned)

    def test_redacts_private_key_blocks(self) -> None:
        text = (
            "before\n-----BEGIN RSA PRIVATE KEY-----\nMIIEow\n-----END RSA PRIVATE KEY-----\n"
            "after"
        )

        cleaned = redact_text(text)

        self.assertNotIn("MIIEow", cleaned)
        self.assertNotIn("BEGIN RSA PRIVATE KEY", cleaned)
        self.assertIn("before", cleaned)
        self.assertIn("after", cleaned)

    def test_redacts_home_path_and_other_user_homes(self) -> None:
        text = "cwd=%s/code/work other=/Users/someoneelse/private" % HOME

        cleaned = redact_text(text)

        self.assertNotIn(HOME, cleaned)
        self.assertNotIn("/Users/someoneelse", cleaned)
        self.assertIn("~", cleaned)

    def test_keeps_ordinary_worker_output(self) -> None:
        text = "worker exited with status 0 without reporting completion"

        self.assertEqual(redact_text(text), text)

    def test_scrub_paths_removes_every_absolute_path(self) -> None:
        text = "cannot read /etc/passwd or /private/tmp/work/report.md"

        cleaned = scrub_paths(text)

        self.assertNotIn("/etc/passwd", cleaned)
        self.assertNotIn("/private/tmp", cleaned)
        self.assertIn(PATH_PLACEHOLDER, cleaned)

    def test_scrub_paths_keeps_relative_names_and_urls(self) -> None:
        text = "opened https://example.com/docs and wrote out/report.md"

        cleaned = scrub_paths(text)

        self.assertEqual(cleaned, text)

    def test_cap_text_bytes_is_utf8_safe(self) -> None:
        text = "án" * 100

        capped = cap_text_bytes(text, 25)

        self.assertLessEqual(len(capped.encode("utf-8")), 25)
        self.assertTrue(text.startswith(capped))

    def test_redact_text_applies_the_cap_after_redaction(self) -> None:
        text = "NEO_TASK_TOKEN=%s trailing words" % ("x" * 500)

        cleaned = redact_text(text, cap=32)

        self.assertLessEqual(len(cleaned.encode("utf-8")), 32)
        self.assertNotIn("x" * 40, cleaned)

    def test_workspace_label_is_a_basename_never_a_path(self) -> None:
        self.assertEqual(workspace_label("/Users/user/code/rocket"), "rocket")
        self.assertEqual(workspace_label("/Users/user/code/rocket/"), "rocket")
        self.assertEqual(workspace_label(HOME), "home")
        self.assertIsNone(workspace_label("/"))
        self.assertIsNone(workspace_label(""))
        self.assertIsNone(workspace_label(None))
        self.assertNotIn("/", workspace_label("/Users/user/code/rocket"))


# -- worker environment -------------------------------------------------------


class WorkerEnvironmentTests(unittest.TestCase):
    def test_default_inherits_the_user_environment(self) -> None:
        source = {"PATH": "/bin", "HOME": "/home/x", "AWS_SECRET_ACCESS_KEY": "s"}

        environment = worker_environment("task-1", "token", environ=source)

        self.assertEqual(environment["AWS_SECRET_ACCESS_KEY"], "s")
        self.assertEqual(environment["NEO_WORKER"], "1")
        self.assertEqual(environment["NEO_TASK_ID"], "task-1")
        self.assertEqual(environment["NEO_TASK_TOKEN"], "token")

    def test_allowlist_keeps_named_essentials_and_neo_variables(self) -> None:
        source = {
            "PATH": "/bin",
            "HOME": "/home/x",
            "OPENAI_API_KEY": "sk-listed",
            "AWS_SECRET_ACCESS_KEY": "drop-me",
            "NEO_PREVIEW_DIR": "/tmp/previews",
            "NEO_WORKER_ENV_ALLOWLIST": "OPENAI_API_KEY",
        }

        environment = worker_environment("task-1", None, environ=source)

        self.assertEqual(environment["OPENAI_API_KEY"], "sk-listed")
        self.assertEqual(environment["PATH"], "/bin")
        self.assertEqual(environment["HOME"], "/home/x")
        self.assertEqual(environment["NEO_PREVIEW_DIR"], "/tmp/previews")
        self.assertNotIn("AWS_SECRET_ACCESS_KEY", environment)
        self.assertNotIn("NEO_TASK_TOKEN", environment)

    def test_malformed_allowlist_is_refused(self) -> None:
        for raw in ("FOO BAR", "1BAD", "OK,,BAD-NAME", " "):
            with self.subTest(raw=raw):
                with self.assertRaises(ValueError):
                    worker_environment(
                        "task-1", None, environ={"PATH": "/bin", "NEO_WORKER_ENV_ALLOWLIST": raw}
                    )


# -- runtime and capability truth ---------------------------------------------


class FakeContext:
    """A minimal stand-in for Hermes' public plugin context."""

    def __init__(self, *, plugins=(), enabled=None, skills=(), tools=None, hooks=None, revisions=None):
        self.plugins = set(plugins)
        self.enabled = enabled
        self.skills = list(skills)
        self.revisions = revisions or {}
        self.registered_tools = {}
        self.registered_hooks = []
        self.registered_skills = []

    def has_plugin(self, name):
        return name in self.plugins

    def is_plugin_enabled(self, name):
        if self.enabled is None:
            return name in self.plugins
        return bool(self.enabled.get(name))

    def find_skills(self):
        return list(self.skills)

    def get_installed_sha(self, name):
        if name == "blender" and name in self.plugins:
            return self.revisions.get(
                name, "9b11ffeda84b077bea9de52d22733c7a41ca581a"
            )
        return self.revisions.get(name)

    def register_tool(self, name=None, toolset=None, schema=None, handler=None, **kwargs):
        self.registered_tools[str(name)] = handler

    def register_hook(self, name=None, callback=None, **kwargs):
        self.registered_hooks.append(str(name))

    def register_skill(self, name=None, path=None, **kwargs):
        self.registered_skills.append(str(name))


class CapabilityContextCase(unittest.TestCase):
    def setUp(self) -> None:
        reset_capability_context()
        reset_runtime()
        self._tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tempdir.cleanup)
        self.root = Path(self._tempdir.name)
        self.launch = LaunchRecorder()
        self.addCleanup(reset_capability_context)
        self.addCleanup(reset_runtime)

    def build(self, **kwargs):
        supervisor = FakeSupervisor()
        runtime = NeoRuntime.build(
            data_dir=self.root / "plugin-data",
            launch=self.launch,
            supervisor=supervisor,
            max_parallel=8,
            **kwargs,
        )
        supervisor.service = runtime.service
        return runtime


class CapabilityTruthTests(CapabilityContextCase):
    def test_registration_captures_host_lookups(self) -> None:
        ctx = FakeContext(plugins={"blender"})

        bridge = register_capability_context(ctx)

        self.assertIsNotNone(bridge)
        self.assertIs(current_capability_context(), bridge)
        self.assertTrue(bridge.has_plugin("blender"))
        self.assertFalse(bridge.has_plugin("browser"))
        self.assertEqual(bridge.find_skills(), [])

    def test_runtime_uses_the_registered_lookups_for_blender(self) -> None:
        register_capability_context(FakeContext(plugins={"blender"}))
        runtime = self.build()

        decision = runtime.blender.readiness()

        self.assertTrue(decision["ready"], decision)
        self.assertEqual(decision["route"], "hermes_worker")

    def test_unreviewed_blender_revision_is_not_ready(self) -> None:
        register_capability_context(
            FakeContext(plugins={"blender"}, revisions={"blender": "a" * 40})
        )
        runtime = self.build()

        self.assertFalse(runtime.blender.readiness()["ready"])

    def test_disabled_plugin_is_not_ready(self) -> None:
        register_capability_context(
            FakeContext(plugins={"blender"}, enabled={"blender": False})
        )
        runtime = self.build()

        decision = runtime.blender.readiness()

        self.assertFalse(decision["ready"], decision)
        self.assertEqual(decision["status"], "needs_approval")
        self.assertTrue(decision["plugin"]["installed"])
        self.assertFalse(decision["plugin"]["enabled"])

    def test_no_registration_fails_closed_for_blender_and_browser(self) -> None:
        runtime = self.build()
        tools = ConversationTools(lambda: runtime)

        blender = json.loads(
            tools.neo_submit_task(
                {"intent": "make a low-poly rocket in Blender", "kind": "blender"},
                session_id="session-1",
            )
        )
        browser = json.loads(
            tools.neo_submit_task(
                {"intent": "open the docs page in a browser", "kind": "browser"},
                session_id="session-1",
            )
        )

        self.assertFalse(blender["ok"], blender)
        self.assertEqual(blender["state"], "needs_approval")
        self.assertFalse(browser["ok"], browser)
        self.assertEqual(browser["error"]["code"], "browser_unavailable")
        self.assertEqual(self.launch.calls, [])
        self.assertEqual(runtime.registry.list_tasks(), [])

    def test_browser_route_needs_a_visible_browser_capability(self) -> None:
        register_capability_context(
            FakeContext(plugins=(), skills=[{"name": "browser", "enabled": True}])
        )
        runtime = self.build()
        tools = ConversationTools(lambda: runtime)

        body = json.loads(
            tools.neo_submit_task(
                {"intent": "open the docs page in a browser", "kind": "browser"},
                session_id="session-1",
            )
        )

        self.assertTrue(body["ok"], body)
        self.assertEqual(body["executor"], "hermes_worker")
        self.assertEqual(self.launch.last()["argv"][0], "hermes")

    def test_disabled_browser_skill_fails_closed(self) -> None:
        register_capability_context(
            FakeContext(plugins=(), skills=[{"name": "browser", "enabled": False}])
        )
        runtime = self.build()
        tools = ConversationTools(lambda: runtime)

        body = json.loads(
            tools.neo_submit_task(
                {"intent": "open the docs page in a browser", "kind": "browser"},
                session_id="session-1",
            )
        )

        self.assertFalse(body["ok"], body)
        self.assertEqual(body["error"]["code"], "browser_unavailable")
        self.assertEqual(runtime.registry.list_tasks(), [])

    def test_a_second_registration_does_not_leak_the_first_profile(self) -> None:
        register_capability_context(FakeContext(plugins={"blender"}))
        first = self.build()

        register_capability_context(FakeContext(plugins={"browser"}, skills=[]))
        second = self.build()

        self.assertTrue(first.blender.readiness()["ready"])
        self.assertFalse(second.blender.readiness()["ready"])
        self.assertTrue(second.capabilities.list_installed(("browser",)).installed[0].enabled)

    def test_reset_returns_to_fail_closed(self) -> None:
        register_capability_context(FakeContext(plugins={"blender"}))
        reset_capability_context()

        self.assertIsNone(current_capability_context())

    def test_package_register_captures_the_context_and_drops_a_stale_runtime(self) -> None:
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "neo_security_package_under_test",
            REPO_ROOT / "__init__.py",
            submodule_search_locations=[str(REPO_ROOT)],
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        self.addCleanup(sys.modules.pop, spec.name, None)
        spec.loader.exec_module(module)
        ctx = FakeContext(plugins={"blender"}, skills=[])

        module.register(ctx)

        bridge = current_capability_context()
        self.assertIsNotNone(bridge)
        self.assertTrue(bridge.has_plugin("blender"))
        self.assertEqual(ctx.registered_hooks, ["pre_llm_call"])

    def test_conversation_never_reads_configuration_files(self) -> None:
        source = (REPO_ROOT / "conversation.py").read_text(encoding="utf-8")

        for forbidden in ("get_config(", "load_config(", "config.yaml", "config.json"):
            self.assertNotIn(forbidden, source)


# -- the conversational surface ------------------------------------------------


class ConversationSafetyCase(unittest.TestCase):
    def setUp(self) -> None:
        reset_capability_context()
        reset_runtime()
        self._tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tempdir.cleanup)
        self.root = Path(self._tempdir.name)
        self.launch = LaunchRecorder()
        self.supervisor = FakeSupervisor()
        self.runtime = NeoRuntime.build(
            data_dir=self.root / "plugin-data",
            launch=self.launch,
            supervisor=self.supervisor,
            max_parallel=8,
            capabilities=CapabilityBridge(
                has_plugin=lambda name: name == "browser",
                find_skills=lambda: (),
            ),
        )
        self.supervisor.service = self.runtime.service
        self.tools = ConversationTools(lambda: self.runtime)
        self.addCleanup(reset_capability_context)
        self.addCleanup(reset_runtime)

    def call(self, name, args=None, **kwargs):
        body = json.loads(getattr(self.tools, name)(args or {}, **kwargs))
        self.assertIsInstance(body, dict)
        return body

    def task(self, *, state="running", kind="coding", intent="do the work", workdir=None):
        row = self.runtime.service.create_task(
            conversation_key="session-1",
            kind=kind,
            intent=intent,
            workdir=workdir or self.runtime.workspace("seed"),
        )
        row = self.runtime.registry.update_task(
            row["id"], expected_version=row["version"], executor="codex"
        )
        if state != row["state"]:
            if row["state"] == "queued":
                row = self.runtime.service.transition(
                    row["id"], "starting", expected_version=row["version"]
                )
            if row["state"] == "starting":
                row = self.runtime.service.transition(
                    row["id"], "running", expected_version=row["version"]
                )
            if state != "running":
                row = self.runtime.service.transition(
                    row["id"], state, expected_version=row["version"],
                    evidence=(
                        CompletionEvidence(executor_confirmed=True)
                        if state == "completed"
                        else None
                    ),
                )
        return row


class ConversationSafetyTests(ConversationSafetyCase):
    def test_status_error_detail_is_path_free(self) -> None:
        task = self.task(state="running")
        self.runtime.registry.update_task(
            task["id"],
            expected_version=task["version"],
            error_code="executor_config",
            error_detail="cannot open %s/secret/report.md" % HOME,
        )

        body = self.call("neo_task_status", {"task_id": task["id"]}, session_id="session-1")

        detail = body["error"]["detail"]
        self.assertNotIn(HOME, detail)
        self.assertNotIn("/secret/report.md", detail)

    def test_status_summary_is_path_free(self) -> None:
        task = self.task(state="running")
        self.runtime.registry.update_task(
            task["id"],
            expected_version=task["version"],
            result_summary="wrote %s/work/out.md" % HOME,
        )

        body = self.call("neo_task_status", {"task_id": task["id"]}, session_id="session-1")

        self.assertNotIn(HOME, body["summary"])

    def test_submit_refusal_detail_is_path_free(self) -> None:
        class BrokenCodex:
            def build_initial_argv(self, intent, *, workdir=None):
                raise OSError("executable not found at %s/bin/codex" % HOME)

        self.runtime.codex = BrokenCodex()

        body = self.call("neo_submit_task", {"intent": "write a parser"}, session_id="session-1")

        self.assertFalse(body["ok"], body)
        self.assertNotIn(HOME, json.dumps(body))

    def test_browser_slice_error_detail_is_path_free(self) -> None:
        class BrokenBrowser:
            def prepare_task(self, intent, conversation, workdir, *, task_id=None):
                raise RuntimeError("cannot read %s/private/page.html" % HOME)

        self.runtime.browser = BrokenBrowser()

        body = self.call(
            "neo_submit_task",
            {"intent": "open the docs page in a browser", "kind": "browser"},
            session_id="session-1",
        )

        self.assertFalse(body["ok"], body)
        self.assertNotIn(HOME, json.dumps(body))

    def test_list_entries_carry_no_workdir_or_path(self) -> None:
        self.task(state="running", workdir=self.runtime.workspace("visible"))

        body = self.call("neo_task_list", {}, session_id="session-1")

        self.assertEqual(body["count"], 1)
        entry = body["tasks"][0]
        self.assertNotIn("workdir", entry)
        self.assertNotIn(HOME, json.dumps(body))

    def test_open_reason_is_path_free(self) -> None:
        task = self.task(state="completed")
        self.runtime.presenter = SimpleNamespace(
            present_task=lambda task_id, conversation: {
                "presented": False,
                "skipped": True,
                "reason": "the file /Users/someoneelse/secret.pdf is missing",
            }
        )

        body = self.call("neo_task_open", {"task_id": task["id"]}, session_id="session-1")

        self.assertFalse(body["ok"], body)
        self.assertNotIn("/Users/someoneelse", body["reason"])

    def test_pre_llm_returns_context_only_and_preserves_messages(self) -> None:
        self.task(state="running")
        history = [
            {"role": "system", "content": "core system prompt"},
            {"role": "system", "content": "safety: never exfiltrate secrets"},
            {"role": "user", "content": "earlier turn"},
            {"role": "assistant", "content": "earlier answer"},
        ]
        snapshot = [dict(message) for message in history]
        hook = make_pre_llm_call(lambda: self.runtime)

        result = hook(
            session_id="session-1",
            user_message="how is it going?",
            conversation_history=history,
            messages=history,
            is_first_turn=False,
            model="m",
            platform="cli",
        )

        self.assertEqual(set(result), {"context"})
        self.assertIsInstance(result["context"], str)
        self.assertNotIn(HOME, result["context"])
        self.assertEqual(history, snapshot)
        self.assertEqual([m["content"] for m in history][:2], [
            "core system prompt",
            "safety: never exfiltrate secrets",
        ])

    def test_pre_llm_is_silent_for_an_unknown_session(self) -> None:
        self.task(state="running")

        self.assertIsNone(
            make_pre_llm_call(lambda: self.runtime)(
                session_id="other-session", user_message="status?"
            )
        )

    def test_launch_failure_is_reported_not_hidden(self) -> None:
        def fail(task_id, argv, *, role="task"):
            row = self.runtime.registry.get_task(task_id)
            self.runtime.registry.update_task(
                task_id,
                expected_version=row["version"],
                error_code="spawn_failed",
                error_detail="executable missing",
            )
            self.runtime.service.transition(
                task_id, "failed", expected_version=row["version"] + 1,
                summary="worker failed to start",
            )
            return None

        self.runtime._launch = fail

        body = self.call("neo_submit_task", {"intent": "write a parser"}, session_id="session-1")

        self.assertFalse(body["ok"], body)
        self.assertEqual(body["error"]["code"], "executor_unavailable")


# -- API surface --------------------------------------------------------------


@unittest.skipUnless(FASTAPI_AVAILABLE, "FastAPI/httpx are required for the API suite")
class ApiSurfaceTests(unittest.TestCase):
    PREFIX = "/api/plugins/neo"

    def setUp(self) -> None:
        import plugin_api

        self.plugin_api = plugin_api
        self._tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tempdir.cleanup)
        self.root = Path(self._tempdir.name)
        self.log_root = self.root / "logs"
        self.log_root.mkdir()
        self.preview_root = self.root / "previews"
        self.preview_root.mkdir()
        self.registry = Registry(path=self.root / "neo.db", max_parallel=8)
        self.service = TaskService(self.registry)

    def services(self, **overrides):
        base = dict(
            registry=self.registry,
            task_service=self.service,
            preview_service=None,
            supervisor=None,
            retry=None,
            capabilities=None,
            installer=None,
            log_root=self.log_root,
            artifact_max_bytes=1024 * 1024,
            log_max_bytes=64 * 1024,
        )
        base.update(overrides)
        return SimpleNamespace(**base)

    def client(self, services=None):
        services = services or self.services()
        app = FastAPI()
        app.include_router(
            self.plugin_api.build_router(lambda: services), prefix=self.PREFIX
        )
        return TestClient(app, raise_server_exceptions=False)

    def create(self, *, kind="coding", workdir=None, intent="do the work"):
        task = self.registry.create_task(
            conversation_key="conv-a",
            kind=kind,
            intent=intent,
            workdir=str(workdir) if workdir is not None else None,
            event_kind="task.created",
        )
        if task["state"] == "starting":
            task = self.registry.transition_task(
                task["id"], "queued", expected_version=task["version"]
            )
        return task

    def workdir(self, name="work"):
        path = self.root / name
        path.mkdir(parents=True, exist_ok=True)
        return path

    def attach_log(self, task, path):
        return self.registry.update_task(
            task["id"], expected_version=task["version"], log_path=str(path)
        )

    def test_task_body_has_a_workspace_label_not_a_workdir(self) -> None:
        workdir = self.workdir("rocket-project")
        task = self.create(workdir=workdir)

        body = self.client().get(self.PREFIX + "/tasks/" + task["id"]).json()

        self.assertNotIn("workdir", body)
        self.assertEqual(body["workspace"], "rocket-project")
        self.assertNotIn(HOME, json.dumps(body))
        self.assertNotIn(str(self.root), json.dumps(body))

    def test_task_list_has_no_absolute_paths(self) -> None:
        self.create(workdir=self.workdir("listed"))
        body = self.client().get(self.PREFIX + "/tasks").json()

        self.assertEqual(body["count"], 1)
        self.assertNotIn(str(self.root), json.dumps(body))
        self.assertNotIn("workdir", body["tasks"][0])

    def test_error_detail_and_summary_are_scrubbed(self) -> None:
        task = self.create()
        self.registry.update_task(
            task["id"],
            expected_version=task["version"],
            error_code="executor_config",
            error_detail="cannot open %s/secret/report.md" % HOME,
            result_summary="wrote %s/work/out.md" % HOME,
        )

        body = self.client().get(self.PREFIX + "/tasks/" + task["id"]).json()

        self.assertNotIn(HOME, json.dumps(body))
        self.assertIn("report.md", body["error_detail"])

    def test_approval_body_is_an_allowlist_without_the_payload(self) -> None:
        task = self.create()
        record = {
            "decision": None,
            "action": "publish the draft",
            "account": "ops@example.com",
            "audience": "public",
            "reason": "publishing leaves the Mac",
            "fingerprint": "f" * 64,
            "payload": {"api_key": "super-secret-payload", "path": "/Users/user/private"},
            "categories": ["publish"],
            "publication": True,
            "requires_approval": True,
            "note": "checked",
            "arbitrary_internal_column": "must-not-leak",
        }
        self.registry.update_task(
            task["id"],
            expected_version=task["version"],
            approval_json=json.dumps(record),
        )

        body = self.client().get(self.PREFIX + "/tasks/" + task["id"]).json()

        approval = body["approval"]
        self.assertEqual(approval["action"], "publish the draft")
        self.assertEqual(approval["decision"], None)
        self.assertNotIn("payload", approval)
        self.assertNotIn("arbitrary_internal_column", approval)
        spelled = json.dumps(body)
        self.assertNotIn("super-secret-payload", spelled)
        self.assertNotIn("/Users/user/private", spelled)

    def test_log_tail_redacts_credentials_and_home_paths(self) -> None:
        task = self.create()
        path = self.log_root / task["id"] / "task.log"
        path.parent.mkdir()
        path.write_text(
            "\n".join(
                (
                    "NEO_TASK_TOKEN=worker-secret-token",
                    "Authorization: Bearer bearer-secret-token",
                    "key sk-proj-abcdefghijklmnopqrstuvwxyz012345",
                    "-----BEGIN RSA PRIVATE KEY-----",
                    "MIIEowIBAAKCAQEA",
                    "-----END RSA PRIVATE KEY-----",
                    "cwd=%s/code/work" % HOME,
                )
            )
            + "\n",
            encoding="utf-8",
        )
        task = self.attach_log(task, path)

        body = self.client().get(
            self.PREFIX + "/tasks/" + task["id"] + "/log", params={"tail": 100}
        ).json()
        spelled = json.dumps(body)

        for secret in (
            "worker-secret-token",
            "bearer-secret-token",
            "sk-proj-abcdefghijklmnopqrstuvwxyz012345",
            "MIIEowIBAAKCAQEA",
            HOME,
        ):
            self.assertNotIn(secret, spelled)
        self.assertIn("~", spelled)

    def test_log_tail_is_capped_after_redaction(self) -> None:
        task = self.create()
        path = self.log_root / task["id"] / "task.log"
        path.parent.mkdir()
        path.write_text(
            ("NEO_TASK_TOKEN=%s\n" % ("x" * 4000)) * 20, encoding="utf-8"
        )
        task = self.attach_log(task, path)
        services = self.services(log_max_bytes=2048)

        body = self.client(services).get(
            self.PREFIX + "/tasks/" + task["id"] + "/log", params={"tail": 1000}
        ).json()

        rendered = "\n".join(body["lines"])
        self.assertLessEqual(len(rendered.encode("utf-8")), 2048)
        self.assertNotIn("x" * 100, rendered)

    def test_log_tail_refuses_another_tasks_log(self) -> None:
        first = self.create()
        second = self.create()
        second_path = self.log_root / ("%s.log" % second["id"])
        second_path.write_text("second task only\n", encoding="utf-8")
        self.attach_log(second, second_path)
        self.attach_log(first, second_path)

        response = self.client().get(
            self.PREFIX + "/tasks/" + first["id"] + "/log"
        )

        self.assertEqual(response.status_code, 404)
        self.assertNotIn("second task only", response.text)

    def test_log_tail_refuses_a_foreign_file_name(self) -> None:
        task = self.create()
        foreign = self.log_root / "some-other-name.log"
        foreign.write_text("not mine\n", encoding="utf-8")
        self.attach_log(task, foreign)

        response = self.client().get(self.PREFIX + "/tasks/" + task["id"] + "/log")

        self.assertEqual(response.status_code, 404)
        self.assertNotIn(str(self.root), response.text)

    def test_artifact_refuses_adversarial_names(self) -> None:
        workdir = self.workdir()
        (workdir / "ok.md").write_text("ok", encoding="utf-8")
        (self.root / "outside.md").write_text("outside", encoding="utf-8")
        task = self.create(workdir=workdir)
        url = self.PREFIX + "/tasks/" + task["id"] + "/artifact"
        client = self.client()

        for name in (
            "../outside.md",
            "..%2F..%2Fetc%2Fpasswd",
            "/etc/passwd",
            "..\\outside.md",
            "nul\x00.md",
            "file:///etc/passwd",
            "C:\\Windows\\win.ini",
        ):
            with self.subTest(name=name):
                response = client.get(url, params={"name": name})
                self.assertIn(response.status_code, (400, 404, 415))
                self.assertNotIn("outside", response.text)
                self.assertNotIn(str(self.root), response.text)

    def test_artifact_refuses_a_symlinked_file(self) -> None:
        workdir = self.workdir()
        secret = self.root / "secret.md"
        secret.write_text("secret", encoding="utf-8")
        (workdir / "link.md").symlink_to(secret)
        (workdir / "dir-link").symlink_to(self.root, target_is_directory=True)
        task = self.create(workdir=workdir)
        url = self.PREFIX + "/tasks/" + task["id"] + "/artifact"
        client = self.client()

        for name in ("link.md", "dir-link/secret.md"):
            with self.subTest(name=name):
                response = client.get(url, params={"name": name})
                self.assertIn(response.status_code, (400, 404))
                self.assertNotIn("secret", response.text)


# -- worker bridge containment ------------------------------------------------


class PreviewContainmentTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tempdir.cleanup)
        self.root = Path(self._tempdir.name)
        self.registry = Registry(path=self.root / "neo.db", max_parallel=8)
        self.service = TaskService(self.registry)
        self.tokens = WorkerTokenStore(self.registry)
        self.bridge = WorkerBridge(
            self.service,
            self.tokens,
            preview_root=self.root / "previews",
            preview_max_bytes=1024,
        )

    def task(self, workdir=None):
        return self.service.create_task(
            conversation_key="conv-a",
            kind="coding",
            intent="produce a preview",
            workdir=str(workdir or self.root / "work"),
        )

    def test_preview_refuses_a_symlinked_source(self) -> None:
        workdir = self.root / "work"
        workdir.mkdir()
        secret = self.root / "secret.png"
        secret.write_bytes(PNG)
        (workdir / "shot.png").symlink_to(secret)
        task = self.task(workdir)
        token = self.tokens.issue(task["id"])

        with self.assertRaises(Exception) as caught:
            self.bridge.neo_task_publish_preview(task["id"], token, "shot.png")

        self.assertNotIsInstance(caught.exception, AssertionError)

    def test_preview_refuses_an_oversize_source(self) -> None:
        workdir = self.root / "work"
        workdir.mkdir()
        (workdir / "big.png").write_bytes(PNG + b"x" * 4096)
        task = self.task(workdir)
        token = self.tokens.issue(task["id"])

        with self.assertRaises(Exception) as caught:
            self.bridge.neo_task_publish_preview(task["id"], token, "big.png")

        self.assertIn("limit", str(caught.exception))

    def test_preview_refuses_a_source_outside_the_workdir(self) -> None:
        workdir = self.root / "work"
        workdir.mkdir()
        outside = self.root / "outside.png"
        outside.write_bytes(PNG)
        task = self.task(workdir)
        token = self.tokens.issue(task["id"])

        with self.assertRaises(Exception):
            self.bridge.neo_task_publish_preview(task["id"], token, str(outside))


# -- spawn and install safety -------------------------------------------------


class SpawnAndInstallSafetyTests(unittest.TestCase):
    def test_spawn_is_argv_only_and_uses_the_worker_environment(self) -> None:
        source = (REPO_ROOT / "workers.py").read_text(encoding="utf-8")
        self.assertIn("shell=False", source)
        self.assertNotIn("shell=True", source)
        self.assertIn("worker_environment(", source)

    def test_follow_up_spawn_is_argv_only_and_uses_the_worker_environment(self) -> None:
        source = (REPO_ROOT / "conversation.py").read_text(encoding="utf-8")
        self.assertIn("shell=False", source)
        self.assertNotIn("shell=True", source)
        self.assertIn("ProcessSupervisor", source)

    def test_install_request_is_pinned_and_flag_free(self) -> None:
        from capabilities import (
            ConsentRequiredError,
            CapabilityDiscovery,
            build_install_request,
            validate_capability,
        )

        capability = validate_capability(
            {
                "name": "blender",
                "repo": "https://github.com/example/blender.git",
                "sha": "9b11ffeda84b077bea9de52d22733c7a41ca581a",
                "tier": "official",
            }
        )

        request = build_install_request(capability)

        self.assertNotIn("https://github.com/example/blender.git", request.argv)
        self.assertIn("--ref", request.argv)
        self.assertEqual(request.argv[request.argv.index("--ref") + 1], capability.sha)
        for flag in ("--force", "-f", "--allow-removed"):
            with self.subTest(flag=flag):
                with self.assertRaises(Exception):
                    build_install_request(capability, extra_args=(flag,))

        discovery = CapabilityDiscovery()
        with self.assertRaises(ConsentRequiredError):
            discovery.install_capability(capability)

    def test_browser_approval_never_publishes(self) -> None:
        from browser import BrowserSlice

        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            registry = Registry(path=root / "neo.db", max_parallel=8)
            service = TaskService(registry)
            launches = []
            runtime = SimpleNamespace(registry=registry, service=service)
            slice_ = BrowserSlice(
                runtime=runtime, launcher=lambda *a, **k: launches.append(a)
            )
            task = service.create_task(
                conversation_key="conv-a",
                kind="browser",
                intent="navigate and read",
                workdir=str(runtime.workspace if False else root),
            )
            service.transition(task["id"], "running", expected_version=task["version"])
            slice_.request_approval(
                registry.get_task(task["id"]),
                reason="publishing leaves the Mac",
                action="publish the draft",
                account="ops@example.com",
                audience="public",
            )

            outcome = slice_.record_approval(task["id"], "approve")

            self.assertFalse(outcome["publication_executed"])
            self.assertEqual(launches, [])


if __name__ == "__main__":
    unittest.main()
