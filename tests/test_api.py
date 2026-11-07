"""Task 14 — dashboard backend API.

The dashboard is a thin client over NEO's durable state. This suite pins the
boundary:

* every route is JSON-only; previews and artifacts travel as base64 text;
* task ids are validated strictly, so a malformed or traversing id is *not
  found* rather than a probe;
* the detail body is an allowlist, so internal fields such as a worker token
  hash can never ride along;
* list filters, cancel and approval are idempotent;
* retry creates a fresh child task and never reopens a terminal one;
* capability reads never install and consent is bound to one exact name + SHA;
* blocking SQLite and file work leaves the event loop.

Sections 4.4-4.7 of the implementation plan.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import stat
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# The backend module lives in ``dashboard/`` because Hermes loads it from there
# as a standalone module; it is imported here as ``plugin_api``, so its own
# directory must be importable too.
DASHBOARD_DIR = REPO_ROOT / "dashboard"
if str(DASHBOARD_DIR) not in sys.path:
    sys.path.insert(0, str(DASHBOARD_DIR))

try:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
except ModuleNotFoundError as exc:  # pragma: no cover - system python lacks fastapi
    raise unittest.SkipTest("FastAPI/httpx are required for the API suite: %s" % exc)

import plugin_api  # noqa: E402
from capabilities import (  # noqa: E402
    CapabilityDiscovery,
    InMemoryConsentStore,
)
from artifacts import completion_evidence  # noqa: E402
from events import TaskService  # noqa: E402
from presentation import PresentationService, TaskPresenter  # noqa: E402
from previews import PreviewService  # noqa: E402
from registry import Registry  # noqa: E402

PREFIX = "/api/plugins/neo"
BLENDER_SHA = "9b11ffeda84b077bea9de52d22733c7a41ca581a"
OTHER_SHA = "0f1e2d3c4b5a69788796a5b4c3d2e1f001122334"
PNG = b"\x89PNG\r\n\x1a\n" + b"pixels" * 4


class FakeSupervisor:
    """Cancel by asking the real service; records every call."""

    def __init__(self, service):
        self.service = service
        self.calls = []

    def cancel(self, task_id):
        self.calls.append(task_id)
        task = self.service.get_task(task_id)
        if task["state"] in ("completed", "failed", "cancelled"):
            return task
        return self.service.transition(
            task_id, "cancelled", expected_version=task["version"]
        )


class BrokenSupervisor:
    def __init__(self):
        self.calls = []

    def cancel(self, task_id):
        self.calls.append(task_id)
        raise RuntimeError("active task has no supervised process")


class FakeRetry:
    """Create a child task the way a retry service should."""

    def __init__(self, service):
        self.service = service
        self.calls = []

    def __call__(self, task):
        self.calls.append(task["id"])
        child = self.service.create_task(
            conversation_key=task["conversation_key"],
            kind=task["kind"],
            intent=task["intent"],
            workdir=task.get("workdir"),
        )
        return self.service.registry.update_task(
            child["id"], expected_version=child["version"], parent_id=task["id"]
        )


class FakeInstaller:
    def __init__(self):
        self.calls = []
        self.sha = None

    def install(self, request):
        self.calls.append(request)
        return {"ok": True, "name": request.name, "sha": request.sha}

    def installed_sha(self, name):
        return self.sha or self.calls[-1].sha


class FakeRunner:
    def __init__(self, stdout="", returncode=0):
        self.calls = []
        self.stdout = stdout
        self.returncode = returncode

    def __call__(self, argv):
        self.calls.append(list(argv))
        return SimpleNamespace(returncode=self.returncode, stdout=self.stdout)


class APITestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tempdir.cleanup)
        self.root = Path(self._tempdir.name)
        self.log_root = self.root / "logs"
        self.log_root.mkdir()
        self.preview_root = self.root / "previews"
        self.preview_root.mkdir()
        self.registry = Registry(path=self.root / "neo.db", max_parallel=8)
        self.service = TaskService(self.registry)
        self.preview = PreviewService(self.registry, preview_root=self.preview_root)

    # -- fixtures --------------------------------------------------------

    def services(self, **overrides):
        base = dict(
            registry=self.registry,
            task_service=self.service,
            preview_service=self.preview,
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

    def client(self, services=None, factory=None):
        services = services if services is not None else self.services()
        factory = factory or (lambda: services)
        app = FastAPI()
        app.include_router(plugin_api.build_router(factory), prefix=PREFIX)
        return TestClient(app, raise_server_exceptions=False)

    def create(
        self,
        conversation_key: str = "conv-a",
        kind: str = "coding",
        workdir=None,
        intent: str = "do the work",
    ) -> dict:
        """Create a task the way NEO does, parked in ``queued``.

        ``Registry.create_task`` starts a task immediately (``starting``) when
        capacity allows, while every case below drives a task from the queue.
        The fixture therefore records the creation event and then parks the task
        in ``queued`` without a second event, so ``task.created`` stays the only
        event a fresh task owns.
        """
        task = self.registry.create_task(
            conversation_key=conversation_key,
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

    def drive(self, task: dict, *states: str) -> dict:
        current = task
        for state in states:
            current = self.service.transition(
                current["id"], state, expected_version=current["version"]
            )
        return current

    def needs_approval(self, **kwargs) -> dict:
        task = self.create(**kwargs)
        return self.drive(task, "starting", "running", "needs_approval")

    def workdir(self, name: str = "work") -> Path:
        path = self.root / name
        path.mkdir(parents=True, exist_ok=True)
        return path

    def assert_json(self, response):
        self.assertIn("application/json", response.headers.get("content-type", ""))
        return response.json()


class HealthAndManifestTests(APITestCase):
    def test_health_is_concise_and_names_no_path(self) -> None:
        response = self.client().get(PREFIX + "/health")
        body = self.assert_json(response)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(body["status"], "ok")
        self.assertEqual(body["plugin"], "neo")
        self.assertEqual(body["version"], plugin_api.PLUGIN_VERSION)
        self.assertNotIn("/", json.dumps(body))

    def test_health_never_resolves_services(self) -> None:
        calls = []

        def explode():
            calls.append(True)
            raise RuntimeError("production services must stay lazy")

        response = self.client(factory=explode).get(PREFIX + "/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(calls, [])

    def test_manifest_matches_the_hermes_dashboard_shape(self) -> None:
        manifest_path = REPO_ROOT / "dashboard" / "manifest.json"
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(data["name"], "neo")
        self.assertEqual(data["label"], "NEO")
        self.assertEqual(data["version"], "0.1.0")
        self.assertEqual(data["entry"], "dist/index.js")
        self.assertEqual(data["api"], "plugin_api.py")
        self.assertEqual(data["tab"]["path"], "/neo")
        self.assertTrue((REPO_ROOT / "dashboard" / "plugin_api.py").is_file())

    def test_router_registers_every_planned_route(self) -> None:
        router = plugin_api.build_router(lambda: self.services())
        found = set()
        for route in router.routes:
            for method in getattr(route, "methods", ()):
                found.add((method, route.path))
        expected = {
            ("GET", "/health"),
            ("GET", "/tasks"),
            ("GET", "/tasks/{task_id}"),
            ("POST", "/tasks/{task_id}/cancel"),
            ("POST", "/tasks/{task_id}/retry"),
            ("GET", "/tasks/{task_id}/events"),
            ("GET", "/tasks/{task_id}/preview"),
            ("GET", "/tasks/{task_id}/artifact"),
            ("GET", "/tasks/{task_id}/log"),
            ("GET", "/tasks/{task_id}/capabilities"),
            ("POST", "/tasks/{task_id}/approve"),
            ("POST", "/tasks/{task_id}/present"),
            ("GET", "/capabilities"),
            ("POST", "/capabilities/consent"),
            ("POST", "/capabilities/install"),
            ("POST", "/capabilities/enable"),
        }
        self.assertEqual(found, expected)


class TaskReadTests(APITestCase):
    def test_list_filters_by_state_kind_and_since(self) -> None:
        first = self.create(kind="coding")
        second = self.create(kind="blender")
        third = self.drive(self.create(kind="coding"), "starting", "running")
        client = self.client()

        queued = self.assert_json(client.get(PREFIX + "/tasks", params={"state": "queued"}))
        self.assertEqual(
            sorted(task["id"] for task in queued["tasks"]),
            sorted([first["id"], second["id"]]),
        )
        blender = self.assert_json(
            client.get(PREFIX + "/tasks", params={"kind": "blender"})
        )
        self.assertEqual([task["id"] for task in blender["tasks"]], [second["id"]])
        both = self.assert_json(
            client.get(PREFIX + "/tasks", params={"kind": "blender", "state": "running"})
        )
        self.assertEqual(both["tasks"], [])
        since = self.assert_json(
            client.get(PREFIX + "/tasks", params={"since": third["created_at"]})
        )
        self.assertEqual([task["id"] for task in since["tasks"]], [third["id"]])

    def test_list_rejects_unknown_filters(self) -> None:
        client = self.client()
        self.assertEqual(
            client.get(PREFIX + "/tasks", params={"state": "sleeping"}).status_code, 422
        )
        self.assertEqual(
            client.get(PREFIX + "/tasks", params={"since": "yesterday"}).status_code, 422
        )
        self.assertEqual(
            client.get(PREFIX + "/tasks", params={"kind": "../etc"}).status_code, 422
        )

    def test_malformed_and_traversing_ids_are_not_found(self) -> None:
        task = self.create()
        client = self.client()
        bad_ids = [
            "..",
            "..%2F..%2Fetc%2Fpasswd",
            "a" * 65,
            "with space",
            "semi;colon",
            "dot.dot",
            task["id"] + "x",
        ]
        for bad in bad_ids:
            with self.subTest(bad=bad):
                response = client.get(PREFIX + "/tasks/" + bad)
                self.assertEqual(response.status_code, 404)
                self.assertNotIn(str(self.root), json.dumps(response.json()))

    def test_detail_is_an_allowlist_and_hides_token_material(self) -> None:
        task = self.needs_approval()
        self.registry.update_task(
            task["id"],
            expected_version=task["version"],
            expected_json=json.dumps(["out.md"]),
            artifacts_json=json.dumps(["out.md"]),
            approval_json=json.dumps({"reason": "publish", "decision": None}),
        )
        leaky = dict(self.registry.get_task(task["id"]))
        leaky.update(
            {
                "token_hash": "deadbeef",
                "worker_token": "super-secret",
                "session_token": "nope",
            }
        )

        class LeakyRegistry:
            def get_task(self, task_id):
                return dict(leaky)

        services = self.services(registry=LeakyRegistry())
        body = self.assert_json(self.client(services).get(PREFIX + "/tasks/" + task["id"]))
        self.assertEqual(body["id"], task["id"])
        self.assertEqual(body["expected"], ["out.md"])
        self.assertEqual(body["artifacts"], ["out.md"])
        self.assertEqual(body["approval"]["reason"], "publish")
        spelled = json.dumps(body)
        for secret in ("token_hash", "worker_token", "session_token", "deadbeef", "super-secret"):
            self.assertNotIn(secret, spelled)

    def test_detail_decodes_malformed_json_without_crashing(self) -> None:
        task = self.create()
        self.registry.update_task(
            task["id"], expected_version=task["version"], expected_json="{not json"
        )
        body = self.assert_json(self.client().get(PREFIX + "/tasks/" + task["id"]))
        self.assertEqual(body["expected"], [])

    def test_unknown_task_is_not_found_everywhere(self) -> None:
        client = self.client()
        for suffix in ("", "/events", "/preview", "/artifact?name=out.md", "/log"):
            with self.subTest(suffix=suffix):
                response = client.get(PREFIX + "/tasks/does-not-exist" + suffix)
                self.assertEqual(response.status_code, 404)


class PreviewAndArtifactTests(APITestCase):
    def publish(self, task: dict, data: bytes = PNG) -> Path:
        directory = self.preview_root / task["id"]
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "rev-1.png"
        path.write_bytes(data)
        self.registry.update_task(
            task["id"],
            expected_version=task["version"],
            preview_path=str(path),
            preview_rev=1,
        )
        return path

    def test_preview_is_json_base64(self) -> None:
        task = self.create()
        self.publish(task)
        response = self.client().get(PREFIX + "/tasks/" + task["id"] + "/preview")
        body = self.assert_json(response)
        self.assertTrue(body["available"])
        self.assertEqual(body["mime"], "image/png")
        self.assertEqual(base64.b64decode(body["data_base64"]), PNG)

    def test_preview_unavailable_when_none_published(self) -> None:
        task = self.create()
        body = self.assert_json(self.client().get(PREFIX + "/tasks/" + task["id"] + "/preview"))
        self.assertFalse(body["available"])
        self.assertIsNone(body["data_base64"])

    def test_artifact_returns_verified_bytes_as_base64(self) -> None:
        workdir = self.workdir()
        (workdir / "out").mkdir()
        payload = b"# report\n"
        (workdir / "out" / "report.md").write_bytes(payload)
        task = self.create(workdir=workdir)
        response = self.client().get(
            PREFIX + "/tasks/" + task["id"] + "/artifact",
            params={"name": "out/report.md"},
        )
        body = self.assert_json(response)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(body["name"], "out/report.md")
        self.assertEqual(body["mime"], "text/markdown")
        self.assertEqual(base64.b64decode(body["data_base64"]), payload)

    def test_artifact_refuses_traversal_absolute_and_unsupported(self) -> None:
        workdir = self.workdir()
        (self.root / "secret.md").write_bytes(b"secret")
        (workdir / "payload.exe").write_bytes(b"MZ")
        task = self.create(workdir=workdir)
        client = self.client()
        url = PREFIX + "/tasks/" + task["id"] + "/artifact"
        cases = [
            ("../secret.md", 400),
            ("/etc/passwd", 400),
            ("out\\report.md", 400),
            ("payload.exe", 415),
            ("missing.md", 404),
        ]
        for name, status in cases:
            with self.subTest(name=name):
                response = client.get(url, params={"name": name})
                self.assertEqual(response.status_code, status)
                self.assertNotIn(str(self.root), response.text)

    def test_artifact_cannot_read_another_tasks_workdir(self) -> None:
        first = self.workdir("first")
        (first / "report.md").write_bytes(b"first task only")
        second = self.workdir("second")
        self.create(workdir=first)
        other = self.create(workdir=second)
        response = self.client().get(
            PREFIX + "/tasks/" + other["id"] + "/artifact",
            params={"name": "../first/report.md"},
        )
        self.assertEqual(response.status_code, 400)

    def test_artifact_oversize_is_refused(self) -> None:
        workdir = self.workdir()
        (workdir / "big.md").write_bytes(b"0123456789")
        task = self.create(workdir=workdir)
        services = self.services(artifact_max_bytes=8)
        response = self.client(services).get(
            PREFIX + "/tasks/" + task["id"] + "/artifact",
            params={"name": "big.md"},
        )
        self.assertEqual(response.status_code, 413)

    def test_artifact_requires_a_name(self) -> None:
        task = self.create(workdir=self.workdir())
        response = self.client().get(PREFIX + "/tasks/" + task["id"] + "/artifact")
        self.assertEqual(response.status_code, 422)


class LogTailTests(APITestCase):
    def attach_log(self, task: dict, path: Path) -> dict:
        return self.registry.update_task(
            task["id"], expected_version=task["version"], log_path=str(path)
        )

    def test_log_tail_is_bounded_and_relative(self) -> None:
        path = self.log_root / "task.log"
        path.write_bytes(b"one\ntwo\nthree\n")
        task = self.attach_log(self.create(), path)
        body = self.assert_json(
            self.client().get(PREFIX + "/tasks/" + task["id"] + "/log", params={"tail": 2})
        )
        self.assertEqual(body["lines"], ["two", "three"])
        self.assertEqual(body["tail"], 2)
        self.assertTrue(body["truncated"])
        self.assertNotIn(str(self.root), json.dumps(body))

    def test_log_tail_bounds_are_validated(self) -> None:
        path = self.log_root / "task.log"
        path.write_bytes(b"one\n")
        task = self.attach_log(self.create(), path)
        client = self.client()
        url = PREFIX + "/tasks/" + task["id"] + "/log"
        for tail in (0, 1001, -1, "many"):
            with self.subTest(tail=tail):
                self.assertEqual(client.get(url, params={"tail": tail}).status_code, 422)

    def test_log_outside_root_symlink_and_missing_are_refused(self) -> None:
        outside = self.root / "outside.log"
        outside.write_bytes(b"secret\n")
        inside = self.log_root / "inside.log"
        inside.write_bytes(b"fine\n")
        link = self.log_root / "link.log"
        link.symlink_to(inside)
        client = self.client()
        cases = [outside, link, self.log_root / "missing.log"]
        for path in cases:
            with self.subTest(path=path.name):
                task = self.attach_log(self.create(), path)
                response = client.get(PREFIX + "/tasks/" + task["id"] + "/log")
                self.assertEqual(response.status_code, 404)
                self.assertNotIn(str(self.root), response.text)

    def test_task_without_log_is_not_found(self) -> None:
        task = self.create()
        response = self.client().get(PREFIX + "/tasks/" + task["id"] + "/log")
        self.assertEqual(response.status_code, 404)


class CancelAndRetryTests(APITestCase):
    def test_cancel_terminal_task_is_idempotent(self) -> None:
        task = self.drive(self.create(), "cancelled")
        supervisor = FakeSupervisor(self.service)
        client = self.client(self.services(supervisor=supervisor))
        for _ in range(2):
            response = client.post(PREFIX + "/tasks/" + task["id"] + "/cancel")
            body = self.assert_json(response)
            self.assertEqual(response.status_code, 200)
            self.assertEqual(body["state"], "cancelled")
        self.assertEqual(supervisor.calls, [])

    def test_cancel_queued_task_delegates_to_supervisor(self) -> None:
        task = self.create()
        supervisor = FakeSupervisor(self.service)
        client = self.client(self.services(supervisor=supervisor))
        response = client.post(PREFIX + "/tasks/" + task["id"] + "/cancel")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["state"], "cancelled")
        self.assertEqual(supervisor.calls, [task["id"]])
        again = client.post(PREFIX + "/tasks/" + task["id"] + "/cancel")
        self.assertEqual(again.status_code, 200)
        self.assertEqual(supervisor.calls, [task["id"]])

    def test_cancel_without_supervisor_is_a_clean_conflict(self) -> None:
        task = self.create()
        response = self.client().post(PREFIX + "/tasks/" + task["id"] + "/cancel")
        self.assertEqual(response.status_code, 409)

    def test_cancel_supervisor_failure_is_not_a_500(self) -> None:
        task = self.drive(self.create(), "starting", "running")
        response = self.client(
            self.services(supervisor=BrokenSupervisor())
        ).post(PREFIX + "/tasks/" + task["id"] + "/cancel")
        self.assertEqual(response.status_code, 409)

    def test_retry_creates_one_child_and_repeats_are_idempotent(self) -> None:
        task = self.drive(self.create(), "cancelled")
        retry = FakeRetry(self.service)
        client = self.client(self.services(retry=retry))
        first = client.post(PREFIX + "/tasks/" + task["id"] + "/retry")
        second = client.post(PREFIX + "/tasks/" + task["id"] + "/retry")
        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 200)
        child = first.json()
        self.assertEqual(child["parent_id"], task["id"])
        self.assertEqual(second.json()["id"], child["id"])
        self.assertEqual(retry.calls, [task["id"]])
        self.assertEqual(self.registry.get_task(task["id"])["state"], "cancelled")

    def test_retry_requires_terminal_state_and_callback(self) -> None:
        running = self.drive(self.create(), "starting", "running")
        retry = FakeRetry(self.service)
        client = self.client(self.services(retry=retry))
        self.assertEqual(
            client.post(PREFIX + "/tasks/" + running["id"] + "/retry").status_code, 409
        )
        self.assertEqual(retry.calls, [])

        terminal = self.drive(self.create(), "cancelled")
        no_retry = self.client(self.services(retry=None))
        self.assertEqual(
            no_retry.post(PREFIX + "/tasks/" + terminal["id"] + "/retry").status_code, 409
        )


class ApprovalTests(APITestCase):
    def approve(self, client, task, payload):
        return client.post(PREFIX + "/tasks/" + task["id"] + "/approve", json=payload)

    def test_approve_runs_and_records_the_decision(self) -> None:
        task = self.needs_approval()
        client = self.client()
        response = self.approve(
            client, task, {"decision": "approve", "note": "publish the page"}
        )
        body = self.assert_json(response)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(body["state"], "running")
        self.assertEqual(body["approval"]["decision"], "approve")
        self.assertEqual(body["approval"]["note"], "publish the page")
        stored = self.registry.get_task(task["id"])
        self.assertIsNotNone(stored["approval_json"])

    def test_repeated_same_decision_is_idempotent(self) -> None:
        task = self.needs_approval()
        client = self.client()
        first = self.approve(client, task, {"decision": "approve"}).json()
        second = self.approve(client, task, {"decision": "approve"})
        self.assertEqual(second.status_code, 200)
        self.assertEqual(second.json()["id"], task["id"])
        self.assertEqual(second.json()["version"], first["version"])
        self.assertEqual(second.json()["state"], "running")

    def test_deny_cancels_and_repeats(self) -> None:
        task = self.needs_approval()
        client = self.client()
        first = self.approve(client, task, {"decision": "deny", "note": "no"})
        self.assertEqual(first.json()["state"], "cancelled")
        again = self.approve(client, task, {"decision": "deny", "note": "no"})
        self.assertEqual(again.status_code, 200)
        self.assertEqual(again.json()["version"], first.json()["version"])

    def test_malformed_decision_is_422(self) -> None:
        task = self.needs_approval()
        client = self.client()
        for payload in ({}, {"decision": "maybe"}, {"decision": 1}, [], "approve"):
            with self.subTest(payload=payload):
                self.assertEqual(self.approve(client, task, payload).status_code, 422)
        self.assertEqual(self.registry.get_task(task["id"])["state"], "needs_approval")

    def test_approve_outside_needs_approval_is_a_conflict(self) -> None:
        running = self.drive(self.create(), "starting", "running")
        response = self.approve(self.client(), running, {"decision": "approve"})
        self.assertEqual(response.status_code, 409)


class EventTests(APITestCase):
    def test_events_are_scoped_to_the_selected_task(self) -> None:
        first = self.create(conversation_key="one")
        second = self.create(conversation_key="two")
        client = self.client()
        body = self.assert_json(client.get(PREFIX + "/tasks/" + first["id"] + "/events"))
        self.assertEqual(body["since"], 0)
        self.assertTrue(body["events"])
        self.assertEqual({event["task_id"] for event in body["events"]}, {first["id"]})
        self.assertEqual({event["kind"] for event in body["events"]}, {"task.created"})
        self.assertTrue(all(isinstance(event["payload"], dict) for event in body["events"]))
        self.assertNotIn(second["id"], json.dumps(body))

    def test_events_honour_the_cursor(self) -> None:
        task = self.create()
        client = self.client()
        body = self.assert_json(client.get(PREFIX + "/tasks/" + task["id"] + "/events"))
        cursor = max(event["id"] for event in body["events"])
        after = self.assert_json(
            client.get(PREFIX + "/tasks/" + task["id"] + "/events", params={"since": cursor})
        )
        self.assertEqual(after["events"], [])
        self.assertEqual(
            client.get(
                PREFIX + "/tasks/" + task["id"] + "/events", params={"since": -1}
            ).status_code,
            422,
        )


class CapabilityRouteTests(APITestCase):
    def discovery(self, runner=None, installer=None, store=None):
        return CapabilityDiscovery(
            run_command=runner,
            has_plugin=lambda name: name == "blender",
            installer=installer,
            consent_store=store or InMemoryConsentStore(),
        )

    def capability_body(self, sha: str = BLENDER_SHA) -> dict:
        return {
            "name": "blender",
            "repo": "https://github.com/NousResearch/hermes-official-plugins",
            "sha": sha,
            "tier": "official",
            "description": "Official Blender Lab MCP server.",
            "version": "0.4.1",
            "subdir": "blender",
        }

    def test_capability_get_is_read_only_until_q_is_given(self) -> None:
        runner = FakeRunner(stdout=json.dumps({"results": [self.capability_body()]}))
        installer = FakeInstaller()
        discovery = self.discovery(runner=runner, installer=installer)
        client = self.client(self.services(capabilities=discovery))

        installed = self.assert_json(client.get(PREFIX + "/capabilities"))
        self.assertEqual(runner.calls, [])
        self.assertEqual(installed["query"], "")
        self.assertTrue(installed["installed"])

        searched = self.assert_json(client.get(PREFIX + "/capabilities", params={"q": "blender"}))
        self.assertEqual(len(runner.calls), 1)
        self.assertIn("--json", runner.calls[0])
        self.assertEqual(searched["query"], "blender")
        self.assertEqual(searched["catalog"][0]["name"], "blender")
        self.assertEqual(installer.calls, [])

    def test_task_capabilities_list_installed_without_network(self) -> None:
        task = self.create(kind="blender")
        runner = FakeRunner()
        discovery = self.discovery(runner=runner)
        body = self.assert_json(
            self.client(self.services(capabilities=discovery)).get(
                PREFIX + "/tasks/" + task["id"] + "/capabilities"
            )
        )
        self.assertEqual(body["task_id"], task["id"])
        self.assertEqual(body["kind"], "blender")
        self.assertEqual(runner.calls, [])

    def test_consent_is_bound_to_the_exact_name_and_sha(self) -> None:
        installer = FakeInstaller()
        discovery = self.discovery(installer=installer)
        client = self.client(self.services(capabilities=discovery))

        consent = self.assert_json(
            client.post(PREFIX + "/capabilities/consent", json=self.capability_body())
        )
        self.assertEqual(consent["capability"], "blender")
        self.assertEqual(consent["sha"], BLENDER_SHA)
        self.assertEqual(installer.calls, [])

        mismatch = client.post(
            PREFIX + "/capabilities/install", json=self.capability_body(OTHER_SHA)
        )
        self.assertEqual(mismatch.status_code, 409)
        self.assertEqual(installer.calls, [])

        installed = self.assert_json(
            client.post(PREFIX + "/capabilities/install", json=self.capability_body())
        )
        self.assertTrue(installed["installed"])
        self.assertEqual(len(installer.calls), 1)
        self.assertEqual(installer.calls[0].sha, BLENDER_SHA)

    def test_malformed_capability_is_422(self) -> None:
        client = self.client(self.services(capabilities=self.discovery()))
        for body in (
            {},
            {"name": "blender", "sha": "abc", "repo": "https://example.com/x", "tier": "official"},
            {"name": "-x", "sha": BLENDER_SHA, "repo": "https://example.com/x", "tier": "official"},
            {"name": "blender", "sha": BLENDER_SHA, "repo": "/tmp/repo", "tier": "official"},
        ):
            with self.subTest(body=body):
                self.assertEqual(
                    client.post(PREFIX + "/capabilities/consent", json=body).status_code, 422
                )

    def test_enable_delegates_or_refuses_honestly(self) -> None:
        client = self.client(self.services(capabilities=self.discovery()))
        self.assertEqual(
            client.post(PREFIX + "/capabilities/enable", json={"name": "blender"}).status_code,
            409,
        )

        class EnableOnly:
            def __init__(self):
                self.calls = []

            def enable_capability(self, name):
                self.calls.append(name)
                return {"name": name, "enabled": True}

        enable = EnableOnly()
        body = self.assert_json(
            self.client(self.services(capabilities=enable)).post(
                PREFIX + "/capabilities/enable", json={"name": "blender"}
            )
        )
        self.assertTrue(body["enabled"])
        self.assertEqual(enable.calls, ["blender"])

    def test_capability_routes_refuse_a_missing_service(self) -> None:
        client = self.client()
        self.assertEqual(client.get(PREFIX + "/capabilities").status_code, 409)
        self.assertEqual(
            client.post(PREFIX + "/capabilities/consent", json=self.capability_body()).status_code,
            409,
        )
        self.assertEqual(
            client.post(PREFIX + "/capabilities/install", json=self.capability_body()).status_code,
            409,
        )


class PresentRouteTests(APITestCase):
    """POST /tasks/{id}/present is idempotent, conversation-bound and threaded."""

    def setUp(self) -> None:
        super().setUp()
        self.opened = []
        self.presentation = PresentationService(
            self.opened.append, ledger=self.registry
        )
        self.presenter = TaskPresenter(self.registry, self.presentation)
        self.present_root = self.workdir("present")
        (self.present_root / "report.md").write_text("# done", encoding="utf-8")

    def completed(self, conversation_key: str = "conv-a") -> dict:
        task = self.create(conversation_key=conversation_key, workdir=self.present_root)
        running = self.drive(task, "starting", "running")
        evidence = completion_evidence(self.present_root, ["report.md"])
        return self.service.transition(
            running["id"],
            "completed",
            expected_version=running["version"],
            evidence=evidence,
        )

    def client_with_presenter(self, presenter=None):
        presenter = presenter or self.presenter
        return self.client(self.services(presenter=presenter))

    def present(self, client, task_id: str, payload=None):
        url = PREFIX + "/tasks/" + task_id + "/present"
        if payload is None:
            return client.post(url)
        return client.post(url, json=payload)

    def test_present_is_idempotent_and_opens_once(self) -> None:
        task = self.completed()
        client = self.client_with_presenter()
        first = self.present(client, task["id"])
        second = self.present(client, task["id"])
        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 200)
        body = self.assert_json(first)
        self.assertTrue(body["presented"])
        self.assertFalse(body["skipped"])
        again = self.assert_json(second)
        self.assertFalse(again["presented"])
        self.assertTrue(again["skipped"])
        self.assertIn("already presented", again["reason"])
        self.assertEqual(len(self.opened), 1)
        self.assertEqual(self.opened[0], str((self.present_root / "report.md").resolve()))

    def test_present_binds_the_task_to_a_conversation(self) -> None:
        task = self.completed(conversation_key="conv-a")
        client = self.client_with_presenter()
        refused = self.present(client, task["id"], {"conversation_key": "conv-b"})
        self.assertEqual(refused.status_code, 404)
        self.assertEqual(self.opened, [])
        allowed = self.present(client, task["id"], {"conversation_key": "conv-a"})
        self.assertEqual(allowed.status_code, 200)
        self.assertTrue(self.assert_json(allowed)["presented"])

    def test_present_unknown_and_malformed_ids_are_not_found(self) -> None:
        client = self.client_with_presenter()
        for task_id in ("does-not-exist", "..", "with space", "a" * 65):
            with self.subTest(task_id=task_id):
                response = client.post(PREFIX + "/tasks/" + task_id + "/present")
                self.assertEqual(response.status_code, 404)
                self.assertNotIn(str(self.root), response.text)
        self.assertEqual(self.opened, [])

    def test_present_incomplete_task_is_skipped_not_opened(self) -> None:
        task = self.create(workdir=self.present_root)
        response = self.present(self.client_with_presenter(), task["id"])
        self.assertEqual(response.status_code, 200)
        body = self.assert_json(response)
        self.assertFalse(body["presented"])
        self.assertTrue(body["skipped"])
        self.assertEqual(self.opened, [])

    def test_present_refuses_a_missing_presenter(self) -> None:
        task = self.completed()
        response = self.present(self.client(), task["id"])
        self.assertEqual(response.status_code, 409)
        self.assertEqual(self.opened, [])

    def test_present_callback_runs_off_the_event_loop(self) -> None:
        observations = []

        class ProbePresenter:
            def present_task(self, task_id, conversation_key=None):
                try:
                    asyncio.get_running_loop()
                    observations.append(True)
                except RuntimeError:
                    observations.append(False)
                return {
                    "task_id": task_id,
                    "version": 1,
                    "presented": False,
                    "skipped": True,
                    "path": None,
                    "reason": "probe",
                }

        task = self.completed()
        response = self.present(self.client_with_presenter(ProbePresenter()), task["id"])
        self.assertEqual(response.status_code, 200)
        self.assertEqual(observations, [False])


class ThreadingTests(APITestCase):
    def test_registry_reads_never_run_on_the_event_loop(self) -> None:
        observations = []

        class ProbeRegistry:
            def __init__(self, inner):
                self.inner = inner

            def get_task(self, task_id):
                try:
                    asyncio.get_running_loop()
                    observations.append(True)
                except RuntimeError:
                    observations.append(False)
                return self.inner.get_task(task_id)

            def __getattr__(self, name):
                return getattr(self.inner, name)

        task = self.create()
        probe = ProbeRegistry(self.registry)
        services = self.services(registry=probe)
        body = self.assert_json(self.client(services).get(PREFIX + "/tasks/" + task["id"]))
        self.assertEqual(body["id"], task["id"])
        self.assertTrue(observations)
        self.assertFalse(any(observations))

    def test_source_uses_the_threadpool_bridge(self) -> None:
        source = (REPO_ROOT / "dashboard" / "plugin_api.py").read_text(encoding="utf-8")
        self.assertIn("run_in_threadpool", source)


if __name__ == "__main__":
    unittest.main()
