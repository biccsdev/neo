"""Task 19 — the Blender vertical slice.

These tests fix the Blender contract before any of it is trusted:

* Blender.app presence is a fact about the Mac, never a claim that a scene is
  ready. Version and path are reported; scene readiness stays ``False`` until a
  worker has actually connected and read the scene.
* Routing is only ``ready`` when the official ``blender`` plugin is installed
  *and* enabled. Otherwise the answer is ``needs_approval`` with the exact
  pinned name, version, SHA and the full disclosure — never a fake task.
* Installing requires an exact :class:`ConsentRecord` for ``blender`` at the
  pinned SHA, delegates to the injected ``CapabilityDiscovery`` installer and
  never emits ``--force``, ``--allow-removed`` or a repository URL.
* A prepared task owns ``blender:9876``, runs through ``hermes_worker`` with
  ``blender:blender`` and ``neo-worker`` preloaded, and carries a prompt that
  says what to build, what to verify, where to save the ``.blend`` and where
  approval is required. A second Blender task queues on the same resource key.
* Completion needs a real ``.blend`` (plus any expected screenshot) verified by
  the artifact layer. A claimed file that is missing, empty or not a Blender
  file never completes a task.
* A busy port 9876 that NEO does not own is reported honestly and is never
  retried into an existing instance.

Every test uses injected fakes. No test launches Blender, starts a process,
touches the network, or installs anything.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from blender import (  # noqa: E402
    BLENDER_APP_PATH,
    BLENDER_PORT,
    RESOURCE_KEY,
    BlenderCompletionError,
    BlenderNotReadyError,
    BlenderPortBusyError,
    BlenderProbe,
    BlenderSlice,
    build_worker_prompt,
    parse_blender_version,
)
from capabilities import (  # noqa: E402
    OFFICIAL_BLENDER,
    CapabilityDiscovery,
    ConsentRecord,
    ConsentRequiredError,
    InMemoryConsentStore,
)
from conversation import ConversationTools, NeoRuntime  # noqa: E402

BLENDER_SHA = "9b11ffeda84b077bea9de52d22733c7a41ca581a"
OTHER_SHA = "0f1e2d3c4b5a69788796a5b4c3d2e1f001122334"
BLENDER_VERSION = "0.4.1"


class ForbiddenRunner:
    """Any command at all is a failure in these tests."""

    def __call__(self, argv):
        raise AssertionError("no command may run: %r" % (list(argv),))


class PluginState:
    """Installed/enabled state for the injected plugin lookups."""

    def __init__(self, installed: bool = False, enabled: bool = False) -> None:
        self.installed = installed
        self.enabled = enabled

    def discovery(
        self,
        *,
        installer=None,
        store=None,
        runner=None,
    ) -> CapabilityDiscovery:
        return CapabilityDiscovery(
            run_command=runner or ForbiddenRunner(),
            has_plugin=lambda name: self.installed and name == "blender",
            is_plugin_enabled=lambda name: self.enabled,
            installer=installer,
            consent_store=store if store is not None else InMemoryConsentStore(),
        )


class FakeInstaller:
    """Records the install request and reports a post-install revision."""

    def __init__(self, sha: Optional[str] = BLENDER_SHA, after=None, raises=None) -> None:
        self.requests = []
        self.verified = []
        self.sha = sha
        self.after = after
        self.raises = raises

    def install(self, request):
        self.requests.append(request)
        if self.raises is not None:
            raise self.raises
        if self.after is not None:
            self.after()
        return None

    def installed_sha(self, name):
        self.verified.append(name)
        return self.sha


class FakeProbe:
    def __init__(self, present: bool = True, version: str = "5.1.2", path: str = BLENDER_APP_PATH):
        self.calls = []
        self.present = present
        self.version = version
        self.path = path

    def __call__(self, path):
        self.calls.append(path)
        return BlenderProbe(path=self.path, present=self.present, version=self.version)


class FakePort:
    def __init__(self, listening: bool = False) -> None:
        self.listening = listening
        self.calls = []

    def __call__(self, host, port):
        self.calls.append((host, port))
        return self.listening


def consent(sha: str = BLENDER_SHA, capability: str = "blender") -> ConsentRecord:
    return ConsentRecord(
        capability=capability,
        sha=sha,
        version=BLENDER_VERSION,
        granted_at="2026-11-07T00:00:00Z",
        source="test",
        nonce="nonce-0001",
    )


def blend_bytes() -> bytes:
    """A file that starts the way Blender's own saves start."""

    return b"BLENDER-v302RENDH" + b"\x00" * 48


class LaunchRecorder:
    def __init__(self) -> None:
        self.calls = []

    def __call__(self, task_id, argv, *, role="task"):
        self.calls.append({"task_id": task_id, "argv": list(argv), "role": role})
        return None

    def last(self) -> dict:
        if not self.calls:
            raise AssertionError("nothing was launched")
        return self.calls[-1]


def make_slice(
    *,
    discovery=None,
    runtime=None,
    probe=None,
    port=None,
    consent_store=None,
) -> BlenderSlice:
    return BlenderSlice(
        runtime=runtime,
        discovery=discovery,
        consent=consent_store,
        probe=probe or FakeProbe(),
        port_check=port or FakePort(False),
    )


# -- probing and readiness ----------------------------------------------------


class ProbeTests(unittest.TestCase):
    def test_default_app_path_is_the_macos_application_bundle(self) -> None:
        self.assertEqual(BLENDER_APP_PATH, "/Applications/Blender.app")

    def test_version_is_parsed_from_blender_output(self) -> None:
        text = "Blender 5.1.2\n\tbuild date: 2026-10-01\n"
        self.assertEqual(parse_blender_version(text), "5.1.2")

    def test_version_parser_refuses_text_without_a_version(self) -> None:
        self.assertEqual(parse_blender_version("no idea"), "")

    def test_missing_app_is_reported_as_absent(self) -> None:
        state = PluginState(installed=True, enabled=True)
        probe = FakeProbe(present=False, version="")
        slice_ = make_slice(discovery=state.discovery(), probe=probe)
        decision = slice_.readiness()
        self.assertFalse(decision["ready"])
        self.assertEqual(decision["status"], "needs_approval")
        self.assertFalse(decision["app"]["present"])
        self.assertIn("Blender.app", decision["reason"])

    def test_probe_reports_path_and_version_but_never_scene_readiness(self) -> None:
        state = PluginState(installed=True, enabled=True)
        slice_ = make_slice(discovery=state.discovery())
        decision = slice_.readiness()
        self.assertTrue(decision["ready"])
        self.assertEqual(decision["app"]["path"], BLENDER_APP_PATH)
        self.assertEqual(decision["app"]["version"], "5.1.2")
        self.assertFalse(decision["scene_ready"])
        self.assertIn("get_objects_summary", decision["scene_note"])


class ReadinessTests(unittest.TestCase):
    def _decision(self, installed: bool, enabled: bool) -> dict:
        state = PluginState(installed=installed, enabled=enabled)
        return make_slice(discovery=state.discovery()).readiness()

    def test_missing_plugin_needs_approval_with_the_pinned_disclosure(self) -> None:
        decision = self._decision(False, False)
        self.assertEqual(decision["status"], "needs_approval")
        self.assertFalse(decision["ready"])
        disclosure = decision["disclosure"]
        self.assertEqual(disclosure["name"], "blender")
        self.assertEqual(disclosure["version"], BLENDER_VERSION)
        self.assertEqual(disclosure["sha"], BLENDER_SHA)
        self.assertTrue(disclosure["pinned"])
        self.assertIsNone(decision["route"])
        self.assertIsNone(decision["skill"])

    def test_disabled_plugin_needs_approval(self) -> None:
        decision = self._decision(True, False)
        self.assertEqual(decision["status"], "needs_approval")
        self.assertTrue(decision["plugin"]["installed"])
        self.assertFalse(decision["plugin"]["enabled"])
        self.assertIn("not enabled", decision["reason"])

    def test_installed_and_enabled_plugin_is_ready(self) -> None:
        decision = self._decision(True, True)
        self.assertEqual(decision["status"], "ready")
        self.assertTrue(decision["ready"])
        self.assertEqual(decision["route"], "hermes_worker")
        self.assertEqual(decision["skill"], "blender:blender")
        self.assertEqual(decision["resource_key"], RESOURCE_KEY)
        self.assertEqual(decision["resource_key"], "blender:9876")

    def test_readiness_never_installs_anything(self) -> None:
        state = PluginState(False, False)
        installer = FakeInstaller()
        store = InMemoryConsentStore()
        discovery = state.discovery(installer=installer, store=store)
        slice_ = make_slice(discovery=discovery)
        slice_.readiness()
        self.assertEqual(installer.requests, [])
        self.assertEqual(discovery.network_calls, 0)
        self.assertEqual(store.records(), ())


class DisclosureTests(unittest.TestCase):
    def test_disclosure_states_every_risk_in_plain_words(self) -> None:
        decision = make_slice(discovery=PluginState(False, False).discovery()).readiness()
        points = " ".join(decision["disclosure"]["points"]).lower()
        self.assertIn("arbitrary python", points)
        self.assertIn("online", points)
        self.assertIn("unauthenticated", points)
        self.assertIn("127.0.0.1:9876", points)
        self.assertIn("download", points)
        self.assertIn("size", points)


# -- consent and install ------------------------------------------------------


class InstallTests(unittest.TestCase):
    def _slice(self, *, installed_after: bool = True):
        state = PluginState(False, False)
        installer = FakeInstaller(after=lambda: None)

        if installed_after:
            def after():
                state.installed = True
                state.enabled = True

            installer.after = after
        store = InMemoryConsentStore()
        discovery = state.discovery(installer=installer, store=store)
        return make_slice(discovery=discovery, consent_store=store), installer, store

    def test_install_without_a_consent_record_is_refused_before_the_installer(self) -> None:
        slice_, installer, _ = self._slice()
        with self.assertRaises(ConsentRequiredError):
            slice_.install(None)
        with self.assertRaises(ConsentRequiredError):
            slice_.install("9b11ffeda84b077bea9de52d22733c7a41ca581a")
        self.assertEqual(installer.requests, [])

    def test_consent_for_another_revision_is_refused(self) -> None:
        slice_, installer, _ = self._slice()
        with self.assertRaises(ConsentRequiredError):
            slice_.install(consent(sha=OTHER_SHA))
        self.assertEqual(installer.requests, [])

    def test_consent_for_another_capability_is_refused(self) -> None:
        slice_, installer, _ = self._slice()
        with self.assertRaises(ConsentRequiredError):
            slice_.install(consent(capability="browser"))
        self.assertEqual(installer.requests, [])

    def test_exact_consent_delegates_a_pinned_install_and_verifies_it(self) -> None:
        slice_, installer, store = self._slice()
        result = slice_.install(consent())

        self.assertEqual(len(installer.requests), 1)
        request = installer.requests[0]
        self.assertEqual(request.name, "blender")
        self.assertEqual(request.sha, BLENDER_SHA)
        self.assertEqual(
            request.argv,
            ("hermes", "plugins", "install", "blender", "--ref", BLENDER_SHA, "--enable"),
        )
        self.assertNotIn("--force", request.argv)
        self.assertNotIn("--allow-removed", request.argv)
        self.assertFalse(any("://" in token for token in request.argv))
        self.assertEqual(installer.verified, ["blender"])

        self.assertTrue(result["installed"])
        self.assertTrue(result["verified"])
        self.assertTrue(result["enabled"])
        self.assertEqual(result["status"], "ready")
        self.assertEqual(store.find("blender", BLENDER_SHA).nonce, "nonce-0001")

    def test_install_that_did_not_enable_the_plugin_is_reported_honestly(self) -> None:
        slice_, installer, _ = self._slice(installed_after=False)
        result = slice_.install(consent())
        self.assertTrue(result["installed"])
        self.assertFalse(result["enabled"])
        self.assertEqual(result["status"], "needs_approval")
        self.assertIn("not enabled", result["reason"])

    def test_mismatched_post_install_revision_is_not_success(self) -> None:
        state = PluginState(False, False)
        installer = FakeInstaller(sha=OTHER_SHA, after=lambda: setattr(state, "installed", True))
        discovery = state.discovery(installer=installer, store=InMemoryConsentStore())
        slice_ = make_slice(discovery=discovery)
        result = slice_.install(consent())
        self.assertFalse(result["installed"])
        self.assertFalse(result["verified"])
        self.assertEqual(result["status"], "needs_approval")


# -- task preparation ---------------------------------------------------------


class SliceRuntimeCase(unittest.TestCase):
    """A real runtime under a temporary directory; nothing is spawned."""

    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.launch = LaunchRecorder()
        self.runtime = NeoRuntime.build(
            data_dir=self.root / "plugin-data",
            launch=self.launch,
            supervisor=object(),
            max_parallel=8,
        )

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def workdir(self) -> str:
        return self.runtime.workspace("blender-task")

    def ready_slice(self, *, port=None, installer=None, state=None):
        state = state or PluginState(True, True)
        discovery = state.discovery(installer=installer)
        return make_slice(
            discovery=discovery, runtime=self.runtime, port=port or FakePort(False)
        )


class PrepareTaskTests(SliceRuntimeCase):
    def test_prepare_returns_the_task_fields_with_the_pinned_worker(self) -> None:
        slice_ = self.ready_slice()
        task_id = "task-1"
        workdir = self.workdir()
        body = slice_.prepare_task(
            "make a low-poly rocket", "session-1", workdir, task_id=task_id
        )

        self.assertEqual(body["kind"], "blender")
        self.assertEqual(body["executor"], "hermes_worker")
        self.assertEqual(body["resource_key"], "blender:9876")
        self.assertEqual(body["capability"], "blender")
        self.assertEqual(body["skills"], ("blender:blender", "neo-worker"))
        self.assertEqual(body["expected_artifacts"], ("scene.blend",))

        argv = list(body["argv"])
        self.assertEqual(argv[0], "hermes")
        self.assertIn("--query-file", argv)
        self.assertEqual(argv[argv.index("--skills") + 1], "blender:blender,neo-worker")
        self.assertNotIn("--", argv)
        prompt_path = Path(argv[argv.index("--query-file") + 1])
        self.assertTrue(prompt_path.is_file())
        self.assertEqual(os.stat(prompt_path).st_mode & 0o777, 0o600)

    def test_prompt_carries_the_intent_and_every_required_step(self) -> None:
        slice_ = self.ready_slice()
        workdir = self.workdir()
        body = slice_.prepare_task(
            "make a low-poly rocket", "session-1", workdir, task_id="task-1"
        )
        prompt = body["prompt"]
        self.assertIn("make a low-poly rocket", prompt)
        self.assertIn("low-poly", prompt.lower())
        self.assertIn("get_objects_summary", prompt)
        self.assertIn("scene.blend", prompt)
        self.assertIn(workdir, prompt)
        self.assertIn("screenshot", prompt.lower())
        self.assertIn("approval", prompt.lower())
        self.assertIn("add-on", prompt.lower())
        self.assertIn("nested", prompt.lower())

    def test_prompt_never_carries_a_token_or_credential(self) -> None:
        slice_ = self.ready_slice()
        body = slice_.prepare_task(
            "make a low-poly rocket", "session-1", self.workdir(), task_id="task-1"
        )
        prompt = body["prompt"]
        self.assertNotIn("NEO_TASK_TOKEN", prompt)
        self.assertNotIn("Bearer ", prompt)
        self.assertNotIn("sk-", prompt)

    def test_prepare_refuses_a_ready_plugin_when_blender_is_missing(self) -> None:
        slice_ = make_slice(
            discovery=PluginState(True, True).discovery(),
            runtime=self.runtime,
            probe=FakeProbe(present=False),
        )
        with self.assertRaises(BlenderNotReadyError) as caught:
            slice_.prepare_task("make a rocket", "session-1", self.workdir(), task_id="t")
        self.assertFalse(caught.exception.decision["ready"])
        self.assertEqual(caught.exception.decision["disclosure"]["sha"], BLENDER_SHA)

    def test_prepare_refuses_when_the_plugin_is_missing(self) -> None:
        slice_ = make_slice(
            discovery=PluginState(False, False).discovery(), runtime=self.runtime
        )
        with self.assertRaises(BlenderNotReadyError) as caught:
            slice_.prepare_task("make a rocket", "session-1", self.workdir(), task_id="t")
        self.assertEqual(caught.exception.decision["status"], "needs_approval")
        self.assertIsNone(caught.exception.decision["route"])

    def test_prepare_requires_an_absolute_existing_workdir(self) -> None:
        slice_ = self.ready_slice()
        for workdir in ("relative/path", "/definitely/not/here", None):
            with self.assertRaises(ValueError):
                slice_.prepare_task("make a rocket", "session-1", workdir, task_id="t")

    def test_prepare_requires_a_conversation_and_an_intent(self) -> None:
        slice_ = self.ready_slice()
        with self.assertRaises(ValueError):
            slice_.prepare_task("make a rocket", "", self.workdir(), task_id="t")
        with self.assertRaises(ValueError):
            slice_.prepare_task("   ", "session-1", self.workdir(), task_id="t")

    def test_prepare_never_installs_or_runs_a_command(self) -> None:
        installer = FakeInstaller()
        state = PluginState(True, True)
        slice_ = self.ready_slice(installer=installer, state=state)
        slice_.prepare_task("make a rocket", "session-1", self.workdir(), task_id="t")
        self.assertEqual(installer.requests, [])
        self.assertEqual(slice_.discovery.network_calls, 0)


class PortTests(SliceRuntimeCase):
    def test_busy_port_that_neo_does_not_own_is_refused(self) -> None:
        port = FakePort(True)
        slice_ = self.ready_slice(port=port)
        with self.assertRaises(BlenderPortBusyError) as caught:
            slice_.prepare_task("make a rocket", "session-1", self.workdir(), task_id="t")
        error = caught.exception
        self.assertEqual(error.status["port"], BLENDER_PORT)
        self.assertFalse(error.status["authenticated"])
        self.assertIn("9876", str(error))
        self.assertEqual(len(port.calls), 1, "a busy port must not be retried")

    def test_second_blender_task_queues_on_the_same_resource_key(self) -> None:
        slice_ = self.ready_slice()
        workdir = self.workdir()
        first = self.runtime.service.create_task(
            conversation_key="session-1",
            kind="blender",
            intent="first rocket",
            workdir=workdir,
            resource_key=slice_.prepare_task(
                "first rocket", "session-1", workdir, task_id="task-1"
            )["resource_key"],
            task_id="task-1",
        )
        self.assertEqual(first["state"], "starting")

        # The port is now listening because NEO's own worker owns it.
        owned = self.ready_slice(port=FakePort(True))
        fields = owned.prepare_task("second rocket", "session-1", workdir, task_id="task-2")
        self.assertEqual(fields["resource_key"], "blender:9876")

        second = self.runtime.service.create_task(
            conversation_key="session-1",
            kind="blender",
            intent="second rocket",
            workdir=workdir,
            resource_key=fields["resource_key"],
            task_id="task-2",
        )
        self.assertEqual(second["state"], "queued")
        self.assertEqual(second["resource_key"], "blender:9876")


# -- completion verification --------------------------------------------------


class CompletionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.workdir = Path(self.tempdir.name)
        self.slice = make_slice(discovery=PluginState(True, True).discovery())

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def task(self, expected=None) -> dict:
        return {
            "id": "task-1",
            "kind": "blender",
            "workdir": str(self.workdir),
            "expected_json": json.dumps(list(expected or [])),
        }

    def write(self, name: str, body: bytes = b"x") -> None:
        path = self.workdir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)

    def test_a_real_blend_and_screenshot_complete_the_task(self) -> None:
        self.write("scene.blend", blend_bytes())
        self.write("shot-01.png", b"\x89PNG\r\n\x1a\n")
        evidence = self.slice.verify_completion(
            self.task(["scene.blend", "shot-01.png"])
        )
        self.assertTrue(evidence.executor_confirmed)
        self.assertEqual(
            set(evidence.verified_artifacts), {"scene.blend", "shot-01.png"}
        )

    def test_a_blend_found_on_disk_is_accepted_when_not_named_up_front(self) -> None:
        self.write("rocket.blend", blend_bytes())
        evidence = self.slice.verify_completion(self.task())
        self.assertEqual(evidence.verified_artifacts, ("rocket.blend",))

    def test_missing_blend_prevents_completion(self) -> None:
        self.write("shot-01.png", b"\x89PNG\r\n\x1a\n")
        with self.assertRaises(BlenderCompletionError) as caught:
            self.slice.verify_completion(self.task(["shot-01.png"]))
        self.assertIn(".blend", str(caught.exception))

    def test_an_empty_blend_is_not_a_real_blend(self) -> None:
        self.write("scene.blend", b"")
        with self.assertRaises(BlenderCompletionError):
            self.slice.verify_completion(self.task(["scene.blend"]))

    def test_a_file_that_is_not_a_blend_is_refused(self) -> None:
        self.write("scene.blend", b"just some text")
        with self.assertRaises(BlenderCompletionError):
            self.slice.verify_completion(self.task(["scene.blend"]))

    def test_an_expected_screenshot_that_is_missing_prevents_completion(self) -> None:
        self.write("scene.blend", blend_bytes())
        with self.assertRaises(BlenderCompletionError):
            self.slice.verify_completion(self.task(["scene.blend", "shot-01.png"]))

    def test_a_blend_outside_the_workdir_is_refused(self) -> None:
        outside = Path(self.tempdir.name).parent / "outside.blend"
        outside.write_bytes(blend_bytes())
        try:
            with self.assertRaises(BlenderCompletionError):
                self.slice.verify_completion(self.task(["../outside.blend"]))
        finally:
            outside.unlink()


class PromptBuilderTests(unittest.TestCase):
    def test_prompt_is_bounded_and_names_the_workdir(self) -> None:
        prompt = build_worker_prompt("make a cube", "/tmp/task")
        self.assertIn("make a cube", prompt)
        self.assertIn("/tmp/task", prompt)
        self.assertLess(len(prompt), 6000)

    def test_prompt_refuses_an_empty_intent(self) -> None:
        with self.assertRaises(ValueError):
            build_worker_prompt("  ", "/tmp/task")


# -- conversation routing -----------------------------------------------------


class ConversationRoutingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.launch = LaunchRecorder()
        self.runtime = NeoRuntime.build(
            data_dir=self.root / "plugin-data",
            launch=self.launch,
            supervisor=object(),
            max_parallel=8,
        )
        self.tools = ConversationTools(lambda: self.runtime)

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def attach(self, *, state=None, probe=None, port=None, installer=None):
        state = state or PluginState(True, True)
        self.runtime.blender = make_slice(
            discovery=state.discovery(installer=installer),
            runtime=self.runtime,
            probe=probe or FakeProbe(),
            port=port or FakePort(False),
        )
        return self.runtime.blender

    def submit(self, intent: str, **kwargs) -> dict:
        payload = self.tools.neo_submit_task(
            {"intent": intent, "kind": "blender"}, session_id="session-1", **kwargs
        )
        body = json.loads(payload)
        self.assertIsInstance(body, dict)
        return body

    def tasks(self) -> list:
        return [
            row
            for row in self.runtime.registry.list_tasks()
            if row.get("kind") == "blender"
        ]

    def test_missing_plugin_returns_needs_approval_and_starts_nothing(self) -> None:
        self.attach(state=PluginState(False, False))
        body = self.submit("make a low-poly rocket")
        self.assertFalse(body["ok"])
        self.assertEqual(body["state"], "needs_approval")
        self.assertTrue(body["requires_approval"])
        self.assertEqual(body["disclosure"]["name"], "blender")
        self.assertEqual(body["disclosure"]["sha"], BLENDER_SHA)
        self.assertEqual(body["disclosure"]["version"], BLENDER_VERSION)
        self.assertEqual(self.launch.calls, [])
        self.assertEqual(self.tasks(), [])

    def test_disabled_plugin_returns_needs_approval(self) -> None:
        self.attach(state=PluginState(True, False))
        body = self.submit("make a low-poly rocket")
        self.assertFalse(body["ok"])
        self.assertEqual(body["state"], "needs_approval")
        self.assertEqual(self.launch.calls, [])
        self.assertEqual(self.tasks(), [])

    def test_missing_blender_app_returns_needs_approval(self) -> None:
        self.attach(probe=FakeProbe(present=False))
        body = self.submit("make a low-poly rocket")
        self.assertFalse(body["ok"])
        self.assertEqual(body["state"], "needs_approval")
        self.assertEqual(self.launch.calls, [])

    def test_busy_port_is_reported_without_creating_or_launching_anything(self) -> None:
        self.attach(port=FakePort(True))
        body = self.submit("make a low-poly rocket")
        self.assertFalse(body["ok"])
        self.assertEqual(body["error"]["code"], "blender_port_busy")
        self.assertEqual(body["port"]["port"], BLENDER_PORT)
        self.assertEqual(self.launch.calls, [])
        self.assertEqual(self.tasks(), [])

    def test_ready_submission_launches_the_pinned_worker_and_locks_the_port(self) -> None:
        self.attach()
        body = self.submit("make a low-poly rocket")
        self.assertTrue(body["ok"], body)
        self.assertEqual(body["kind"], "blender")
        self.assertEqual(body["executor"], "hermes_worker")
        self.assertEqual(body["state"], "starting")

        task = self.runtime.registry.get_task(body["task_id"])
        self.assertEqual(task["resource_key"], "blender:9876")
        self.assertEqual(task["executor"], "hermes_worker")
        self.assertEqual(json.loads(task["expected_json"]), ["scene.blend"])

        argv = self.launch.last()["argv"]
        self.assertEqual(argv[argv.index("--skills") + 1], "blender:blender,neo-worker")
        prompt_path = Path(argv[argv.index("--query-file") + 1])
        self.assertIn("make a low-poly rocket", prompt_path.read_text(encoding="utf-8"))

    def test_second_submission_queues_behind_the_first(self) -> None:
        self.attach()
        first = self.submit("make a low-poly rocket")
        self.assertEqual(first["state"], "starting")
        # The first worker now owns the port; the second task must still queue.
        self.attach(port=FakePort(True))
        second = self.submit("make a low-poly robot")
        self.assertTrue(second["ok"], second)
        self.assertEqual(second["state"], "queued")
        queued = self.runtime.registry.get_task(second["task_id"])
        self.assertEqual(queued["state"], "queued")
        self.assertEqual(queued["resource_key"], "blender:9876")

    def test_submit_never_installs_the_plugin(self) -> None:
        installer = FakeInstaller()
        self.attach(installer=installer)
        self.submit("make a low-poly rocket")
        self.assertEqual(installer.requests, [])

    def test_without_a_slice_the_legacy_worker_route_still_runs(self) -> None:
        body = self.submit("make a low-poly rocket")
        self.assertTrue(body["ok"], body)
        argv = self.launch.last()["argv"]
        self.assertEqual(argv[argv.index("--skills") + 1], "blender:blender,neo-worker")


if __name__ == "__main__":
    unittest.main()
