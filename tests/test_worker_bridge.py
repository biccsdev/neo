"""Task 8 — scoped worker bridge tools.

A worker holds ``NEO_TASK_ID`` and its per-task token. The bridge is the only
surface it may call: it authorises every call against the task's stored token
digest, answers identically for an unknown task and a wrong token, and never
touches another task's row, preview directory or events.

The behaviour under test, from the implementation plan (Task 8, sections
4.1-4.6):

* update is authenticated and scoped to the caller's own task;
* completion is proved by verified artifacts beneath the task workdir, never by
  the caller's assertion, and persists canonical artifact names;
* heartbeat advances only the heartbeat clock;
* preview publish copies one allowlisted image into a task-scoped directory
  under the byte cap, allocates a revision that concurrent calls cannot share,
  and emits ``task.preview_updated`` for a path that already exists;
* the OpenAI-style schemas carry ``task_id`` and ``token`` and no shell field;
* the worker skill tells the worker the same rules in prose.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import sqlite3
import sys
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import schemas  # noqa: E402
import neo_tools as tools  # noqa: E402
from artifacts import (  # noqa: E402
    ArtifactEscapesWorkdirError,
    ArtifactNotFoundError,
    EmptyExpectedArtifactsError,
    MissingWorkdirError,
    UnsupportedArtifactTypeError,
)
from events import TaskService  # noqa: E402
from registry import InvalidWorkerTokenError, Registry, VersionConflictError  # noqa: E402
from state_machine import TerminalStateError  # noqa: E402
from neo_tools import (  # noqa: E402
    DEFAULT_PREVIEW_MAX_BYTES,
    PreviewTooLargeError,
    WorkerBridge,
    WorkerTokenStore,
    WorkerToolError,
)

TOKEN_ERROR = "worker token rejected"


class WorkerBridgeTestCase(unittest.TestCase):
    """Two tasks in one registry, each with its own workdir, token and preview."""

    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.db_path = self.root / "registry.db"
        self.registry = Registry(path=self.db_path, max_parallel=8)
        self.service = TaskService(self.registry)
        self.tokens = WorkerTokenStore(self.registry)
        self.previews = self.root / "previews"
        self.bridge = WorkerBridge(
            self.service,
            self.tokens,
            preview_root=self.previews,
            preview_max_bytes=4096,
        )
        self.task = self.create_task("first")
        self.workdir = Path(self.task["workdir"])
        self.token = self.tokens.issue(self.task["id"])

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    # -- helpers ---------------------------------------------------------

    def create_task(self, name: str, *, workdir=True, task_id=None, expected=None) -> dict:
        directory = None
        if workdir:
            directory = str(self.root / ("work-%s" % name))
            Path(directory).mkdir(parents=True, exist_ok=True)
        task = self.service.create_task(
            conversation_key="conversation",
            kind="coding",
            intent="do the work for %s" % name,
            workdir=directory,
            task_id=task_id,
        )
        if expected is not None:
            task = self.registry.update_task(
                task["id"],
                expected_version=task["version"],
                expected_json=json.dumps(expected),
            )
        return task

    def snapshot(self, task_id: str):
        return self.registry.get_task(task_id)

    def events(self, task_id=None) -> list:
        return [
            event
            for event in self.service.events_since("conversation")
            if task_id is None or event["task_id"] == task_id
        ]

    def start(self, task: dict, token: str) -> dict:
        return self.bridge.neo_task_update(
            task["id"], token, state="running", expected_version=task["version"]
        )

    def write(self, name: str, data: bytes = b"payload", workdir=None) -> Path:
        path = Path(workdir or self.workdir) / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return path

    # -- authorisation ---------------------------------------------------

    def test_every_bridge_call_needs_the_per_task_token(self) -> None:
        task = self.start(self.task, self.token)
        before = self.snapshot(task["id"])
        second = self.create_task("second")
        second_token = self.tokens.issue(second["id"])
        events_before = len(self.events(task["id"]))
        source = self.write("shot.png", b"\x89PNG\r\n\x1a\n")

        calls = {
            "update": lambda candidate: self.bridge.neo_task_update(
                task["id"], candidate, state="waiting"
            ),
            "complete": lambda candidate: self.bridge.neo_task_complete(
                task["id"], candidate
            ),
            "publish": lambda candidate: self.bridge.neo_task_publish_preview(
                task["id"], candidate, "shot.png"
            ),
            "heartbeat": lambda candidate: self.bridge.neo_task_heartbeat(
                task["id"], candidate
            ),
            "get": lambda candidate: self.bridge.neo_task_get(task["id"], candidate),
        }
        for name, call in calls.items():
            for candidate in ("", None, "wrong-token", second_token):
                with self.subTest(call=name, candidate=candidate):
                    with self.assertRaises(InvalidWorkerTokenError):
                        call(candidate)
                    self.assertEqual(self.snapshot(task["id"]), before)
                    self.assertEqual(len(self.events(task["id"])), events_before)

        self.assertTrue(source.exists())
        self.assertFalse(self.previews.exists())

    def test_unknown_task_and_wrong_token_are_indistinguishable(self) -> None:
        messages = []
        for task_id, token in (
            ("missing", self.token),
            (self.task["id"], "wrong-token"),
            ("missing", "wrong-token"),
        ):
            with self.subTest(task_id=task_id, token=token):
                with self.assertRaises(InvalidWorkerTokenError) as caught:
                    self.bridge.neo_task_get(task_id, token)
                messages.append(str(caught.exception))

        self.assertEqual(len(set(messages)), 1)
        self.assertIn(TOKEN_ERROR, messages[0])

    def test_another_tasks_token_cannot_touch_this_task(self) -> None:
        other = self.create_task("other")
        other_token = self.tokens.issue(other["id"])
        before = self.snapshot(self.task["id"])

        for call in (
            lambda: self.bridge.neo_task_update(
                self.task["id"], other_token, state="running"
            ),
            lambda: self.bridge.neo_task_complete(self.task["id"], other_token),
            lambda: self.bridge.neo_task_heartbeat(self.task["id"], other_token),
        ):
            with self.assertRaises(InvalidWorkerTokenError):
                call()

        self.assertEqual(self.snapshot(self.task["id"]), before)
        self.assertEqual(len(self.events(self.task["id"])), 1)

    # -- update ----------------------------------------------------------

    def test_bridge_update_runs_one_authenticated_transition(self) -> None:
        task = self.start(self.task, self.token)

        waiting = self.bridge.neo_task_update(
            task["id"], self.token, state="waiting", detail="blocked on input"
        )

        self.assertEqual(waiting["state"], "waiting")
        events = self.events(task["id"])
        self.assertEqual([event["kind"] for event in events][-1], "task.updated")
        self.assertEqual(events[-1]["payload"]["detail"], "blocked on input")

    # -- completion ------------------------------------------------------

    def test_completion_without_any_expected_artifact_is_refused(self) -> None:
        task = self.start(self.task, self.token)

        with self.assertRaises(EmptyExpectedArtifactsError):
            self.bridge.neo_task_complete(task["id"], self.token)

        self.assertEqual(self.snapshot(task["id"])["state"], "running")
        self.assertNotIn("task.completed", [event["kind"] for event in self.events()])

    def test_completion_with_a_missing_artifact_is_refused(self) -> None:
        task = self.start(self.task, self.token)

        with self.assertRaises(ArtifactNotFoundError):
            self.bridge.neo_task_complete(task["id"], self.token, ["out/report.md"])

        self.assertEqual(self.snapshot(task["id"])["state"], "running")
        self.assertNotIn("task.completed", [event["kind"] for event in self.events()])

    def test_completion_needs_the_stored_expectation_too(self) -> None:
        task = self.create_task("stored", expected=["out/report.md"])
        token = self.tokens.issue(task["id"])
        running = self.start(task, token)
        self.write("out/other.md", workdir=task["workdir"])

        with self.assertRaises(ArtifactNotFoundError):
            self.bridge.neo_task_complete(
                running["id"], token, ["out/other.md"]
            )

        self.assertEqual(self.snapshot(task["id"])["state"], "running")

    def test_completion_with_a_verified_artifact_emits_one_event(self) -> None:
        task = self.create_task("verified", expected=["out/report.md"])
        token = self.tokens.issue(task["id"])
        self.start(task, token)
        self.write("out/report.md", b"# report\n", workdir=task["workdir"])

        completed = self.bridge.neo_task_complete(task["id"], token)

        self.assertEqual(completed["state"], "completed")
        self.assertEqual(json.loads(completed["expected_json"]), ["out/report.md"])
        self.assertEqual(json.loads(completed["artifacts_json"]), ["out/report.md"])
        kinds = [event["kind"] for event in self.events(task["id"])]
        self.assertEqual(kinds.count("task.completed"), 1)
        self.assertEqual(kinds[-1], "task.completed")

    def test_supplied_artifacts_are_canonicalised_before_completion(self) -> None:
        task = self.start(self.task, self.token)
        self.write("shots/final.png", b"\x89PNG\r\n\x1a\n")

        completed = self.bridge.neo_task_complete(
            task["id"], self.token, ["./shots/final.png"]
        )

        self.assertEqual(completed["state"], "completed")
        self.assertEqual(json.loads(completed["expected_json"]), ["shots/final.png"])
        self.assertEqual(json.loads(completed["artifacts_json"]), ["shots/final.png"])

    def test_completion_cannot_escape_the_workdir(self) -> None:
        task = self.start(self.task, self.token)
        self.write("outside.png", b"\x89PNG\r\n\x1a\n", workdir=self.root)

        with self.assertRaises(ArtifactEscapesWorkdirError):
            self.bridge.neo_task_complete(task["id"], self.token, ["../outside.png"])

        self.assertEqual(self.snapshot(task["id"])["state"], "running")

    def test_completion_is_impossible_without_a_workdir(self) -> None:
        task = self.create_task("nowork", workdir=False)
        token = self.tokens.issue(task["id"])
        self.start(task, token)

        with self.assertRaises(MissingWorkdirError):
            self.bridge.neo_task_complete(task["id"], token, ["out/report.md"])

        self.assertEqual(self.snapshot(task["id"])["state"], "running")

    def test_second_completion_emits_no_second_event(self) -> None:
        task = self.start(self.task, self.token)
        self.write("out/report.md")
        self.bridge.neo_task_complete(task["id"], self.token, ["out/report.md"])

        with self.assertRaises((TerminalStateError, InvalidWorkerTokenError)):
            self.bridge.neo_task_complete(
                task["id"], self.token, ["out/report.md"]
            )

        kinds = [event["kind"] for event in self.events(task["id"])]
        self.assertEqual(kinds.count("task.completed"), 1)

    # -- heartbeat -------------------------------------------------------

    def test_heartbeat_advances_only_the_heartbeat_clock(self) -> None:
        task = self.start(self.task, self.token)
        events_before = len(self.events(task["id"]))

        first = self.bridge.neo_task_heartbeat(task["id"], self.token)
        time.sleep(0.01)
        second = self.bridge.neo_task_heartbeat(task["id"], self.token)

        self.assertEqual(first["state"], "running")
        self.assertEqual(second["state"], "running")
        self.assertIsNotNone(first["heartbeat_at"])
        self.assertGreaterEqual(second["heartbeat_at"], first["heartbeat_at"])
        self.assertEqual(len(self.events(task["id"])), events_before)
        stored = self.snapshot(task["id"])
        self.assertEqual(stored["state"], "running")
        self.assertEqual(stored["heartbeat_at"], second["heartbeat_at"])

    # -- preview publish -------------------------------------------------

    def test_publish_copies_into_a_task_scoped_preview(self) -> None:
        task = self.start(self.task, self.token)
        source = self.write("a.png", b"\x89PNG\r\n\x1a\n" + b"0" * 64)

        result = self.bridge.neo_task_publish_preview(task["id"], self.token, "a.png")

        destination = Path(result["preview_path"])
        self.assertTrue(destination.is_file())
        self.assertEqual(destination.read_bytes(), source.read_bytes())
        self.assertEqual(result["preview_rev"], 1)
        self.assertEqual(result["mime"], "image/png")
        self.assertTrue(str(destination.resolve()).startswith(str(self.previews.resolve())))
        stored = self.snapshot(task["id"])
        self.assertEqual(stored["preview_rev"], 1)
        self.assertEqual(stored["preview_path"], str(destination))

        event = self.events(task["id"])[-1]
        self.assertEqual(event["kind"], "task.preview_updated")
        payload = event["payload"]
        self.assertEqual(payload["task_id"], task["id"])
        self.assertEqual(payload["rev"], 1)
        self.assertEqual(payload["path"], str(destination))
        self.assertEqual(payload["mime"], "image/png")
        self.assertTrue(Path(payload["path"]).is_file(), "event must follow the copy")

    def test_second_publish_keeps_the_first_revision_on_disk(self) -> None:
        task = self.start(self.task, self.token)
        self.write("a.png", b"one")
        first = self.bridge.neo_task_publish_preview(task["id"], self.token, "a.png")
        self.write("a.png", b"two")
        second = self.bridge.neo_task_publish_preview(task["id"], self.token, "a.png")

        self.assertEqual(first["preview_rev"], 1)
        self.assertEqual(second["preview_rev"], 2)
        self.assertNotEqual(first["preview_path"], second["preview_path"])
        self.assertTrue(Path(first["preview_path"]).is_file())
        self.assertTrue(Path(second["preview_path"]).is_file())

    def test_publish_accepts_an_absolute_source_inside_the_workdir(self) -> None:
        task = self.start(self.task, self.token)
        source = self.write("a.jpg", b"\xff\xd8\xff")

        result = self.bridge.neo_task_publish_preview(
            task["id"], self.token, str(source)
        )

        self.assertTrue(Path(result["preview_path"]).is_file())
        self.assertEqual(result["mime"], "image/jpeg")

    def test_publish_refuses_sources_outside_the_workdir(self) -> None:
        task = self.start(self.task, self.token)
        outside = self.write("outside.png", b"x", workdir=self.root)

        for candidate in ("../outside.png", str(outside)):
            with self.subTest(candidate=candidate):
                with self.assertRaises(ArtifactEscapesWorkdirError):
                    self.bridge.neo_task_publish_preview(
                        task["id"], self.token, candidate
                    )

        self.assertEqual(self.snapshot(task["id"])["preview_rev"], 0)
        self.assertEqual(
            [event["kind"] for event in self.events(task["id"])].count(
                "task.preview_updated"
            ),
            0,
        )

    def test_publish_refuses_a_symlink_source(self) -> None:
        task = self.start(self.task, self.token)
        real = self.write("real.png", b"x", workdir=self.root)
        (self.workdir / "link.png").symlink_to(real)

        with self.assertRaises(ArtifactEscapesWorkdirError):
            self.bridge.neo_task_publish_preview(task["id"], self.token, "link.png")

        self.assertEqual(self.snapshot(task["id"])["preview_rev"], 0)

    def test_publish_refuses_unsupported_types(self) -> None:
        task = self.start(self.task, self.token)
        self.write("notes.gif", b"GIF89a")
        self.write("notes.txt", b"text")

        for candidate in ("notes.gif", "notes.txt"):
            with self.subTest(candidate=candidate):
                with self.assertRaises(UnsupportedArtifactTypeError):
                    self.bridge.neo_task_publish_preview(
                        task["id"], self.token, candidate
                    )

        self.assertEqual(self.snapshot(task["id"])["preview_rev"], 0)

    def test_publish_refuses_an_oversize_image(self) -> None:
        task = self.start(self.task, self.token)
        self.write("big.png", b"0" * (4097))

        with self.assertRaises(PreviewTooLargeError):
            self.bridge.neo_task_publish_preview(task["id"], self.token, "big.png")

        self.assertEqual(self.snapshot(task["id"])["preview_rev"], 0)
        self.assertFalse(
            any(self.previews.rglob("*.png")) if self.previews.exists() else False
        )

    def test_default_preview_cap_is_two_mebibytes(self) -> None:
        self.assertEqual(DEFAULT_PREVIEW_MAX_BYTES, 2 * 1024 * 1024)

    def test_publish_requires_a_workdir(self) -> None:
        task = self.create_task("nowork", workdir=False)
        token = self.tokens.issue(task["id"])
        self.start(task, token)

        with self.assertRaises(MissingWorkdirError):
            self.bridge.neo_task_publish_preview(task["id"], token, "a.png")

    def test_task_id_cannot_escape_the_preview_root(self) -> None:
        task = self.create_task("tricky", task_id="../escape")
        token = self.tokens.issue(task["id"])
        self.start(task, token)
        self.write("a.png", b"x", workdir=task["workdir"])

        result = self.bridge.neo_task_publish_preview(task["id"], token, "a.png")

        destination = Path(result["preview_path"]).resolve()
        self.assertEqual(destination.parent.parent, self.previews.resolve())
        self.assertNotIn("..", destination.parts)
        self.assertFalse((self.previews.parent / "escape").exists())

    def test_another_task_never_sees_this_preview(self) -> None:
        task = self.start(self.task, self.token)
        self.write("a.png", b"x")
        published = self.bridge.neo_task_publish_preview(task["id"], self.token, "a.png")
        other = self.create_task("neighbour")
        other_token = self.tokens.issue(other["id"])

        seen = self.bridge.neo_task_get(other["id"], other_token)

        self.assertIsNone(seen["preview_path"])
        self.assertEqual(seen["preview_rev"], 0)
        self.assertEqual(seen["id"], other["id"])
        sibling = Path(published["preview_path"]).parent.parent
        self.assertEqual(sibling, self.previews.resolve())

    def test_concurrent_publishes_get_distinct_revisions(self) -> None:
        task = self.start(self.task, self.token)
        self.write("a.png", b"x" * 128)
        barrier = threading.Barrier(2)

        def publish() -> dict:
            barrier.wait(timeout=10)
            return self.bridge.neo_task_publish_preview(
                task["id"], self.token, "a.png"
            )

        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(lambda _: publish(), range(2)))

        revisions = sorted(result["preview_rev"] for result in results)
        self.assertEqual(revisions, [1, 2])
        paths = {result["preview_path"] for result in results}
        self.assertEqual(len(paths), 2)
        for path in paths:
            self.assertTrue(Path(path).is_file())
        self.assertEqual(self.snapshot(task["id"])["preview_rev"], 2)
        preview_events = [
            event
            for event in self.events(task["id"])
            if event["kind"] == "task.preview_updated"
        ]
        self.assertEqual([event["payload"]["rev"] for event in preview_events], [1, 2])
        self.assertEqual(
            self.snapshot(task["id"])["preview_path"], preview_events[-1]["payload"]["path"]
        )

    # -- task read -------------------------------------------------------

    def test_get_returns_only_the_callers_task(self) -> None:
        task = self.start(self.task, self.token)
        self.write("out/report.md")
        self.bridge.neo_task_complete(task["id"], self.token, ["out/report.md"])

        seen = self.bridge.neo_task_get(task["id"], self.token)

        self.assertEqual(seen["id"], task["id"])
        self.assertEqual(seen["state"], "completed")
        self.assertEqual(seen["expected_artifacts"], ["out/report.md"])
        self.assertEqual(seen["artifacts"], ["out/report.md"])
        for hidden in (
            "conversation_key",
            "dedupe_key",
            "token_hash",
            "token",
            "expected_json",
            "artifacts_json",
        ):
            self.assertNotIn(hidden, seen)

    # -- schemas ---------------------------------------------------------

    def test_schemas_cover_the_bridge_and_require_task_and_token(self) -> None:
        self.assertEqual(
            set(schemas.WORKER_TOOL_NAMES),
            {
                "neo_task_update",
                "neo_task_complete",
                "neo_task_publish_preview",
                "neo_task_heartbeat",
                "neo_task_get",
            },
        )
        forbidden = {"command", "cmd", "shell", "script", "argv", "exec", "terminal"}
        for schema in schemas.worker_tool_schemas():
            with self.subTest(schema=schema["function"]["name"]):
                function = schema["function"]
                self.assertEqual(schema["type"], "function")
                self.assertTrue(function["description"].strip())
                parameters = function["parameters"]
                self.assertEqual(parameters["type"], "object")
                self.assertIn("task_id", parameters["required"])
                self.assertIn("token", parameters["required"])
                self.assertTrue(forbidden.isdisjoint(parameters["properties"]))
                self.assertFalse(parameters.get("additionalProperties", True))

    def test_schema_returns_independent_copies(self) -> None:
        first = schemas.worker_tool_schemas()
        first[0]["function"]["name"] = "mutated"
        second = schemas.worker_tool_schemas()

        self.assertIn(second[0]["function"]["name"], schemas.WORKER_TOOL_NAMES)

    # -- worker skill ----------------------------------------------------

    def test_worker_skill_documents_the_bridge_contract(self) -> None:
        skill = (REPO_ROOT / "skills" / "neo-worker" / "SKILL.md").read_text(
            encoding="utf-8"
        )

        for required in (
            "NEO_TASK_ID",
            "NEO_TASK_TOKEN",
            "neo_task_update",
            "neo_task_publish_preview",
            "neo_task_complete",
            "neo_task_heartbeat",
            "approval",
            "nested",
        ):
            with self.subTest(required=required):
                self.assertIn(required, skill)
        self.assertRegex(skill, r"(?m)^---\s*$")
        self.assertRegex(skill, r"(?m)^name:\s*neo-worker\s*$")


if __name__ == "__main__":
    unittest.main()
