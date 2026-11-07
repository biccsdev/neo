"""Task 12 — task-scoped preview service.

The dashboard shows one image for one task. This module is the only thing that
turns a persisted ``preview_path`` into bytes, so every rule about *which* bytes
is proven here:

* a preview belongs to exactly one task, read by id from the registry — a
  missing or deleted task is unavailable, never another task's image;
* when the caller supplies a conversation key it must equal the task's
  immutable binding;
* the persisted path must be a real regular file strictly beneath the preview
  root, reached without a symlink, with an allowlisted image extension, a
  non-zero size and within the byte cap;
* bytes are read once from an open descriptor and hashed, and the revision
  reported is the one whose path was still current after the read — a
  concurrent publish either yields the newer bytes with the newer revision or
  an honest "changed while reading", never a mix;
* the response is JSON-safe (base64 text, no binary body, no server path), so
  the API layer never needs a binary response path.

Section 4.1-4.6 of the implementation plan.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import sys
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Optional
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import previews  # noqa: E402
import tools  # noqa: E402
from previews import (  # noqa: E402
    DEFAULT_PREVIEW_MAX_BYTES,
    PREVIEW_MIME_TYPES,
    PreviewService,
)
from registry import Registry  # noqa: E402

PNG = b"\x89PNG\r\n\x1a\n" + b"pixels" * 4
JPEG = b"\xff\xd8\xff\xe0" + b"jpeg-body" * 3


class PreviewTestCase(unittest.TestCase):
    """One temp database, one preview root, helpers that act like a publisher."""

    def setUp(self) -> None:
        self._tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tempdir.cleanup)
        self.root = Path(self._tempdir.name)
        self.preview_root = self.root / "previews"
        self.preview_root.mkdir()
        self.registry = Registry(path=self.root / "neo.db", max_parallel=8)
        self.service = PreviewService(self.registry, preview_root=self.preview_root)

    # -- fixtures --------------------------------------------------------

    def make_task(self, conversation_key: str = "conv-a", kind: str = "coding") -> dict:
        return self.registry.create_task(
            conversation_key=conversation_key,
            kind=kind,
            intent="build a page",
        )

    def version_of(self, task_id: str) -> int:
        task = self.registry.get_task(task_id)
        assert task is not None
        return int(task["version"])

    def publish(
        self,
        task: dict,
        data: bytes = PNG,
        *,
        revision: int = 1,
        directory: Optional[str] = None,
        name: Optional[str] = None,
    ) -> Path:
        """Write a preview file and record it the way the worker bridge does."""
        task_dir = self.preview_root / (directory or task["id"])
        task_dir.mkdir(parents=True, exist_ok=True)
        path = task_dir / (name or "rev-%d.png" % revision)
        path.write_bytes(data)
        self.registry.update_task(
            task["id"],
            expected_version=self.version_of(task["id"]),
            preview_path=str(path),
            preview_rev=revision,
        )
        return path

    def retire(self, task: dict) -> None:
        """Drive a task to a terminal state so it can be deleted."""
        self.registry.transition_task(
            task["id"], "failed", expected_version=self.version_of(task["id"])
        )

    def assert_unavailable(
        self, result: dict, reason_fragment: Optional[str] = None
    ) -> None:
        self.assertFalse(result["available"])
        self.assertIsNone(result["data_base64"])
        self.assertIsNone(result["rev"])
        self.assertIsNone(result["version"])
        self.assertIsNone(result["mime"])
        self.assertIsNone(result["bytes"])
        self.assertIsNone(result["sha256"])
        self.assertIsInstance(result["reason"], str)
        self.assertTrue(result["reason"].strip())
        self.assertNotIn(str(self.root), result["reason"])
        self.assertNotIn(str(self.preview_root), result["reason"])
        if reason_fragment is not None:
            self.assertIn(reason_fragment, result["reason"].lower())

    def assert_available(self, result: dict, expected: bytes, revision: int = 1) -> None:
        self.assertTrue(result["available"])
        self.assertEqual(result["rev"], revision)
        self.assertEqual(result["bytes"], len(expected))
        self.assertEqual(result["sha256"], hashlib.sha256(expected).hexdigest())
        self.assertIsNone(result["reason"])
        self.assertIsInstance(result["data_base64"], str)
        self.assertEqual(base64.b64decode(result["data_base64"]), expected)


class SelectedPreviewTests(PreviewTestCase):
    """Only the selected task's own image is ever returned."""

    def test_valid_preview_is_returned_as_json_safe_base64(self) -> None:
        task = self.make_task()
        self.publish(task)
        result = self.service.get_preview(task["id"])
        self.assert_available(result, PNG)
        self.assertEqual(result["task_id"], task["id"])
        self.assertEqual(result["mime"], "image/png")
        self.assertEqual(result["version"], self.version_of(task["id"]))
        # JSON-safe: it can travel through the plugin API as-is.
        self.assertEqual(json.loads(json.dumps(result)), result)

    def test_base64_roundtrip_equals_the_file_on_disk(self) -> None:
        task = self.make_task()
        path = self.publish(task, JPEG, name="rev-1.jpeg")
        result = self.service.get_preview(task["id"])
        self.assertEqual(base64.b64decode(result["data_base64"]), path.read_bytes())
        self.assertEqual(result["mime"], "image/jpeg")

    def test_revision_mime_and_size_come_from_the_persisted_record(self) -> None:
        task = self.make_task()
        self.publish(task, PNG, revision=1)
        first = self.service.get_preview(task["id"])
        self.publish(task, JPEG, revision=2, name="rev-2.jpg")
        second = self.service.get_preview(task["id"])
        self.assertEqual(first["rev"], 1)
        self.assertEqual(first["mime"], "image/png")
        self.assertEqual(second["rev"], 2)
        self.assertEqual(second["mime"], "image/jpeg")
        self.assertGreater(second["version"], first["version"])
        self.assert_available(second, JPEG, revision=2)

    def test_two_tasks_never_share_a_preview(self) -> None:
        first_task = self.make_task()
        second_task = self.make_task()
        self.publish(first_task, PNG)
        self.publish(second_task, JPEG, name="rev-1.jpg")
        first = self.service.get_preview(first_task["id"])
        second = self.service.get_preview(second_task["id"])
        self.assert_available(first, PNG)
        self.assert_available(second, JPEG)
        self.assertNotEqual(first["sha256"], second["sha256"])

    def test_webp_is_supported(self) -> None:
        task = self.make_task()
        self.publish(task, b"RIFF....WEBP", name="rev-1.webp")
        result = self.service.get_preview(task["id"])
        self.assertEqual(result["mime"], "image/webp")
        self.assertEqual(base64.b64decode(result["data_base64"]), b"RIFF....WEBP")

    def test_allowed_extensions_match_the_publish_policy(self) -> None:
        # Drift between publish and read would mean a published preview that
        # can never be served.
        self.assertEqual(set(PREVIEW_MIME_TYPES), set(tools.PREVIEW_MIME_TYPES))
        self.assertEqual(PREVIEW_MIME_TYPES, tools.PREVIEW_MIME_TYPES)
        self.assertEqual(DEFAULT_PREVIEW_MAX_BYTES, tools.DEFAULT_PREVIEW_MAX_BYTES)


class TaskScopeTests(PreviewTestCase):
    """Lookup by id, bound to the conversation that owns the task."""

    def test_missing_task_is_unavailable(self) -> None:
        self.assert_unavailable(
            self.service.get_preview("does-not-exist"), "task"
        )

    def test_malformed_task_ids_are_unavailable_not_errors(self) -> None:
        for hostile in ("", "../../etc/passwd", "..", "\x00"):
            with self.subTest(task_id=hostile):
                self.assert_unavailable(self.service.get_preview(hostile))

    def test_non_string_task_id_is_unavailable(self) -> None:
        self.assert_unavailable(self.service.get_preview(None))  # type: ignore[arg-type]

    def test_task_with_no_published_preview_is_unavailable(self) -> None:
        task = self.make_task()
        self.assert_unavailable(self.service.get_preview(task["id"]), "preview")

    def test_matching_conversation_key_returns_the_preview(self) -> None:
        task = self.make_task(conversation_key="conv-a")
        self.publish(task)
        result = self.service.get_preview(task["id"], conversation_key="conv-a")
        self.assert_available(result, PNG)

    def test_different_conversation_key_is_unavailable(self) -> None:
        task = self.make_task(conversation_key="conv-a")
        self.publish(task)
        result = self.service.get_preview(task["id"], conversation_key="conv-b")
        self.assert_unavailable(result, "conversation")
        self.assertNotIn(task["id"], result["reason"])

    def test_missing_conversation_key_is_unavailable(self) -> None:
        task = self.make_task(conversation_key="conv-a")
        self.publish(task)
        self.assert_unavailable(
            self.service.get_preview(task["id"], conversation_key="")
        )

    def test_conversation_binding_cannot_be_repointed(self) -> None:
        task = self.make_task(conversation_key="conv-a")
        self.publish(task)
        with self.assertRaises(ValueError):
            self.registry.update_task(
                task["id"],
                expected_version=task["version"],
                conversation_key="conv-b",
            )
        self.assert_available(
            self.service.get_preview(task["id"], conversation_key="conv-a"), PNG
        )

    def test_deleted_task_is_unavailable_and_falls_back_to_nothing(self) -> None:
        deleted = self.make_task()
        surviving = self.make_task()
        self.publish(deleted, PNG)
        self.publish(surviving, JPEG, name="rev-1.jpg")
        self.retire(deleted)
        self.assertTrue(self.registry.delete_task(deleted["id"]))

        result = self.service.get_preview(deleted["id"])
        self.assert_unavailable(result, "task")
        self.assertIsNone(result["task_id"])

        self.assert_available(self.service.get_preview(surviving["id"]), JPEG)

    def test_deleted_task_with_a_stale_path_still_reports_unavailable(self) -> None:
        task = self.make_task()
        path = self.publish(task)
        self.retire(task)
        self.assertTrue(self.registry.delete_task(task["id"]))
        self.assertTrue(path.exists())  # the file outlives the row
        self.assert_unavailable(self.service.get_preview(task["id"]))


class PathValidationTests(PreviewTestCase):
    """Containment, regular-file status and the extension allowlist."""

    def test_missing_file_is_unavailable(self) -> None:
        task = self.make_task()
        path = self.publish(task)
        path.unlink()
        self.assert_unavailable(self.service.get_preview(task["id"]), "missing")

    def test_empty_file_is_unavailable(self) -> None:
        task = self.make_task()
        self.publish(task, b"")
        self.assert_unavailable(self.service.get_preview(task["id"]), "empty")

    def test_oversize_file_is_unavailable(self) -> None:
        task = self.make_task()
        self.publish(task, b"x" * 32)
        service = PreviewService(
            self.registry, preview_root=self.preview_root, max_bytes=16
        )
        self.assert_unavailable(service.get_preview(task["id"]), "limit")

    def test_size_equal_to_the_cap_is_allowed(self) -> None:
        task = self.make_task()
        self.publish(task, b"x" * 16)
        service = PreviewService(
            self.registry, preview_root=self.preview_root, max_bytes=16
        )
        result = service.get_preview(task["id"])
        self.assert_available(result, b"x" * 16)

    def test_unsupported_extension_is_unavailable(self) -> None:
        task = self.make_task()
        self.publish(task, b"GIF89a", name="rev-1.gif")
        self.assert_unavailable(self.service.get_preview(task["id"]), "type")

    def test_missing_extension_is_unavailable(self) -> None:
        task = self.make_task()
        self.publish(task, PNG, name="rev-1")
        self.assert_unavailable(self.service.get_preview(task["id"]), "type")

    def test_uppercase_extension_is_not_on_the_allowlist(self) -> None:
        # The allowlist names the four extensions exactly as the publisher
        # writes them; anything else is outside the policy.
        task = self.make_task()
        self.publish(task, PNG, name="rev-1.PNG")
        self.assert_unavailable(self.service.get_preview(task["id"]), "type")

    def test_path_and_revision_that_disagree_are_refused(self) -> None:
        task = self.make_task()
        path = self.publish(task, PNG, revision=1)
        self.registry.update_task(
            task["id"],
            expected_version=self.version_of(task["id"]),
            preview_path=str(path),
            preview_rev=7,
        )
        self.assert_unavailable(self.service.get_preview(task["id"]), "revision")

    def test_symlink_inside_the_root_is_still_refused(self) -> None:
        task = self.make_task()
        real = self.publish(task, PNG)
        link = real.parent / "rev-2.png"
        link.symlink_to(real)
        self.registry.update_task(
            task["id"],
            expected_version=self.version_of(task["id"]),
            preview_path=str(link),
            preview_rev=2,
        )
        self.assert_unavailable(self.service.get_preview(task["id"]), "symbolic")

    def test_symlink_file_is_refused(self) -> None:
        task = self.make_task()
        outside = self.root / "outside.png"
        outside.write_bytes(PNG)
        link = self.preview_root / task["id"] / "rev-1.png"
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(outside)
        self.registry.update_task(
            task["id"],
            expected_version=self.version_of(task["id"]),
            preview_path=str(link),
            preview_rev=1,
        )
        self.assert_unavailable(self.service.get_preview(task["id"]), "symbolic")

    def test_symlinked_parent_directory_is_refused(self) -> None:
        task = self.make_task()
        real = self.preview_root / "elsewhere"
        real.mkdir()
        (real / "rev-1.png").write_bytes(PNG)
        link_dir = self.preview_root / task["id"]
        link_dir.symlink_to(real, target_is_directory=True)
        self.registry.update_task(
            task["id"],
            expected_version=self.version_of(task["id"]),
            preview_path=str(link_dir / "rev-1.png"),
            preview_rev=1,
        )
        self.assert_unavailable(self.service.get_preview(task["id"]), "symbolic")

    def test_traversal_out_of_the_preview_root_is_refused(self) -> None:
        task = self.make_task()
        secret = self.root / "secret.png"
        secret.write_bytes(PNG)
        self.registry.update_task(
            task["id"],
            expected_version=self.version_of(task["id"]),
            preview_path=str(self.preview_root / ".." / "secret.png"),
            preview_rev=1,
        )
        self.assert_unavailable(self.service.get_preview(task["id"]), "outside")

    def test_absolute_path_outside_the_preview_root_is_refused(self) -> None:
        task = self.make_task()
        secret = self.root / "secret.png"
        secret.write_bytes(PNG)
        self.registry.update_task(
            task["id"],
            expected_version=self.version_of(task["id"]),
            preview_path=str(secret),
            preview_rev=1,
        )
        self.assert_unavailable(self.service.get_preview(task["id"]), "outside")

    def test_relative_persisted_path_is_refused(self) -> None:
        task = self.make_task()
        self.publish(task)
        current = self.registry.get_task(task["id"])
        self.registry.update_task(
            task["id"],
            expected_version=current["version"],
            preview_path="%s/rev-1.png" % task["id"],
            preview_rev=1,
        )
        self.assert_unavailable(self.service.get_preview(task["id"]), "invalid")

    def test_the_preview_root_itself_is_refused(self) -> None:
        task = self.make_task()
        self.registry.update_task(
            task["id"],
            expected_version=self.version_of(task["id"]),
            preview_path=str(self.preview_root),
            preview_rev=1,
        )
        self.assert_unavailable(self.service.get_preview(task["id"]))

    def test_directory_in_place_of_a_preview_is_refused(self) -> None:
        task = self.make_task()
        directory = self.preview_root / task["id"] / "rev-1.png"
        directory.mkdir(parents=True)
        self.registry.update_task(
            task["id"],
            expected_version=self.version_of(task["id"]),
            preview_path=str(directory),
            preview_rev=1,
        )
        self.assert_unavailable(self.service.get_preview(task["id"]), "regular")

    def test_non_string_persisted_path_is_refused(self) -> None:
        task = self.make_task()
        self.registry.update_task(
            task["id"],
            expected_version=self.version_of(task["id"]),
            preview_path=123,
            preview_rev=1,
        )
        self.assert_unavailable(self.service.get_preview(task["id"]))

    def test_missing_zero_revision_is_unavailable(self) -> None:
        task = self.make_task()
        path = self.publish(task)
        self.registry.update_task(
            task["id"],
            expected_version=self.version_of(task["id"]),
            preview_rev=0,
        )
        self.assertTrue(path.exists())
        self.assert_unavailable(self.service.get_preview(task["id"]), "revision")

    def test_storage_root_that_does_not_exist_is_unavailable(self) -> None:
        task = self.make_task()
        self.publish(task)
        service = PreviewService(
            self.registry, preview_root=self.root / "absent-previews"
        )
        self.assert_unavailable(service.get_preview(task["id"]))


class SnapshotConsistencyTests(PreviewTestCase):
    """A concurrent publish never produces a revision/content mismatch."""

    def test_concurrent_replacement_retries_and_reports_matching_bytes(self) -> None:
        task = self.make_task()
        self.publish(task, PNG, revision=1)
        newer = b"NEWER-REVISION-BYTES" * 3
        replaced = []
        original = PreviewService._read_snapshot

        def racing_read(service_self, path, *, want_data=True):
            if not replaced:
                replaced.append(True)
                self.publish(task, newer, revision=2, name="rev-2.png")
            return original(service_self, path, want_data=want_data)

        with unittest.mock.patch.object(
            PreviewService, "_read_snapshot", racing_read
        ):
            result = self.service.get_preview(task["id"])

        self.assertTrue(replaced)
        self.assert_available(result, newer, revision=2)
        self.assertEqual(result["mime"], "image/png")
        self.assertNotEqual(
            result["sha256"], hashlib.sha256(PNG).hexdigest()
        )

    def test_endless_replacement_gives_an_honest_unavailable_result(self) -> None:
        task = self.make_task()
        self.publish(task, PNG, revision=1)
        original = PreviewService._read_snapshot
        counter = {"n": 1}

        def always_racing(service_self, path, *, want_data=True):
            counter["n"] += 1
            self.publish(
                task,
                b"revision-%d" % counter["n"],
                revision=counter["n"],
                name="rev-%d.png" % counter["n"],
            )
            return original(service_self, path, want_data=want_data)

        with unittest.mock.patch.object(
            PreviewService, "_read_snapshot", always_racing
        ):
            result = self.service.get_preview(task["id"])

        self.assert_unavailable(result, "changed")

    def test_parallel_readers_never_mix_revisions(self) -> None:
        task = self.make_task()
        contents = {1: PNG}
        for revision in range(2, 6):
            contents[revision] = (
                b"revision-%d-" % revision + bytes([64 + revision]) * (32 * revision)
            )
        expected_sha = {
            revision: hashlib.sha256(data).hexdigest()
            for revision, data in contents.items()
        }
        self.publish(task, contents[1], revision=1)

        stop = threading.Event()
        results = []
        lock = threading.Lock()

        def reader() -> None:
            while not stop.is_set():
                result = self.service.get_preview(task["id"])
                with lock:
                    results.append(result)
                time.sleep(0.001)

        def publisher() -> None:
            try:
                for revision in range(2, 6):
                    time.sleep(0.004)
                    self.publish(
                        task,
                        contents[revision],
                        revision=revision,
                        name="rev-%d.png" % revision,
                    )
            finally:
                stop.set()

        with ThreadPoolExecutor(max_workers=5) as pool:
            readers = [pool.submit(reader) for _ in range(4)]
            pool.submit(publisher).result()
            for future in readers:
                future.result()

        self.assertTrue(results)
        served = 0
        for result in results:
            if not result["available"]:
                # The only acceptable refusal under churn is the honest one.
                self.assertIn("changed", result["reason"].lower())
                continue
            served += 1
            content = contents[result["rev"]]
            self.assertEqual(result["sha256"], expected_sha[result["rev"]])
            self.assertEqual(result["bytes"], len(content))
            self.assertEqual(base64.b64decode(result["data_base64"]), content)
        self.assertGreater(served, 0)

        final = self.service.get_preview(task["id"])
        self.assert_available(final, contents[5], revision=5)

    def test_descriptor_reports_the_same_revision_as_the_bytes(self) -> None:
        task = self.make_task()
        self.publish(task, PNG, revision=1)
        self.publish(task, JPEG, revision=2, name="rev-2.jpg")
        body = self.service.get_preview(task["id"])
        descriptor = self.service.preview_descriptor(task["id"])
        self.assertTrue(descriptor["available"])
        self.assertEqual(descriptor["rev"], body["rev"])
        self.assertEqual(descriptor["version"], body["version"])
        self.assertEqual(descriptor["mime"], body["mime"])
        self.assertEqual(descriptor["bytes"], body["bytes"])
        self.assertEqual(descriptor["sha256"], body["sha256"])
        self.assertIsNone(descriptor["data_base64"])


class DescriptorTests(PreviewTestCase):
    """Metadata for API headers, without shipping the image."""

    def test_descriptor_carries_no_image_data(self) -> None:
        task = self.make_task()
        self.publish(task, PNG)
        descriptor = self.service.preview_descriptor(task["id"])
        self.assertEqual(descriptor["task_id"], task["id"])
        self.assertIsNone(descriptor["data_base64"])
        self.assertEqual(descriptor["bytes"], len(PNG))
        self.assertEqual(json.loads(json.dumps(descriptor)), descriptor)

    def test_descriptor_for_missing_task_is_unavailable(self) -> None:
        self.assert_unavailable(self.service.preview_descriptor("nope"))

    def test_descriptor_honours_the_conversation_binding(self) -> None:
        task = self.make_task(conversation_key="conv-a")
        self.publish(task)
        self.assertTrue(
            self.service.preview_descriptor(
                task["id"], conversation_key="conv-a"
            )["available"]
        )
        self.assert_unavailable(
            self.service.preview_descriptor(task["id"], conversation_key="conv-b")
        )


class UnavailableResultTests(PreviewTestCase):
    """Every refusal is a clean, path-free JSON object."""

    def test_unavailable_result_shape(self) -> None:
        result = self.service.unavailable("no preview has been published")
        self.assertFalse(result["available"])
        self.assertIsNone(result["data_base64"])
        self.assertEqual(result["reason"], "no preview has been published")
        self.assertEqual(json.loads(json.dumps(result)), result)

    def test_unavailable_scrubs_absolute_paths_from_a_hostile_reason(self) -> None:
        hostile = "cannot read %s/%s/rev-1.png" % (self.preview_root, "abc")
        result = self.service.unavailable(hostile)
        self.assertNotIn(str(self.root), result["reason"])
        self.assertNotIn(str(self.preview_root), result["reason"])
        self.assertNotIn("rev-1.png", result["reason"])
        self.assertIn("preview", result["reason"].lower())

    def test_unavailable_forgives_a_non_string_reason(self) -> None:
        result = self.service.unavailable(None)  # type: ignore[arg-type]
        self.assertFalse(result["available"])
        self.assertIsInstance(result["reason"], str)
        self.assertTrue(result["reason"])

    def test_unavailable_never_carries_bytes(self) -> None:
        result = self.service.unavailable("preview file is missing")
        self.assertIsNone(result["bytes"])
        self.assertIsNone(result["sha256"])
        self.assertIsNone(result["data_base64"])


class ConfigurationTests(PreviewTestCase):
    """Construction refuses nonsense, and defaults follow the publish policy."""

    def test_default_max_bytes_is_two_mebibytes(self) -> None:
        service = PreviewService(self.registry, preview_root=self.preview_root)
        self.assertEqual(service.max_bytes, 2 * 1024 * 1024)
        self.assertEqual(service.max_bytes, DEFAULT_PREVIEW_MAX_BYTES)

    def test_byte_cap_must_be_a_positive_integer(self) -> None:
        for bad in (0, -1, True, 1.5, "1024"):
            with self.subTest(max_bytes=bad):
                with self.assertRaises(ValueError):
                    PreviewService(
                        self.registry,
                        preview_root=self.preview_root,
                        max_bytes=bad,
                    )

    def test_registry_must_expose_get_task(self) -> None:
        with self.assertRaises(TypeError):
            PreviewService(object(), preview_root=self.preview_root)

    def test_default_root_follows_the_publish_environment(self) -> None:
        with unittest.mock.patch.dict(
            os.environ, {"NEO_PREVIEW_DIR": str(self.root / "custom-previews")}
        ):
            service = PreviewService(self.registry)
        self.assertEqual(
            service.preview_root, (self.root / "custom-previews").resolve()
        )

    def test_preview_root_is_exposed_as_a_resolved_path(self) -> None:
        self.assertTrue(self.service.preview_root.is_absolute())
        self.assertEqual(self.service.preview_root, self.preview_root.resolve())


class NoBinaryResponseTests(PreviewTestCase):
    """The API layer must never need a binary body to serve a preview."""

    def test_result_contains_only_json_values(self) -> None:
        task = self.make_task()
        self.publish(task)
        result = self.service.get_preview(task["id"])

        def walk(value):
            if isinstance(value, dict):
                for item in value.values():
                    walk(item)
            elif isinstance(value, (list, tuple)):
                for item in value:
                    walk(item)
            else:
                self.assertNotIsInstance(value, (bytes, bytearray, Path))

        walk(result)
        self.assertIsInstance(result["data_base64"], str)

    def test_module_assumes_no_binary_response_path(self) -> None:
        source = Path(previews.__file__).read_text(encoding="utf-8")
        for forbidden in ("ctx.rest", "send_file", "FileResponse", "StreamingResponse"):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
