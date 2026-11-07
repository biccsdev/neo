"""Task 6 — artifact verification and presentation policy.

Two rules from the plan are proven here. An artifact is only real when it is a
regular file strictly beneath the task workdir, reached without a symlink and
with an extension NEO knows how to open. Presentation is a policy decision, not
a side effect: only a completed, evidence-backed task may open its primary
verified artifact, at most once per task version, and opening anything is never
proof that the work succeeded.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import threading
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import artifacts  # noqa: E402
from artifacts import (  # noqa: E402
    DEFAULT_ALLOWED_EXTENSIONS,
    ArtifactError,
    ArtifactEscapesWorkdirError,
    ArtifactNotFoundError,
    ArtifactNotRegularFileError,
    EmptyExpectedArtifactsError,
    MissingWorkdirError,
    UnsupportedArtifactTypeError,
    completion_evidence,
    verify_artifact,
    verify_artifacts,
)
from presentation import PresentationResult, PresentationService  # noqa: E402
from state_machine import CompletionEvidence  # noqa: E402


class ArtifactTestCase(unittest.TestCase):
    """Shared fixtures: one temp root, one task workdir beneath it."""

    def setUp(self) -> None:
        self._tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tempdir.cleanup)
        self.root = Path(self._tempdir.name)
        self.workdir = self.root / "task"
        self.workdir.mkdir()

    def write(self, name: str, text: str = "done") -> Path:
        path = self.workdir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        return path

    def assert_safe_message(self, error: BaseException) -> None:
        """A refusal must never echo the absolute workdir back at the caller."""
        message = str(error)
        self.assertNotIn(str(self.workdir), message)
        self.assertNotIn(str(self.root), message)


class ArtifactVerificationTests(ArtifactTestCase):
    def test_regular_file_beneath_workdir_is_returned_resolved(self) -> None:
        self.write("reports/summary.md", "# summary")
        artifact = verify_artifact(self.workdir, "reports/summary.md")
        self.assertEqual(artifact.name, "reports/summary.md")
        self.assertTrue(artifact.path.is_absolute())
        self.assertEqual(artifact.path, (self.workdir / "reports/summary.md").resolve())
        self.assertEqual(artifact.path, artifact.path.resolve())

    def test_leading_dot_segments_are_canonicalised(self) -> None:
        self.write("page.html", "<h1>hi</h1>")
        artifact = verify_artifact(self.workdir, "./page.html")
        self.assertEqual(artifact.name, "page.html")
        self.assertEqual(artifact.path.name, "page.html")

    def test_verify_artifacts_preserves_requested_order(self) -> None:
        self.write("b.md")
        self.write("a.png")
        verified = verify_artifacts(self.workdir, ["b.md", "a.png"])
        self.assertEqual([artifact.name for artifact in verified], ["b.md", "a.png"])

    def test_missing_workdir_is_refused(self) -> None:
        with self.assertRaises(MissingWorkdirError) as caught:
            verify_artifact(None, "a.md")
        self.assert_safe_message(caught.exception)
        with self.assertRaises(MissingWorkdirError):
            verify_artifact("", "a.md")
        with self.assertRaises(MissingWorkdirError):
            verify_artifact(self.root / "absent", "a.md")

    def test_workdir_that_is_a_file_is_refused(self) -> None:
        not_a_dir = self.root / "plain.txt"
        not_a_dir.write_text("x")
        with self.assertRaises(MissingWorkdirError):
            verify_artifact(not_a_dir, "a.md")

    def test_missing_artifact_is_refused(self) -> None:
        with self.assertRaises(ArtifactNotFoundError) as caught:
            verify_artifact(self.workdir, "missing.png")
        self.assert_safe_message(caught.exception)
        self.assertIn("missing.png", str(caught.exception))

    def test_directory_is_refused_even_with_an_allowed_extension(self) -> None:
        (self.workdir / "bundle.html").mkdir()
        with self.assertRaises(ArtifactNotRegularFileError) as caught:
            verify_artifact(self.workdir, "bundle.html")
        self.assertIn("directory", str(caught.exception))

    def test_named_pipe_is_refused(self) -> None:
        fifo = self.workdir / "stream.txt"
        try:
            os.mkfifo(fifo)
        except (AttributeError, OSError):
            self.skipTest("named pipes are not available on this platform")
        with self.assertRaises(ArtifactNotRegularFileError) as caught:
            verify_artifact(self.workdir, "stream.txt")
        self.assertIn("pipe", str(caught.exception))

    def test_absolute_path_is_refused(self) -> None:
        target = self.write("outside.md")
        with self.assertRaises(ArtifactEscapesWorkdirError) as caught:
            verify_artifact(self.workdir, str(target))
        self.assert_safe_message(caught.exception)

    def test_relative_traversal_is_refused(self) -> None:
        (self.root / "escape.md").write_text("secret")
        with self.assertRaises(ArtifactEscapesWorkdirError):
            verify_artifact(self.workdir, "../escape.md")
        with self.assertRaises(ArtifactEscapesWorkdirError):
            verify_artifact(self.workdir, "reports/../../escape.md")

    def test_inner_dot_dot_is_refused_even_when_it_stays_inside(self) -> None:
        self.write("a.md")
        with self.assertRaises(ArtifactEscapesWorkdirError):
            verify_artifact(self.workdir, "sub/../a.md")

    def test_symlink_to_outside_file_is_refused(self) -> None:
        outside = self.root / "outside.md"
        outside.write_text("secret")
        (self.workdir / "link.md").symlink_to(outside)
        with self.assertRaises(ArtifactEscapesWorkdirError) as caught:
            verify_artifact(self.workdir, "link.md")
        self.assert_safe_message(caught.exception)

    def test_symlink_to_inside_file_is_refused_as_a_component(self) -> None:
        self.write("real.md")
        (self.workdir / "alias.md").symlink_to(self.workdir / "real.md")
        with self.assertRaises(ArtifactEscapesWorkdirError):
            verify_artifact(self.workdir, "alias.md")

    def test_symlinked_directory_component_is_refused(self) -> None:
        outside = self.root / "outside"
        outside.mkdir()
        (outside / "page.html").write_text("<p>x</p>")
        (self.workdir / "linked").symlink_to(outside, target_is_directory=True)
        with self.assertRaises(ArtifactEscapesWorkdirError):
            verify_artifact(self.workdir, "linked/page.html")

    def test_symlinked_workdir_itself_is_allowed_when_it_resolves(self) -> None:
        self.write("a.md")
        alias = self.root / "task-alias"
        alias.symlink_to(self.workdir, target_is_directory=True)
        artifact = verify_artifact(alias, "a.md")
        self.assertEqual(artifact.path, (self.workdir / "a.md").resolve())

    def test_workdir_itself_is_not_an_artifact(self) -> None:
        with self.assertRaises(ArtifactError):
            verify_artifact(self.workdir, ".")

    def test_final_component_must_be_a_file_not_a_symlink(self) -> None:
        self.write("real.md")
        (self.workdir / "final.md").symlink_to(self.workdir / "real.md")
        with self.assertRaises(ArtifactEscapesWorkdirError):
            verify_artifact(self.workdir, "final.md")

    def test_intermediate_component_that_is_a_file_is_refused(self) -> None:
        self.write("report.md")
        with self.assertRaises(ArtifactNotFoundError):
            verify_artifact(self.workdir, "report.md/inner.md")

    def test_url_like_names_are_refused(self) -> None:
        with self.assertRaises(ArtifactError) as caught:
            verify_artifact(self.workdir, "https://example.com/report.md")
        self.assertIn("URL", str(caught.exception))

    def test_default_allowlist_covers_neo_result_families(self) -> None:
        for name in (
            "report.pdf",
            "notes.md",
            "app.py",
            "index.html",
            "render.png",
            "clip.mp4",
            "voice.mp3",
            "scene.blend",
        ):
            with self.subTest(name=name):
                self.write(name)
                artifact = verify_artifact(self.workdir, name)
                self.assertEqual(artifact.name, name)
        self.assertIn(".blend", DEFAULT_ALLOWED_EXTENSIONS)
        self.assertNotIn(".exe", DEFAULT_ALLOWED_EXTENSIONS)

    def test_extension_matching_is_case_insensitive(self) -> None:
        self.write("SCREEN.PNG")
        self.write("Scene.BLEND")
        self.assertEqual(verify_artifact(self.workdir, "SCREEN.PNG").name, "SCREEN.PNG")
        self.assertEqual(verify_artifact(self.workdir, "Scene.BLEND").name, "Scene.BLEND")

    def test_unexpected_extension_is_refused(self) -> None:
        self.write("payload.exe")
        with self.assertRaises(UnsupportedArtifactTypeError) as caught:
            verify_artifact(self.workdir, "payload.exe")
        self.assertIn(".exe", str(caught.exception))
        self.write("script")
        with self.assertRaises(UnsupportedArtifactTypeError):
            verify_artifact(self.workdir, "script")

    def test_caller_supplied_allowlist_extends_or_replaces_the_default(self) -> None:
        self.write("bundle.xyz")
        with self.assertRaises(UnsupportedArtifactTypeError):
            verify_artifact(self.workdir, "bundle.xyz")
        artifact = verify_artifact(
            self.workdir, "bundle.xyz", allowed_extensions={".xyz"}
        )
        self.assertEqual(artifact.name, "bundle.xyz")
        self.write("page.html")
        with self.assertRaises(UnsupportedArtifactTypeError):
            verify_artifact(self.workdir, "page.html", allowed_extensions={"xyz"})

    def test_allowlist_entries_are_normalised(self) -> None:
        self.write("bundle.xyz")
        self.assertEqual(
            verify_artifact(self.workdir, "bundle.xyz", allowed_extensions=["XYZ"]).name,
            "bundle.xyz",
        )

    def test_default_allowlist_is_immutable_and_lowercase(self) -> None:
        self.assertIsInstance(DEFAULT_ALLOWED_EXTENSIONS, frozenset)
        self.assertTrue(all(entry == entry.lower() for entry in DEFAULT_ALLOWED_EXTENSIONS))
        self.assertTrue(all(entry.endswith(".") is False for entry in DEFAULT_ALLOWED_EXTENSIONS))


class CompletionEvidenceTests(ArtifactTestCase):
    def test_verified_artifacts_become_executor_confirmed_evidence(self) -> None:
        self.write("index.html", "<h1>done</h1>")
        self.write("shot.png")
        evidence = completion_evidence(
            self.workdir, ["./index.html", "shot.png"]
        )
        self.assertIsInstance(evidence, CompletionEvidence)
        self.assertTrue(evidence.executor_confirmed)
        self.assertEqual(evidence.expected_artifacts, ("index.html", "shot.png"))
        self.assertEqual(evidence.verified_artifacts, ("index.html", "shot.png"))
        self.assertEqual(evidence.missing_artifacts(), ())

    def test_missing_artifact_prevents_evidence(self) -> None:
        self.write("index.html")
        with self.assertRaises(ArtifactNotFoundError):
            completion_evidence(self.workdir, ["index.html", "absent.png"])

    def test_symlink_escape_prevents_evidence(self) -> None:
        outside = self.root / "outside.md"
        outside.write_text("secret")
        (self.workdir / "link.md").symlink_to(outside)
        with self.assertRaises(ArtifactEscapesWorkdirError):
            completion_evidence(self.workdir, ["link.md"])

    def test_unsupported_type_prevents_evidence(self) -> None:
        self.write("payload.bin")
        with self.assertRaises(UnsupportedArtifactTypeError):
            completion_evidence(self.workdir, ["payload.bin"])

    def test_empty_expectation_requires_an_explicit_opt_in(self) -> None:
        with self.assertRaises(EmptyExpectedArtifactsError):
            completion_evidence(self.workdir, [])
        with self.assertRaises(EmptyExpectedArtifactsError):
            completion_evidence(self.workdir, None)
        evidence = completion_evidence(self.workdir, [], allow_empty=True)
        self.assertTrue(evidence.executor_confirmed)
        self.assertEqual(tuple(evidence.expected_artifacts), ())
        self.assertEqual(tuple(evidence.verified_artifacts), ())

    def test_completion_evidence_passes_the_allowlist_through(self) -> None:
        self.write("bundle.xyz")
        evidence = completion_evidence(
            self.workdir, ["bundle.xyz"], allowed_extensions={".xyz"}
        )
        self.assertEqual(evidence.verified_artifacts, ("bundle.xyz",))


def completed_task(
    workdir: Path,
    *,
    task_id: str = "task-1",
    version: int = 2,
    state: str = "completed",
    expected=None,
    verified=None,
) -> dict:
    expected_names = ["report.md"] if expected is None else list(expected)
    verified_names = expected_names if verified is None else list(verified)
    return {
        "id": task_id,
        "state": state,
        "version": version,
        "workdir": str(workdir),
        "expected_json": json.dumps(expected_names),
        "artifacts_json": json.dumps(verified_names),
    }


class PresentationPolicyTests(ArtifactTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.opened: list[str] = []

    def opener(self, path: str) -> None:
        self.opened.append(path)

    def service(self, opener=None, **kwargs) -> PresentationService:
        return PresentationService(opener or self.opener, **kwargs)

    def test_completed_verified_task_opens_primary_artifact_once(self) -> None:
        self.write("report.md", "# done")
        self.write("shot.png")
        task = completed_task(self.workdir, expected=["report.md", "shot.png"])
        result = self.service().present(task)
        self.assertTrue(result.presented)
        self.assertEqual(result.path, str((self.workdir / "report.md").resolve()))
        self.assertEqual(result.task_id, "task-1")
        self.assertEqual(result.version, 2)
        self.assertEqual(self.opened, [str((self.workdir / "report.md").resolve())])

    def test_primary_artifact_is_the_first_expected_entry(self) -> None:
        self.write("a.png")
        self.write("b.md")
        task = completed_task(self.workdir, expected=["b.md", "a.png"])
        result = self.service().present(task)
        self.assertTrue(result.presented)
        self.assertEqual(result.path, str((self.workdir / "b.md").resolve()))
        self.assertEqual(len(self.opened), 1)

    def test_incomplete_states_never_open(self) -> None:
        self.write("report.md")
        for state in ("queued", "running", "waiting", "needs_input", "cancelling", "failed", "cancelled"):
            with self.subTest(state=state):
                task = completed_task(self.workdir, state=state)
                result = self.service().present(task)
                self.assertFalse(result.presented)
                self.assertIsNone(result.path)
                self.assertIn(state, result.reason)
        self.assertEqual(self.opened, [])

    def test_missing_expected_artifact_never_opens(self) -> None:
        task = completed_task(self.workdir)
        result = self.service().present(task)
        self.assertFalse(result.presented)
        self.assertIn("report.md", result.reason)
        self.assertEqual(self.opened, [])

    def test_artifact_that_escaped_after_completion_never_opens(self) -> None:
        outside = self.root / "outside.md"
        outside.write_text("secret")
        (self.workdir / "report.md").symlink_to(outside)
        task = completed_task(self.workdir)
        result = self.service().present(task)
        self.assertFalse(result.presented)
        self.assertEqual(self.opened, [])

    def test_unverified_artifact_is_refused_even_when_the_file_exists(self) -> None:
        self.write("report.md")
        task = completed_task(self.workdir, expected=["report.md"], verified=[])
        result = self.service().present(task)
        self.assertFalse(result.presented)
        self.assertIn("report.md", result.reason)
        self.assertEqual(self.opened, [])

    def test_no_expected_artifacts_is_a_clean_skip(self) -> None:
        task = completed_task(self.workdir, expected=[], verified=[])
        result = self.service().present(task)
        self.assertFalse(result.presented)
        self.assertTrue(result.reason)
        self.assertEqual(self.opened, [])

    def test_unreadable_evidence_is_a_skip_not_a_crash(self) -> None:
        self.write("report.md")
        task = completed_task(self.workdir)
        task["expected_json"] = "not json"
        result = self.service().present(task)
        self.assertFalse(result.presented)
        self.assertEqual(self.opened, [])
        task["expected_json"] = json.dumps([1, 2])
        self.assertFalse(self.service().present(task).presented)

    def test_missing_workdir_is_a_skip(self) -> None:
        task = completed_task(self.workdir)
        task["workdir"] = None
        result = self.service().present(task)
        self.assertFalse(result.presented)
        self.assertEqual(self.opened, [])

    def test_missing_version_is_a_skip(self) -> None:
        self.write("report.md")
        task = completed_task(self.workdir)
        del task["version"]
        result = self.service().present(task)
        self.assertFalse(result.presented)
        self.assertEqual(self.opened, [])

    def test_same_task_version_presents_only_once(self) -> None:
        self.write("report.md")
        task = completed_task(self.workdir)
        service = self.service()
        first = service.present(task)
        second = service.present(task)
        self.assertTrue(first.presented)
        self.assertFalse(second.presented)
        self.assertIn("already presented", second.reason)
        self.assertEqual(len(self.opened), 1)

    def test_a_new_version_may_present_again(self) -> None:
        self.write("report.md")
        service = self.service()
        self.assertTrue(service.present(completed_task(self.workdir, version=2)).presented)
        self.assertTrue(service.present(completed_task(self.workdir, version=3)).presented)
        self.assertEqual(len(self.opened), 2)

    def test_a_different_task_is_independent(self) -> None:
        self.write("report.md")
        service = self.service()
        self.assertTrue(service.present(completed_task(self.workdir, task_id="a")).presented)
        self.assertTrue(service.present(completed_task(self.workdir, task_id="b")).presented)
        self.assertEqual(len(self.opened), 2)

    def test_concurrent_calls_open_at_most_once_per_version(self) -> None:
        self.write("report.md")
        task = completed_task(self.workdir, version=7)
        calls: list[str] = []
        gate = threading.Barrier(8)

        def slow_opener(path: str) -> None:
            calls.append(path)
            threading.Event().wait(0.02)

        service = self.service(slow_opener)
        results: list[PresentationResult] = []

        def worker() -> None:
            gate.wait()
            results.append(service.present(task))

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual(len(calls), 1)
        self.assertEqual(sum(1 for result in results if result.presented), 1)

    def test_failed_opener_is_not_marked_presented_and_can_retry(self) -> None:
        self.write("report.md")
        task = completed_task(self.workdir)
        attempts: list[str] = []
        failures = [OSError("no handler")]

        def flaky_opener(path: str) -> None:
            attempts.append(path)
            if failures:
                raise failures.pop()

        service = self.service(flaky_opener)
        first = service.present(task)
        self.assertFalse(first.presented)
        self.assertIn("opener failed", first.reason)
        second = service.present(task)
        self.assertTrue(second.presented)
        self.assertEqual(len(attempts), 2)
        third = service.present(task)
        self.assertFalse(third.presented)
        self.assertIn("already presented", third.reason)
        self.assertEqual(len(attempts), 2)

    def test_presentation_never_mutates_the_task_record(self) -> None:
        self.write("report.md")
        task = completed_task(self.workdir)
        snapshot = dict(task)
        self.service().present(task)
        self.assertEqual(task, snapshot)

    def test_service_accepts_no_publication_callbacks(self) -> None:
        with self.assertRaises(TypeError):
            PresentationService(self.opener, publish=lambda *args: None)
        with self.assertRaises(TypeError):
            PresentationService(self.opener, send=lambda *args: None)
        service = self.service()
        for attribute in ("publish", "send", "post", "open_url", "browser"):
            self.assertFalse(hasattr(service, attribute))
        with self.assertRaises(TypeError):
            PresentationService("not callable")  # type: ignore[arg-type]

    def test_url_like_expected_artifact_is_never_opened(self) -> None:
        task = completed_task(
            self.workdir,
            expected=["https://example.com/report.md"],
            verified=["https://example.com/report.md"],
        )
        result = self.service().present(task)
        self.assertFalse(result.presented)
        self.assertEqual(self.opened, [])

    def test_caller_allowlist_is_honoured(self) -> None:
        self.write("bundle.xyz")
        task = completed_task(self.workdir, expected=["bundle.xyz"])
        self.assertFalse(self.service().present(task).presented)
        self.assertEqual(self.opened, [])
        result = self.service(allowed_extensions={".xyz"}).present(task)
        self.assertTrue(result.presented)
        self.assertEqual(len(self.opened), 1)

    def test_result_is_a_small_dict_like_record(self) -> None:
        result = PresentationResult(presented=False, reason="skipped", task_id="t", version=1)
        self.assertEqual(
            result.as_dict(),
            {"presented": False, "path": None, "reason": "skipped", "task_id": "t", "version": 1},
        )


class PersistedCompletionIntegrationTests(ArtifactTestCase):
    """The evidence helper's names must survive a real registry round trip."""

    def test_registry_persisted_evidence_presents_the_primary_artifact(self) -> None:
        from registry import Registry  # imported here so path setup runs first

        self.write("index.html", "<h1>done</h1>")
        self.write("shot.png")
        registry = Registry(path=self.root / "neo.db")
        task = registry.create_task(
            conversation_key="conversation",
            kind="coding",
            intent="build the page",
            workdir=str(self.workdir),
        )
        running = registry.transition_task(
            task["id"], "running", expected_version=task["version"]
        )
        evidence = completion_evidence(self.workdir, ["index.html", "shot.png"])
        completed = registry.transition_task(
            task["id"],
            "completed",
            expected_version=running["version"],
            evidence=evidence,
        )
        self.assertEqual(completed["state"], "completed")
        self.assertEqual(
            json.loads(completed["expected_json"]), ["index.html", "shot.png"]
        )
        self.assertEqual(
            json.loads(completed["artifacts_json"]), ["index.html", "shot.png"]
        )
        opener: list = []
        result = PresentationService(opener.append).present(completed)
        self.assertTrue(result.presented)
        self.assertEqual(result.path, str((self.workdir / "index.html").resolve()))
        self.assertEqual(len(opener), 1)


if __name__ == "__main__":
    unittest.main()
