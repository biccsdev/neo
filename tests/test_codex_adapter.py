"""Task 9 — Codex adapter: argv contracts and exact-thread JSONL parsing."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from executors.base import (  # noqa: E402
    ExecutorConfigError,
    ExecutorResult,
    validate_argv,
)
from executors.codex import CodexExecutor  # noqa: E402

FIXTURES = REPO_ROOT / "fixtures" / "executors"
WORKDIR = "/Users/user/code/work/demo"
THREAD = "01a1b80e-039e-78c1-87bf-555defb84f3a"


def read_lines(name: str) -> list:
    return (FIXTURES / name).read_text(encoding="utf-8").splitlines()


class SharedBaseContractTests(unittest.TestCase):
    """The base result type refuses states it cannot honestly represent."""

    def test_unknown_status_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            ExecutorResult(status="done", session_id=THREAD)

    def test_completed_requires_a_session_id(self) -> None:
        with self.assertRaises(ValueError):
            ExecutorResult(status="completed", final_message="OK")

    def test_failed_requires_error_evidence(self) -> None:
        with self.assertRaises(ValueError):
            ExecutorResult(status="failed")

    def test_argv_must_be_an_array_of_plain_tokens(self) -> None:
        for command in ("codex exec --json", "", [], None, ["codex", ""], ["codex", 3]):
            with self.subTest(command=command):
                with self.assertRaises(ValueError):
                    validate_argv(command)

    def test_argv_validation_returns_a_plain_list(self) -> None:
        argv = validate_argv(("codex", "exec"))
        self.assertIsInstance(argv, list)
        self.assertEqual(argv, ["codex", "exec"])


class CodexArgvTests(unittest.TestCase):
    def setUp(self) -> None:
        self.executor = CodexExecutor()

    def test_initial_argv_is_the_exact_spec_array(self) -> None:
        argv = self.executor.build_initial_argv("Say OK", workdir=WORKDIR)
        self.assertEqual(
            argv,
            [
                "codex",
                "exec",
                "--json",
                "--sandbox",
                "workspace-write",
                "--skip-git-repo-check",
                "--cd",
                WORKDIR,
                "--",
                "Say OK",
            ],
        )
        self.assertIsInstance(argv, list)
        self.assertTrue(all(isinstance(part, str) for part in argv))

    def test_initial_argv_never_emits_last(self) -> None:
        argv = self.executor.build_initial_argv("Say OK", workdir=WORKDIR)
        self.assertNotIn("--last", argv)

    def test_prompt_stays_one_token_and_is_never_shell_joined(self) -> None:
        prompt = "Fix `x` && echo pwned; rm -rf /tmp/nothing # not a comment"
        argv = self.executor.build_initial_argv(prompt, workdir=WORKDIR)
        self.assertEqual(argv[-1], prompt)
        self.assertEqual(argv.count(prompt), 1)
        joined = "".join(argv[:-1])
        self.assertNotIn(prompt, joined)

    def test_prompt_must_be_a_non_empty_string(self) -> None:
        for prompt in ("", "   ", None, ["echo hi"], b"echo hi"):
            with self.subTest(prompt=prompt):
                with self.assertRaises(ExecutorConfigError):
                    self.executor.build_initial_argv(prompt, workdir=WORKDIR)

    def test_workdir_is_required_and_must_be_absolute(self) -> None:
        for workdir in (None, "", "relative/dir", 7):
            with self.subTest(workdir=workdir):
                with self.assertRaises(ExecutorConfigError):
                    self.executor.build_initial_argv("Say OK", workdir=workdir)

    def test_resume_argv_places_parent_flags_before_resume(self) -> None:
        argv = self.executor.build_resume_argv(
            THREAD, "And now say DONE", workdir=WORKDIR
        )
        self.assertEqual(
            argv,
            [
                "codex",
                "exec",
                "--json",
                "--sandbox",
                "workspace-write",
                "--skip-git-repo-check",
                "--cd",
                WORKDIR,
                "resume",
                THREAD,
                "And now say DONE",
            ],
        )
        self.assertLess(argv.index("--cd"), argv.index("resume"))
        self.assertEqual(argv[argv.index("resume") + 1], THREAD)
        self.assertNotIn("--last", argv)

    def test_resume_rejects_an_option_like_follow_up(self) -> None:
        with self.assertRaises(ExecutorConfigError):
            self.executor.build_resume_argv(THREAD, "--last", workdir=WORKDIR)

    def test_resume_requires_an_exact_thread_id(self) -> None:
        for thread_id in (None, "", "--last", "two words", "has\nnewline"):
            with self.subTest(thread_id=thread_id):
                with self.assertRaises(ExecutorConfigError):
                    self.executor.build_resume_argv(thread_id, "next", workdir=WORKDIR)

    def test_queue_argv_and_queued_status_for_active_follow_up(self) -> None:
        argv = self.executor.build_queue_argv(THREAD, "And now say DONE")
        self.assertEqual(
            argv,
            ["codex", "queue", "--thread", THREAD, "--message", "And now say DONE"],
        )
        self.assertNotIn("--last", argv)
        result = self.executor.queued_follow_up(THREAD, "And now say DONE")
        self.assertEqual(result.status, "queued")
        self.assertEqual(result.session_id, THREAD)
        self.assertTrue(result.ok is False)
        self.assertIn("queued", result.summary)

    def test_no_codex_argv_ever_contains_last(self) -> None:
        builders = [
            self.executor.build_initial_argv("Say OK", workdir=WORKDIR),
            self.executor.build_resume_argv(THREAD, "next", workdir=WORKDIR),
            self.executor.build_queue_argv(THREAD, "next"),
        ]
        for argv in builders:
            with self.subTest(argv=argv):
                self.assertFalse(any(part.startswith("--last") for part in argv))


class CodexParseTests(unittest.TestCase):
    def setUp(self) -> None:
        self.executor = CodexExecutor()

    def parse(self, name: str, **kwargs) -> ExecutorResult:
        return self.executor.parse_stream(read_lines(name), **kwargs)

    def test_thread_started_captures_the_thread_id_once(self) -> None:
        result = self.parse("codex_success.jsonl")
        self.assertEqual(result.session_id, THREAD)
        self.assertEqual(result.status, "completed")
        self.assertEqual(result.final_message, "OK")
        self.assertEqual(result.summary, "OK")
        self.assertIsNone(result.error)

    def test_event_count_covers_every_valid_event(self) -> None:
        result = self.parse("codex_success.jsonl")
        self.assertEqual(result.event_count, 5)

    def test_resumed_run_with_expected_thread_completes(self) -> None:
        result = self.parse("codex_success.jsonl", expected_session_id=THREAD)
        self.assertEqual(result.status, "completed")
        self.assertEqual(result.session_id, THREAD)

    def test_duplicate_matching_thread_id_is_idempotent(self) -> None:
        lines = read_lines("codex_success.jsonl")
        lines.insert(1, '{"type":"thread.started","thread_id":"%s"}' % THREAD)
        result = self.executor.parse_stream(lines)
        self.assertEqual(result.status, "completed")
        self.assertEqual(result.session_id, THREAD)

    def test_duplicate_different_thread_id_is_an_error(self) -> None:
        result = self.parse("codex_thread_conflict.jsonl")
        self.assertEqual(result.status, "failed")
        self.assertEqual(result.error_code, "thread_conflict")
        self.assertIn("thread.started", result.error)
        self.assertEqual(result.session_id, THREAD)

    def test_resume_mismatch_is_rejected(self) -> None:
        result = self.parse(
            "codex_success.jsonl", expected_session_id="01a1b80e-ffff-78c1-87bf-000000000000"
        )
        self.assertEqual(result.status, "failed")
        self.assertEqual(result.error_code, "session_mismatch")
        self.assertIn("expected", result.error)

    def test_turn_completed_without_thread_id_is_missing_session(self) -> None:
        result = self.parse("codex_missing_thread.jsonl")
        self.assertEqual(result.status, "failed")
        self.assertEqual(result.error_code, "missing_session_id")
        self.assertIsNone(result.session_id)

    def test_silent_stream_with_exit_zero_is_not_completed(self) -> None:
        result = self.parse("codex_stream_without_terminal.jsonl", exit_code=0)
        self.assertEqual(result.status, "failed")
        self.assertEqual(result.error_code, "incomplete_stream")
        self.assertEqual(result.session_id, THREAD)
        self.assertEqual(result.final_message, "Python 3.14.0 is installed.")

    def test_nonzero_exit_without_terminal_event_is_failure(self) -> None:
        result = self.parse("codex_stream_without_terminal.jsonl", exit_code=2)
        self.assertEqual(result.status, "failed")
        self.assertEqual(result.error_code, "exit_nonzero")
        self.assertIn("2", result.error)

    def test_real_turn_failed_stream_is_an_honest_failure(self) -> None:
        result = self.parse("codex_turn_failed.jsonl", exit_code=1)
        self.assertEqual(result.status, "failed")
        self.assertEqual(result.error_code, "turn_failed")
        self.assertIn("not supported", result.error)
        self.assertEqual(result.session_id, THREAD)
        self.assertIsNone(result.final_message)
        self.assertEqual(result.event_count, 5)

    def test_turn_failed_without_exit_code_still_fails(self) -> None:
        result = self.parse("codex_turn_failed.jsonl", exit_code=0)
        self.assertEqual(result.status, "failed")
        self.assertEqual(result.error_code, "turn_failed")

    def test_malformed_line_is_recorded_and_fails_honestly(self) -> None:
        result = self.parse("codex_malformed.jsonl")
        self.assertEqual(result.status, "failed")
        self.assertEqual(result.error_code, "malformed_stream")
        self.assertIn("line 3", result.error)
        self.assertEqual(result.session_id, THREAD)

    def test_empty_stream_is_not_completion(self) -> None:
        result = self.executor.parse_stream([], exit_code=0)
        self.assertEqual(result.status, "failed")
        self.assertEqual(result.error_code, "incomplete_stream")
        self.assertIsNone(result.session_id)

    def test_blank_lines_are_ignored(self) -> None:
        lines = read_lines("codex_success.jsonl")
        lines.insert(0, "")
        lines.append("   ")
        result = self.executor.parse_stream(lines)
        self.assertEqual(result.status, "completed")
        self.assertEqual(result.event_count, 5)

    def test_stream_must_be_an_iterable_of_lines_not_a_blob(self) -> None:
        with self.assertRaises(ExecutorConfigError):
            self.executor.parse_stream('{"type":"turn.completed"}')

    def test_expected_thread_id_is_validated_before_use(self) -> None:
        for expected in ("--last", "two words", ""):
            with self.subTest(expected=expected):
                with self.assertRaises(ExecutorConfigError):
                    self.executor.parse_stream([], expected_session_id=expected)

    def test_unknown_event_types_are_counted_but_ignored(self) -> None:
        lines = read_lines("codex_success.jsonl")
        lines.insert(1, '{"type":"future.event","payload":{"x":1}}')
        result = self.executor.parse_stream(lines)
        self.assertEqual(result.status, "completed")
        self.assertEqual(result.event_count, 6)


if __name__ == "__main__":
    unittest.main()
