"""Task 10 — Claude adapter: explicit UUID sessions and stream-json parsing."""

from __future__ import annotations

import re
import sys
import unittest
import uuid
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from executors.base import ExecutorConfigError, ExecutorResult  # noqa: E402
from executors.claude import ClaudeExecutor  # noqa: E402

FIXTURES = REPO_ROOT / "fixtures" / "executors"
SESSION = "8b0c4b1a-1111-4a2b-9c3d-000000000001"
OTHER_SESSION = "8b0c4b1a-1111-4a2b-9c3d-00000000ffff"
# The auth-failure fixture keeps the session id the live CLI actually reported.
AUTH_SESSION = "11111111-2222-4333-8444-555555555555"
UUID_PATTERN = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
)


def read_lines(name: str) -> list:
    return (FIXTURES / name).read_text(encoding="utf-8").splitlines()


class ClaudeArgvTests(unittest.TestCase):
    def test_initial_argv_generates_an_explicit_uuid(self) -> None:
        executor = ClaudeExecutor()
        argv = executor.build_initial_argv("Say OK")
        self.assertEqual(argv[:2], ["claude", "-p"])
        self.assertEqual(argv[2], "--session-id")
        self.assertRegex(argv[3], UUID_PATTERN)
        self.assertEqual(
            argv,
            [
                "claude",
                "-p",
                "--session-id",
                argv[3],
                "--output-format",
                "stream-json",
                "--verbose",
                "--",
                "Say OK",
            ],
        )

    def test_each_initial_run_generates_a_distinct_uuid(self) -> None:
        executor = ClaudeExecutor()
        first = executor.build_initial_argv("Say OK")[3]
        second = executor.build_initial_argv("Say OK")[3]
        self.assertNotEqual(first, second)
        self.assertRegex(first, UUID_PATTERN)
        self.assertRegex(second, UUID_PATTERN)

    def test_explicit_uuid_is_accepted_and_validated(self) -> None:
        fixed = uuid.UUID(SESSION)
        executor = ClaudeExecutor(uuid_factory=lambda: fixed)
        argv = executor.build_initial_argv("Say OK")
        self.assertEqual(argv[3], SESSION)
        with self.assertRaises(ExecutorConfigError):
            executor.build_initial_argv("Say OK", session_id="not-a-uuid")
        with self.assertRaises(ExecutorConfigError):
            ClaudeExecutor(uuid_factory=lambda: "not-a-uuid").build_initial_argv("Say OK")

    def test_resume_argv_is_exact_and_never_forks(self) -> None:
        executor = ClaudeExecutor()
        argv = executor.build_resume_argv(SESSION, "And now say DONE")
        self.assertEqual(
            argv,
            [
                "claude",
                "-p",
                "--resume",
                SESSION,
                "--output-format",
                "stream-json",
                "--verbose",
                "--",
                "And now say DONE",
            ],
        )
        self.assertLess(argv.index("--resume"), argv.index("--output-format"))
        self.assertEqual(argv[argv.index("--resume") + 1], SESSION)

    def test_no_builder_emits_fork_session_or_continue(self) -> None:
        executor = ClaudeExecutor()
        builders = [
            executor.build_initial_argv("Say OK"),
            executor.build_resume_argv(SESSION, "next"),
        ]
        for argv in builders:
            with self.subTest(argv=argv):
                self.assertNotIn("--fork-session", argv)
                self.assertNotIn("--continue", argv)
                self.assertNotIn("-c", argv)

    def test_resume_requires_an_exact_uuid(self) -> None:
        executor = ClaudeExecutor()
        for session_id in (None, "", "--resume", "not-a-uuid", "8b0c4b1a-1111-4a2b-9c3d"):
            with self.subTest(session_id=session_id):
                with self.assertRaises(ExecutorConfigError):
                    executor.build_resume_argv(session_id, "next")

    def test_prompt_stays_one_token(self) -> None:
        executor = ClaudeExecutor()
        prompt = "Fix `x` && echo pwned; rm -rf /tmp/nothing"
        argv = executor.build_initial_argv(prompt)
        self.assertEqual(argv[-1], prompt)
        self.assertEqual(argv.count(prompt), 1)

    def test_active_follow_up_is_refused_rather_than_copied(self) -> None:
        executor = ClaudeExecutor()
        with self.assertRaises(ExecutorConfigError) as raised:
            executor.build_queue_argv(SESSION, "next")
        self.assertIn("exact session", str(raised.exception))


class ClaudeParseTests(unittest.TestCase):
    def setUp(self) -> None:
        self.executor = ClaudeExecutor()

    def parse(self, name: str, **kwargs) -> ExecutorResult:
        return self.executor.parse_stream(read_lines(name), **kwargs)

    def test_result_success_requires_the_stored_uuid(self) -> None:
        result = self.parse("claude_success.jsonl", expected_session_id=SESSION)
        self.assertEqual(result.status, "completed")
        self.assertEqual(result.session_id, SESSION)
        self.assertEqual(result.final_message, "OK")
        self.assertEqual(result.summary, "OK")
        self.assertIsNone(result.error)

    def test_system_init_is_counted_as_an_event(self) -> None:
        result = self.parse("claude_success.jsonl")
        self.assertEqual(result.event_count, 3)

    def test_copied_session_id_is_rejected(self) -> None:
        result = self.parse("claude_copied_session.jsonl", expected_session_id=SESSION)
        self.assertEqual(result.status, "failed")
        self.assertEqual(result.error_code, "session_conflict")
        self.assertIn(OTHER_SESSION, result.error)

    def test_resume_with_a_different_returned_id_fails(self) -> None:
        result = self.parse("claude_success.jsonl", expected_session_id=OTHER_SESSION)
        self.assertEqual(result.status, "failed")
        self.assertEqual(result.error_code, "session_mismatch")

    def test_observed_auth_failure_is_not_success(self) -> None:
        result = self.parse("claude_auth_failed.jsonl", exit_code=1)
        self.assertEqual(result.status, "failed")
        self.assertEqual(result.error_code, "result_error")
        self.assertIn("Not logged in", result.error)
        self.assertEqual(result.session_id, AUTH_SESSION)

    def test_error_subtype_is_a_failure(self) -> None:
        lines = read_lines("claude_success.jsonl")
        lines[-1] = (
            '{"type":"result","subtype":"error_during_execution","is_error":true,'
            '"num_turns":1,"result":null,"session_id":"%s",'
            '"terminal_reason":"error"}' % SESSION
        )
        result = self.executor.parse_stream(lines)
        self.assertEqual(result.status, "failed")
        self.assertEqual(result.error_code, "result_error")

    def test_question_tool_reports_needs_input(self) -> None:
        result = self.parse("claude_needs_input.jsonl")
        self.assertEqual(result.status, "needs_input")
        self.assertEqual(result.session_id, SESSION)
        self.assertIn("question", result.summary.lower())

    def test_permission_denial_reports_needs_approval(self) -> None:
        result = self.parse("claude_needs_approval.jsonl")
        self.assertEqual(result.status, "needs_approval")
        self.assertEqual(result.session_id, SESSION)
        self.assertIn("permission", result.summary.lower())

    def test_silence_never_infers_needs_input_or_approval(self) -> None:
        result = self.parse("claude_silence.jsonl", exit_code=0)
        self.assertEqual(result.status, "failed")
        self.assertEqual(result.error_code, "incomplete_stream")
        self.assertNotIn(result.status, {"needs_input", "needs_approval"})
        self.assertEqual(result.final_message, "Still thinking.")

    def test_nonzero_exit_without_result_is_failure(self) -> None:
        result = self.parse("claude_silence.jsonl", exit_code=1)
        self.assertEqual(result.status, "failed")
        self.assertEqual(result.error_code, "exit_nonzero")

    def test_malformed_stream_is_an_honest_failure(self) -> None:
        result = self.parse("claude_malformed.jsonl")
        self.assertEqual(result.status, "failed")
        self.assertEqual(result.error_code, "malformed_stream")
        self.assertIn("line 2", result.error)

    def test_missing_session_id_is_never_fabricated(self) -> None:
        result = self.parse("claude_missing_session.jsonl")
        self.assertEqual(result.status, "failed")
        self.assertEqual(result.error_code, "missing_session_id")
        self.assertIsNone(result.session_id)

    def test_empty_stream_is_not_completion(self) -> None:
        result = self.executor.parse_stream([], exit_code=0)
        self.assertEqual(result.status, "failed")
        self.assertEqual(result.error_code, "incomplete_stream")

    def test_expected_session_id_is_validated_before_use(self) -> None:
        for expected in ("--fork-session", "not-a-uuid", ""):
            with self.subTest(expected=expected):
                with self.assertRaises(ExecutorConfigError):
                    self.executor.parse_stream([], expected_session_id=expected)

    def test_stream_must_be_an_iterable_of_lines(self) -> None:
        with self.assertRaises(ExecutorConfigError):
            self.executor.parse_stream('{"type":"result","subtype":"success"}')


if __name__ == "__main__":
    unittest.main()
