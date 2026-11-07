"""Task 11 — Hermes worker adapter: detached worker argv, prompt file, stream."""

from __future__ import annotations

import os
import stat
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from executors.base import (  # noqa: E402
    STATUS_COMPLETED,
    STATUS_FAILED,
    ExecutorConfigError,
)
from executors.hermes_worker import (  # noqa: E402
    NestedWorkerRefusedError,
    HermesWorkerExecutor,
    validate_worker_environment,
    worker_environment,
)

FIXTURES = REPO_ROOT / "fixtures" / "executors"
PROMPT_RESOURCE = REPO_ROOT / "resources" / "worker-prompt.md"
SKILL = REPO_ROOT / "skills" / "neo-worker" / "SKILL.md"

SESSION = "local-d495be544248caa76186dbbf8fc4a04938ad5ad4e2949c52f637d080e95fe402"
OTHER_SESSION = "local-cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc"
WORKDIR = "/Users/user/code/work/demo"
TOKEN = "tok-8f14e45fceea167a5a36dedd4bea2543"
TASK_ID = "task-11f0c0de"


def read_lines(name: str) -> list:
    return (FIXTURES / name).read_text(encoding="utf-8").splitlines()


def lines(*events: str) -> list:
    return list(events)


class HermesWorkerArgvTests(unittest.TestCase):
    def setUp(self) -> None:
        self.executor = HermesWorkerExecutor(
            profile="neo",
            model="deepseek-v4-flash",
            provider="deepseek",
        )

    def test_initial_argv_is_an_exact_pinned_array(self) -> None:
        argv = self.executor.build_initial_argv("/tmp/prompt.md", workdir=WORKDIR)
        self.assertEqual(
            argv,
            [
                "hermes",
                "-p",
                "neo",
                "chat",
                "--oneshot",
                "-Q",
                "--format",
                "stream-json",
                "--query-file",
                "/tmp/prompt.md",
                "--in",
                WORKDIR,
                "--source",
                "tool",
                "-m",
                "deepseek-v4-flash",
                "--provider",
                "deepseek",
                "--skills",
                "neo-worker",
                "--max-turns",
                "12",
            ],
        )
        self.assertIsInstance(argv, list)

    def test_initial_argv_never_uses_inline_prompt_or_shell(self) -> None:
        prompt = "summarise the build failure; $(rm -rf /) `id`"
        with tempfile.TemporaryDirectory() as tmp:
            path = self.executor.write_prompt_file(
                os.path.join(tmp, "prompt.md"),
                task_id=TASK_ID,
                intent=prompt,
                workdir=WORKDIR,
            )
            argv = self.executor.build_initial_argv(path, workdir=WORKDIR)
            self.assertIn("--query-file", argv)
            self.assertEqual(argv[argv.index("--query-file") + 1], path)
            self.assertNotIn("-q", argv)
            self.assertNotIn("--query", argv)
            for token in argv:
                self.assertNotIn(prompt, token)
                self.assertNotIn("rm -rf", token)
            self.assertIsInstance(argv, list)

    def test_query_file_must_be_a_real_absolute_path(self) -> None:
        for bad in ("-", "prompt.md", "./prompt.md", "", " -"):
            with self.subTest(bad=bad):
                with self.assertRaises(ExecutorConfigError):
                    self.executor.build_initial_argv(bad, workdir=WORKDIR)
        with self.assertRaises(ExecutorConfigError):
            self.executor.build_initial_argv(None, workdir=WORKDIR)

    def test_workdir_must_be_absolute(self) -> None:
        for bad in ("demo", "", None, "-"):
            with self.subTest(bad=bad):
                with self.assertRaises(ExecutorConfigError):
                    self.executor.build_initial_argv("/tmp/prompt.md", workdir=bad)

    def test_max_turns_is_always_present_and_validated(self) -> None:
        argv = self.executor.build_initial_argv("/tmp/prompt.md", workdir=WORKDIR)
        self.assertEqual(argv[argv.index("--max-turns") + 1], "12")
        custom = HermesWorkerExecutor(profile="neo", max_turns=3)
        self.assertEqual(
            custom.build_initial_argv("/tmp/prompt.md", workdir=WORKDIR)[
                custom.build_initial_argv("/tmp/prompt.md", workdir=WORKDIR).index(
                    "--max-turns"
                )
                + 1
            ],
            "3",
        )
        for bad in (0, -1, True, "12", 1.5):
            with self.subTest(bad=bad):
                with self.assertRaises(ExecutorConfigError):
                    HermesWorkerExecutor(profile="neo", max_turns=bad)

    def test_run_budget_is_optional(self) -> None:
        default_argv = self.executor.build_initial_argv("/tmp/prompt.md", workdir=WORKDIR)
        self.assertNotIn("--run-budget", default_argv)

        budgeted = HermesWorkerExecutor(profile="neo", run_budget=90)
        argv = budgeted.build_initial_argv("/tmp/prompt.md", workdir=WORKDIR)
        self.assertEqual(argv[-2:], ["--run-budget", "90"])
        self.assertNotIn("--run-budget", self.executor.build_resume_argv(
            SESSION, "/tmp/prompt.md", workdir=WORKDIR
        ))

        for bad in (0, -5, True, "90", 1.5):
            with self.subTest(bad=bad):
                with self.assertRaises(ExecutorConfigError):
                    HermesWorkerExecutor(profile="neo", run_budget=bad)

    def test_profile_model_provider_and_skills_are_configurable(self) -> None:
        executor = HermesWorkerExecutor(
            profile="neo-voice",
            model="deepseek-v4-flash",
            provider="deepseek",
            skills=("neo-worker", "neo-preview"),
            binary="/Users/user/.local/bin/hermes",
        )
        argv = executor.build_initial_argv("/tmp/prompt.md", workdir=WORKDIR)
        self.assertEqual(argv[0], "/Users/user/.local/bin/hermes")
        self.assertEqual(argv[argv.index("-p") + 1], "neo-voice")
        self.assertEqual(argv[argv.index("-m") + 1], "deepseek-v4-flash")
        self.assertEqual(argv[argv.index("--provider") + 1], "deepseek")
        self.assertEqual(argv[argv.index("--skills") + 1], "neo-worker,neo-preview")

    def test_absent_profile_and_model_are_omitted_not_invented(self) -> None:
        argv = HermesWorkerExecutor(skills=("neo-worker",)).build_initial_argv(
            "/tmp/prompt.md", workdir=WORKDIR
        )
        self.assertNotIn("-p", argv)
        self.assertNotIn("-m", argv)
        self.assertNotIn("--provider", argv)
        self.assertIn("--skills", argv)

    def test_namespaced_model_id_is_accepted(self) -> None:
        executor = HermesWorkerExecutor(
            profile="neo", model="anthropic/claude-sonnet-4", provider="deepseek"
        )
        argv = executor.build_initial_argv("/tmp/prompt.md", workdir=WORKDIR)
        self.assertEqual(argv[argv.index("-m") + 1], "anthropic/claude-sonnet-4")

    def test_flag_injection_in_config_values_is_refused(self) -> None:
        for kwargs in (
            {"profile": "--yolo"},
            {"model": "-m"},
            {"provider": "--api-key"},
            {"skills": ("neo-worker --yolo",)},
            {"source": "-Q"},
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ExecutorConfigError):
                    HermesWorkerExecutor(**kwargs)

    def test_no_secret_or_api_key_flag_is_ever_emitted(self) -> None:
        executor = HermesWorkerExecutor(profile="neo", provider="deepseek")
        for argv in (
            executor.build_initial_argv("/tmp/prompt.md", workdir=WORKDIR),
            executor.build_resume_argv(SESSION, "/tmp/prompt.md", workdir=WORKDIR),
        ):
            self.assertNotIn("--api-key", argv)
            self.assertNotIn("--base-url", argv)
            self.assertNotIn("--yolo", argv)
            self.assertNotIn("--continue", argv)
            self.assertNotIn("--worktree", argv)
            for token in argv:
                self.assertNotIn(TOKEN, token)


class HermesWorkerResumeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.executor = HermesWorkerExecutor(
            profile="neo", model="deepseek-v4-flash", provider="deepseek"
        )

    def test_resume_argv_pins_the_exact_session(self) -> None:
        argv = self.executor.build_resume_argv(SESSION, "/tmp/prompt.md", workdir=WORKDIR)
        self.assertEqual(
            argv,
            [
                "hermes",
                "-p",
                "neo",
                "chat",
                "--oneshot",
                "-Q",
                "--resume",
                SESSION,
                "--format",
                "stream-json",
                "--query-file",
                "/tmp/prompt.md",
                "--in",
                WORKDIR,
                "--source",
                "tool",
                "-m",
                "deepseek-v4-flash",
                "--provider",
                "deepseek",
                "--skills",
                "neo-worker",
                "--max-turns",
                "12",
            ],
        )
        self.assertEqual(argv[argv.index("--resume") + 1], SESSION)

    def test_resume_preserves_the_same_model_provider_profile_and_workdir(self) -> None:
        initial = self.executor.build_initial_argv("/tmp/prompt.md", workdir=WORKDIR)
        resumed = self.executor.build_resume_argv(
            SESSION, "/tmp/prompt.md", workdir=WORKDIR
        )
        for flag in (
            "-p",
            "-m",
            "--provider",
            "--in",
            "--source",
            "--skills",
            "--max-turns",
        ):
            with self.subTest(flag=flag):
                self.assertEqual(
                    initial[initial.index(flag) + 1], resumed[resumed.index(flag) + 1]
                )

    def test_latest_and_continue_are_refused(self) -> None:
        for bad in ("latest", "LATEST", "", " ", "--last", "-continue", "$(id)"):
            with self.subTest(bad=bad):
                with self.assertRaises(ExecutorConfigError):
                    self.executor.build_resume_argv(bad, "/tmp/prompt.md", workdir=WORKDIR)
        argv = self.executor.build_resume_argv(SESSION, "/tmp/prompt.md", workdir=WORKDIR)
        self.assertNotIn("--continue", argv)
        self.assertNotIn("-c", argv)
        self.assertNotIn("latest", argv)

    def test_resume_workdir_is_required(self) -> None:
        with self.assertRaises(ExecutorConfigError):
            self.executor.build_resume_argv(SESSION, "/tmp/prompt.md", workdir="demo")


class HermesWorkerEnvironmentTests(unittest.TestCase):
    def test_worker_environment_carries_the_three_contract_keys(self) -> None:
        environment = worker_environment(TASK_ID, TOKEN)
        self.assertEqual(environment["NEO_WORKER"], "1")
        self.assertEqual(environment["NEO_TASK_ID"], TASK_ID)
        self.assertEqual(environment["NEO_TASK_TOKEN"], TOKEN)

    def test_worker_environment_preserves_the_supervisor_base(self) -> None:
        environment = worker_environment(TASK_ID, TOKEN, base={"PATH": "/usr/bin"})
        self.assertEqual(environment["PATH"], "/usr/bin")
        self.assertEqual(environment["NEO_WORKER"], "1")
        self.assertNotIn("PATH", worker_environment(TASK_ID, TOKEN))

    def test_a_worker_cannot_spawn_another_worker(self) -> None:
        with self.assertRaises(NestedWorkerRefusedError):
            worker_environment(TASK_ID, TOKEN, base={"NEO_WORKER": "1"})
        self.assertTrue(issubclass(NestedWorkerRefusedError, ExecutorConfigError))

    def test_environment_validation_names_missing_keys(self) -> None:
        with self.assertRaises(ExecutorConfigError) as caught:
            validate_worker_environment({"NEO_WORKER": "1"})
        message = str(caught.exception)
        self.assertIn("NEO_TASK_ID", message)
        self.assertIn("NEO_TASK_TOKEN", message)

        with self.assertRaises(ExecutorConfigError):
            validate_worker_environment(
                {"NEO_WORKER": "1", "NEO_TASK_ID": TASK_ID, "NEO_TASK_TOKEN": ""}
            )
        self.assertEqual(
            validate_worker_environment(
                {"NEO_WORKER": "1", "NEO_TASK_ID": TASK_ID, "NEO_TASK_TOKEN": TOKEN}
            ),
            {"NEO_WORKER": "1", "NEO_TASK_ID": TASK_ID, "NEO_TASK_TOKEN": TOKEN},
        )

    def test_token_never_reaches_argv_or_the_prompt_file(self) -> None:
        executor = HermesWorkerExecutor(
            profile="neo", model="deepseek-v4-flash", provider="deepseek"
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = executor.write_prompt_file(
                os.path.join(tmp, "prompt.md"),
                task_id=TASK_ID,
                intent="add a preview screenshot",
                workdir=WORKDIR,
                forbidden_values=(TOKEN,),
            )
            text = Path(path).read_text(encoding="utf-8")
            self.assertNotIn(TOKEN, text)
            for argv in (
                executor.build_initial_argv(path, workdir=WORKDIR),
                executor.build_resume_argv(SESSION, path, workdir=WORKDIR),
            ):
                for token in argv:
                    self.assertNotIn(TOKEN, token)


class WorkerPromptFileTests(unittest.TestCase):
    def setUp(self) -> None:
        self.executor = HermesWorkerExecutor(profile="neo")
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        self.path = os.path.join(self.tempdir.name, "prompt.md")

    def write(self, *, intent="render two stills", **kwargs) -> str:
        return self.executor.write_prompt_file(
            self.path, task_id=TASK_ID, intent=intent, workdir=WORKDIR, **kwargs
        )

    def test_prompt_contains_task_intent_and_boundaries(self) -> None:
        path = self.write(intent="render two stills of the office")
        text = Path(path).read_text(encoding="utf-8")
        self.assertIn("render two stills of the office", text)
        self.assertIn(TASK_ID, text)
        self.assertIn(WORKDIR, text)
        for required in (
            "neo_task_update",
            "neo_task_complete",
            "approval",
            "nested",
            "process exit",
        ):
            with self.subTest(required=required):
                self.assertIn(required, text)

    def test_prompt_file_is_restrictive_mode(self) -> None:
        path = self.write()
        mode = stat.S_IMODE(os.stat(path).st_mode)
        self.assertEqual(mode, 0o600)

    def test_write_is_atomic_and_leaves_no_temporary_files(self) -> None:
        Path(self.path).write_text("stale content that must vanish", encoding="utf-8")
        os.chmod(self.path, 0o644)
        path = self.write(intent="fresh intent")
        text = Path(path).read_text(encoding="utf-8")
        self.assertNotIn("stale content", text)
        self.assertEqual(stat.S_IMODE(os.stat(path).st_mode), 0o600)
        leftovers = [name for name in os.listdir(self.tempdir.name) if name != "prompt.md"]
        self.assertEqual(leftovers, [])

    def test_secret_values_are_refused_before_writing(self) -> None:
        with self.assertRaises(ExecutorConfigError):
            self.write(intent="token is %s" % TOKEN, forbidden_values=(TOKEN,))
        self.assertFalse(os.path.exists(self.path))

    def test_environment_assignments_and_key_blocks_are_refused(self) -> None:
        for intent in (
            "export NEO_TASK_TOKEN=abc123",
            "use OPENAI_API_KEY=sk-live-1234",
            "-----BEGIN OPENSSH PRIVATE KEY-----",
        ):
            with self.subTest(intent=intent):
                with self.assertRaises(ExecutorConfigError):
                    self.write(intent=intent)
                self.assertFalse(os.path.exists(self.path))

    def test_bad_paths_and_intents_are_refused(self) -> None:
        for bad_path in (
            "prompt.md",
            "-",
            "",
            os.path.join(self.tempdir.name, "missing", "p.md"),
        ):
            with self.subTest(bad_path=bad_path):
                with self.assertRaises(ExecutorConfigError):
                    self.executor.write_prompt_file(
                        bad_path, task_id=TASK_ID, intent="x", workdir=WORKDIR
                    )
        for bad_intent in ("", "   "):
            with self.subTest(bad_intent=bad_intent):
                with self.assertRaises(ExecutorConfigError):
                    self.write(intent=bad_intent)
        with self.assertRaises(ExecutorConfigError):
            self.write(intent=None)  # type: ignore[arg-type]
        for bad_task in ("", "../escape", "task id with spaces", 12):
            with self.subTest(bad_task=bad_task):
                with self.assertRaises(ExecutorConfigError):
                    self.executor.write_prompt_file(
                        self.path, task_id=bad_task, intent="x", workdir=WORKDIR
                    )

    def test_multiline_intent_is_preserved_verbatim(self) -> None:
        intent = "line one\nline two $(date)\nline three `id`"
        path = self.write(intent=intent)
        text = Path(path).read_text(encoding="utf-8")
        self.assertIn("line one\nline two $(date)\nline three `id`", text)

    def test_shipped_template_exists_and_is_loadable(self) -> None:
        self.assertTrue(PROMPT_RESOURCE.is_file())
        template = PROMPT_RESOURCE.read_text(encoding="utf-8")
        for placeholder in ("{{TASK_ID}}", "{{WORKDIR}}", "{{INTENT}}"):
            with self.subTest(placeholder=placeholder):
                self.assertIn(placeholder, template)


class HermesWorkerStreamTests(unittest.TestCase):
    def setUp(self) -> None:
        self.executor = HermesWorkerExecutor(profile="neo")

    def test_success_fixture_completes_with_the_reported_session(self) -> None:
        result = self.executor.parse_stream(read_lines("hermes_success.jsonl"))
        self.assertEqual(result.status, STATUS_COMPLETED)
        self.assertEqual(result.session_id, SESSION)
        self.assertEqual(result.final_message, "NEO-CANARY-OK")
        self.assertEqual(result.event_count, 9)
        self.assertIsNone(result.error)

    def test_text_deltas_concatenate_when_the_result_has_no_text(self) -> None:
        stream = lines(
            '{"type": "system", "subtype": "init", "session_id": ""}',
            '{"type": "text", "text": "Hel"}',
            '{"type": "text", "text": "lo "}',
            '{"type": "text", "text": "world"}',
            '{"type": "result", "session_id": "%s", "exit_code": 0, "text": ""}' % SESSION,
        )
        result = self.executor.parse_stream(stream)
        self.assertEqual(result.status, STATUS_COMPLETED)
        self.assertEqual(result.final_message, "Hello world")

    def test_result_text_wins_over_deltas(self) -> None:
        stream = lines(
            '{"type": "system", "subtype": "init", "session_id": ""}',
            '{"type": "text", "text": "draft"}',
            '{"type": "result", "session_id": "%s", "exit_code": 0,'
            ' "text": "final"}' % SESSION,
        )
        result = self.executor.parse_stream(stream)
        self.assertEqual(result.final_message, "final")

    def test_empty_init_session_is_not_an_error(self) -> None:
        result = self.executor.parse_stream(read_lines("hermes_success.jsonl"))
        self.assertEqual(result.status, STATUS_COMPLETED)
        self.assertNotEqual(result.error_code, "malformed_stream")

    def test_resume_fixture_matches_the_expected_session(self) -> None:
        result = self.executor.parse_stream(
            read_lines("hermes_resume.jsonl"), expected_session_id=SESSION
        )
        self.assertEqual(result.status, STATUS_COMPLETED)
        self.assertEqual(result.session_id, SESSION)
        self.assertEqual(result.final_message, "RESUMED-OK")

    def test_returned_session_mismatch_is_an_honest_failure(self) -> None:
        result = self.executor.parse_stream(
            read_lines("hermes_success.jsonl"), expected_session_id=OTHER_SESSION
        )
        self.assertEqual(result.status, STATUS_FAILED)
        self.assertEqual(result.error_code, "session_mismatch")
        self.assertIn(SESSION, result.error)
        self.assertIn(OTHER_SESSION, result.error)

    def test_missing_result_session_is_a_failure(self) -> None:
        result = self.executor.parse_stream(read_lines("hermes_missing_session.jsonl"))
        self.assertEqual(result.status, STATUS_FAILED)
        self.assertEqual(result.error_code, "missing_session_id")
        self.assertIsNone(result.session_id)

    def test_malformed_stream_is_a_failure(self) -> None:
        result = self.executor.parse_stream(read_lines("hermes_malformed.jsonl"))
        self.assertEqual(result.status, STATUS_FAILED)
        self.assertEqual(result.error_code, "malformed_stream")
        self.assertIn("line 3", result.error)

    def test_stream_without_a_result_is_a_failure(self) -> None:
        result = self.executor.parse_stream(read_lines("hermes_no_result.jsonl"))
        self.assertEqual(result.status, STATUS_FAILED)
        self.assertEqual(result.error_code, "incomplete_stream")
        self.assertIn("result", result.error)

    def test_result_level_nonzero_exit_is_a_failure(self) -> None:
        result = self.executor.parse_stream(read_lines("hermes_nonzero_exit.jsonl"))
        self.assertEqual(result.status, STATUS_FAILED)
        self.assertEqual(result.error_code, "exit_nonzero")
        self.assertEqual(result.final_message, "half")

    def test_rejected_gateway_option_fixture_reports_exit_nonzero(self) -> None:
        result = self.executor.parse_stream(read_lines("hermes_result_error.jsonl"))
        self.assertEqual(result.status, STATUS_FAILED)
        self.assertEqual(result.error_code, "exit_nonzero")
        self.assertIn("--run-budget", result.error)
        self.assertIsNone(result.session_id)

    def test_process_level_nonzero_exit_is_a_failure(self) -> None:
        result = self.executor.parse_stream(
            read_lines("hermes_success.jsonl"), exit_code=1
        )
        self.assertEqual(result.status, STATUS_FAILED)
        self.assertEqual(result.error_code, "exit_nonzero")

    def test_timeout_is_an_explicit_error_code(self) -> None:
        result = self.executor.parse_stream(
            read_lines("hermes_no_result.jsonl"), timed_out=True
        )
        self.assertEqual(result.status, STATUS_FAILED)
        self.assertEqual(result.error_code, "timeout")

    def test_result_error_with_zero_exit_is_a_failure(self) -> None:
        stream = lines(
            '{"type": "system", "subtype": "init", "session_id": ""}',
            '{"type": "result", "session_id": "%s", "exit_code": 0, "text": "",'
            ' "error": "provider refused the request"}' % SESSION,
        )
        result = self.executor.parse_stream(stream)
        self.assertEqual(result.status, STATUS_FAILED)
        self.assertEqual(result.error_code, "result_error")
        self.assertIn("provider refused", result.error)

    def test_conflicting_session_ids_across_a_turn_are_a_failure(self) -> None:
        stream = lines(
            '{"type": "system", "subtype": "init", "session_id": "%s"}' % OTHER_SESSION,
            '{"type": "text", "text": "hi"}',
            '{"type": "result", "session_id": "%s", "exit_code": 0,'
            ' "text": "hi"}' % SESSION,
        )
        result = self.executor.parse_stream(stream)
        self.assertEqual(result.status, STATUS_FAILED)
        self.assertEqual(result.error_code, "session_conflict")

    def test_an_error_event_alone_does_not_fake_completion(self) -> None:
        stream = lines(
            '{"type": "system", "subtype": "init", "session_id": ""}',
            '{"type": "error", "message": "tool layer unavailable"}',
        )
        result = self.executor.parse_stream(stream)
        self.assertEqual(result.status, STATUS_FAILED)
        self.assertEqual(result.error_code, "incomplete_stream")
        self.assertIn("tool layer unavailable", result.error)

    def test_single_string_stream_is_refused(self) -> None:
        with self.assertRaises(ExecutorConfigError):
            self.executor.parse_stream('{"type": "system"}\n')

    def test_event_without_type_is_malformed(self) -> None:
        result = self.executor.parse_stream(
            lines(
                '{"type": "system", "subtype": "init", "session_id": ""}',
                '{"session_id": "x"}',
            )
        )
        self.assertEqual(result.status, STATUS_FAILED)
        self.assertEqual(result.error_code, "malformed_stream")


class WorkerDocumentationTests(unittest.TestCase):
    def test_prompt_resource_states_the_worker_boundaries(self) -> None:
        text = PROMPT_RESOURCE.read_text(encoding="utf-8")
        for required in (
            "bridge",
            "neo_task_complete",
            "approval",
            "nested",
            "exit",
        ):
            with self.subTest(required=required):
                self.assertIn(required, text)

    def test_skill_names_the_detached_worker_session(self) -> None:
        text = SKILL.read_text(encoding="utf-8")
        for required in (
            "NEO_WORKER",
            "detached",
            "bridge tools",
            "nested NEO",
            "approval",
            "process exit",
        ):
            with self.subTest(required=required):
                self.assertIn(required, text)


if __name__ == "__main__":
    unittest.main()
