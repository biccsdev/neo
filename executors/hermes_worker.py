"""Detached Hermes worker executor: prompt-file runs and exact-session resumes.

A NEO worker is a ``hermes chat`` session started detached from the viewer. It
is launched with an argv array only — never a shell string and never an inline
prompt — because the task intent is arbitrary user text. The intent travels in a
restrictive prompt file (mode ``0600``, atomically replaced) passed with
``--query-file``, so quotes, ``$(...)`` and backticks are preserved verbatim and
interpreted by nobody.

The CLI's live stream shape is captured in ``fixtures/executors/hermes_*.jsonl``:
``system/init`` arrives with an *empty* session id, text deltas follow, and the
real session id appears once, in the terminal ``result`` event, alongside
``exit_code``. A result session id is therefore the only session handle this
adapter will ever report, and a run without one is an honest failure — the
worker can never be resumed exactly, so it cannot be called complete.

Secrets stay out of the process's visible surface. The bridge token reaches the
worker as ``NEO_TASK_TOKEN`` in its environment, supplied by
``workers.ProcessSupervisor``; this adapter never accepts a token, never emits
``--api-key``, and refuses to build a prompt that contains one.

``--run-budget`` is emitted only when a budget is configured: some gateway CLI
deployments reject the flag outright (captured in
``hermes_result_error.jsonl``), so it is opt-in rather than assumed.

``--skills`` is a precondition, not a suggestion: the preloaded skill must be
installed in the target profile. With the bundled skill missing, the live CLI
answers a result event carrying ``error: unknown_skill`` and exit code 1, which
this adapter reports as ``exit_nonzero`` rather than a quiet success.
"""

from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path
from typing import Iterable, List, Mapping, Optional, Sequence, Union

from executors.base import (
    STATUS_COMPLETED,
    STATUS_FAILED,
    ExecutorAdapter,
    ExecutorConfigError,
    ExecutorResult,
    StreamCollector,
    check_line_source,
    clip,
    validate_argv,
    validate_exit_code,
    validate_session_id,
    validate_workdir,
)

DEFAULT_BINARY = "hermes"
DEFAULT_SOURCE = "tool"
DEFAULT_SKILLS = ("neo-worker",)
DEFAULT_MAX_TURNS = 12

WORKER_FLAG = "NEO_WORKER"
WORKER_TASK_ENV = "NEO_TASK_ID"
WORKER_TOKEN_ENV = "NEO_TASK_TOKEN"

PROMPT_TEMPLATE = Path(__file__).resolve().parents[1] / "resources" / "worker-prompt.md"
PROMPT_MODE = 0o600

_SAFE_TASK_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
_SAFE_TOKEN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
# Model ids are often namespaced ("anthropic/claude-sonnet-4"), so the model and
# provider tokens additionally allow "/", "@" and "+". Everything else is still
# refused: no whitespace, no leading dash, so none of it can pose as a flag.
_SAFE_MODEL_TOKEN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/@+\-]{0,127}$")
_PLACEHOLDERS = ("{{TASK_ID}}", "{{WORKDIR}}", "{{INTENT}}")

# Anything that looks like a credential assignment or a key block must never be
# rendered into a prompt file, even if a caller passes it as "intent".
_SECRET_PATTERNS = (
    re.compile(
        r"(?i)\b(?:NEO_TASK_TOKEN|NEO_TASK_ID|OPENAI_API_KEY|ANTHROPIC_API_KEY"
        r"|DEEPSEEK_API_KEY|HERMES_API_KEY|AWS_SECRET_ACCESS_KEY)\s*="
    ),
    re.compile(r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----"),
    re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._\-]{8,}"),
    re.compile(r"\bsk-[A-Za-z0-9_\-]{8,}"),
)


class NestedWorkerRefusedError(ExecutorConfigError):
    """A process already marked as a NEO worker tried to start another one."""


def worker_environment(
    task_id: str,
    token: str,
    *,
    base: Optional[Mapping[str, str]] = None,
) -> dict:
    """Return the child environment the supervisor must hand a worker.

    The three keys are the whole contract: the worker flag, its task id, and its
    per-task token. The token belongs here and nowhere else — not in argv, not in
    the prompt file, not in a log line. A base environment that already carries
    ``NEO_WORKER=1`` means a worker is spawning a worker, which is refused at
    this boundary so nothing downstream has to trust the prompt for it.
    """

    if not isinstance(task_id, str) or not _SAFE_TASK_ID.fullmatch(task_id):
        raise ExecutorConfigError("task id is not a safe worker task id: %r" % (task_id,))
    if not isinstance(token, str) or not token.strip():
        raise ExecutorConfigError("worker token must be a non-empty string")
    environment = dict(base or {})
    if environment.get(WORKER_FLAG) == "1":
        raise NestedWorkerRefusedError(
            "a NEO worker cannot spawn another NEO worker; recursion is refused "
            "at the worker's tool layer"
        )
    environment[WORKER_FLAG] = "1"
    environment[WORKER_TASK_ENV] = task_id
    environment[WORKER_TOKEN_ENV] = token
    return environment


def validate_worker_environment(environment: Mapping[str, str]) -> dict:
    """Return the contract keys, refusing an environment that cannot report."""

    if not isinstance(environment, Mapping):
        raise ExecutorConfigError("worker environment must be a mapping")
    missing = [
        key
        for key in (WORKER_FLAG, WORKER_TASK_ENV, WORKER_TOKEN_ENV)
        if not isinstance(environment.get(key), str)
        or not str(environment.get(key)).strip()
    ]
    if missing:
        raise ExecutorConfigError(
            "worker environment is missing required keys: %s" % ", ".join(missing)
        )
    return {
        key: environment[key] for key in (WORKER_FLAG, WORKER_TASK_ENV, WORKER_TOKEN_ENV)
    }


def _safe_flag_value(value, *, label: str, namespaced: bool = False) -> str:
    pattern = _SAFE_MODEL_TOKEN if namespaced else _SAFE_TOKEN
    if not isinstance(value, str) or not pattern.fullmatch(value):
        raise ExecutorConfigError("%s is not a safe argv token: %r" % (label, value))
    return value


def _validate_query_file(path) -> str:
    """A prompt path, never a prompt: absolute, not stdin, not a flag."""

    if not isinstance(path, str) or not path.strip():
        raise ExecutorConfigError("query file path must be a non-empty string")
    if path == "-" or path.startswith("-"):
        raise ExecutorConfigError(
            "query file must be a real file path, not stdin or a flag: %r" % (path,)
        )
    if "\n" in path or "\r" in path or "\x00" in path:
        raise ExecutorConfigError("query file path contains a control character")
    if not os.path.isabs(path):
        raise ExecutorConfigError("query file path must be absolute: %r" % (path,))
    return path


def _positive_int(value, *, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ExecutorConfigError("%s must be a positive integer: %r" % (label, value))
    return value


class HermesWorkerExecutor(ExecutorAdapter):
    """Build and interpret ``hermes chat --oneshot --format stream-json``."""

    name = "hermes_worker"

    def __init__(
        self,
        binary: str = DEFAULT_BINARY,
        *,
        profile: Optional[str] = None,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        skills: Union[str, Sequence[str]] = DEFAULT_SKILLS,
        max_turns: int = DEFAULT_MAX_TURNS,
        run_budget: Optional[int] = None,
        source: str = DEFAULT_SOURCE,
        prompt_template: Optional[Union[str, Path]] = None,
    ) -> None:
        if not isinstance(binary, str) or not binary.strip():
            raise ExecutorConfigError("hermes binary must be a non-empty string")
        if binary.startswith("-"):
            raise ExecutorConfigError("hermes binary must not be a flag: %r" % (binary,))
        self.binary = binary
        self.profile = (
            None if profile is None else _safe_flag_value(profile, label="profile")
        )
        self.model = (
            None
            if model is None
            else _safe_flag_value(model, label="model", namespaced=True)
        )
        self.provider = (
            None
            if provider is None
            else _safe_flag_value(provider, label="provider", namespaced=True)
        )
        self.skills = self._validate_skills(skills)
        self.max_turns = _positive_int(max_turns, label="max_turns")
        self.run_budget = (
            None if run_budget is None else _positive_int(run_budget, label="run_budget")
        )
        self.source = _safe_flag_value(source, label="source")
        self.prompt_template = Path(prompt_template) if prompt_template else PROMPT_TEMPLATE

    @staticmethod
    def _validate_skills(skills) -> tuple:
        if isinstance(skills, str):
            skills = (skills,)
        if isinstance(skills, (bytes, bytearray)) or not isinstance(skills, Sequence):
            raise ExecutorConfigError("skills must be a sequence of skill names")
        values = tuple(_safe_flag_value(item, label="skill") for item in skills)
        if not values:
            raise ExecutorConfigError("at least one skill must be preloaded")
        return values

    # -- argv -----------------------------------------------------------

    def _head(self, *, workdir) -> List[str]:
        argv = [self.binary]
        if self.profile is not None:
            argv += ["-p", self.profile]
        argv += ["chat", "--oneshot", "-Q"]
        return argv

    def _tail(self, *, workdir) -> List[str]:
        argv = ["--source", self.source]
        if self.model is not None:
            argv += ["-m", self.model]
        if self.provider is not None:
            argv += ["--provider", self.provider]
        argv += ["--skills", ",".join(self.skills)]
        argv += ["--max-turns", str(self.max_turns)]
        if self.run_budget is not None:
            argv += ["--run-budget", str(self.run_budget)]
        return argv

    def build_initial_argv(self, prompt, *, workdir: Optional[str] = None) -> List[str]:
        """``prompt`` is the *path* of the prompt file, never prompt text."""

        path = _validate_query_file(prompt)
        cwd = validate_workdir(workdir)
        argv = (
            self._head(workdir=cwd)
            + ["--format", "stream-json", "--query-file", path, "--in", cwd]
            + self._tail(workdir=cwd)
        )
        return validate_argv(argv)

    def build_resume_argv(
        self, session_id, prompt, *, workdir: Optional[str] = None
    ) -> List[str]:
        """Resume the exact session id the CLI reported; never a recency lookup."""

        session = validate_session_id(session_id, label="worker session id")
        if session.lower() == "latest":
            raise ExecutorConfigError(
                "worker resume requires the exact session id, never 'latest'"
            )
        path = _validate_query_file(prompt)
        cwd = validate_workdir(workdir)
        argv = (
            self._head(workdir=cwd)
            + [
                "--resume",
                session,
                "--format",
                "stream-json",
                "--query-file",
                path,
                "--in",
                cwd,
            ]
            + self._tail(workdir=cwd)
        )
        return validate_argv(argv)

    # -- prompt file -----------------------------------------------------

    def write_prompt_file(
        self,
        path,
        *,
        task_id: str,
        intent: str,
        workdir: str,
        forbidden_values: Iterable[str] = (),
    ) -> str:
        """Write the worker prompt atomically, mode ``0600``, and return its path.

        The prompt is built from the task intent plus the fixed approval,
        recursion and completion boundaries. It never receives a token, an API
        key or an environment dump: a caller can additionally pass known secret
        values as ``forbidden_values`` and the write is refused if any appears.
        """

        target = _validate_query_file(path)
        if not isinstance(task_id, str) or not _SAFE_TASK_ID.fullmatch(task_id):
            raise ExecutorConfigError(
                "task id is not a safe worker task id: %r" % (task_id,)
            )
        if not isinstance(intent, str) or not intent.strip():
            raise ExecutorConfigError("task intent must be a non-empty string")
        cwd = validate_workdir(workdir)

        parent = os.path.dirname(target)
        if not os.path.isdir(parent):
            raise ExecutorConfigError(
                "prompt file directory does not exist: %r" % (parent,)
            )

        text = self._render_prompt(task_id=task_id, intent=intent, workdir=cwd)
        self._screen_prompt(text, forbidden_values)

        handle_path = None
        try:
            descriptor, handle_path = tempfile.mkstemp(
                prefix=".worker-prompt-", suffix=".tmp", dir=parent
            )
            os.fchmod(descriptor, PROMPT_MODE)
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                handle.write(text)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(handle_path, target)
            handle_path = None
        except OSError as error:
            raise ExecutorConfigError(
                "could not write the worker prompt file: %s" % (error,)
            ) from None
        finally:
            if handle_path is not None:
                try:
                    os.unlink(handle_path)
                except OSError:
                    pass
        return target

    def _render_prompt(self, *, task_id: str, intent: str, workdir: str) -> str:
        try:
            template = self.prompt_template.read_text(encoding="utf-8")
        except OSError as error:
            raise ExecutorConfigError(
                "worker prompt template is unreadable: %s" % (error,)
            ) from None
        for placeholder in _PLACEHOLDERS:
            if placeholder not in template:
                raise ExecutorConfigError(
                    "worker prompt template is missing %s" % (placeholder,)
                )
        return (
            template.replace("{{TASK_ID}}", task_id)
            .replace("{{WORKDIR}}", workdir)
            .replace("{{INTENT}}", intent.strip())
        )

    @staticmethod
    def _screen_prompt(text: str, forbidden_values: Iterable[str]) -> None:
        for value in forbidden_values or ():
            if isinstance(value, str) and value.strip() and value in text:
                raise ExecutorConfigError(
                    "refusing to write a worker prompt containing a secret value"
                )
        for pattern in _SECRET_PATTERNS:
            if pattern.search(text):
                raise ExecutorConfigError(
                    "refusing to write a worker prompt containing credential material"
                )

    # -- parsing --------------------------------------------------------

    def parse_stream(
        self,
        lines,
        *,
        expected_session_id: Optional[str] = None,
        exit_code: int = 0,
        timed_out: bool = False,
    ) -> ExecutorResult:
        """Decode one worker turn; completion needs a terminal result event.

        The captured stream puts an empty session id in ``system/init`` and the
        real id in ``result``, so an empty init id is normal and only a non-empty
        result id counts. Malformed lines, conflicting ids, a missing result, a
        nonzero exit, a timeout and a missing session id are each reported with
        their own error code; none of them can be mistaken for success.
        """

        expected = (
            validate_session_id(expected_session_id, label="expected session id")
            if expected_session_id is not None
            else None
        )
        exit_code = validate_exit_code(exit_code)
        if not isinstance(timed_out, bool):
            raise ExecutorConfigError("timed_out must be a boolean")
        collector = StreamCollector()

        session_id: Optional[str] = None
        conflict: Optional[tuple] = None
        result_event: Optional[dict] = None
        deltas: List[str] = []
        notices: List[str] = []

        def note(source: str, raw) -> None:
            nonlocal session_id, conflict
            if raw is None:
                return
            if not isinstance(raw, str):
                collector.malformed.append(
                    "%s event reported a non-text session_id" % (source,)
                )
                return
            if not raw.strip():
                # The real CLI reports an empty id at init; that is not an error
                # and it is not a session handle.
                return
            try:
                value = validate_session_id(raw.strip(), label="reported session id")
            except ExecutorConfigError:
                collector.malformed.append(
                    "%s event reported an unsafe session_id: %r" % (source, raw)
                )
                return
            if session_id is None:
                session_id = value
            elif value != session_id and conflict is None:
                conflict = (session_id, value)

        for event in collector.read(check_line_source(lines)):
            kind = event.get("type")
            if not isinstance(kind, str) or not kind:
                collector.malformed.append("event without a string type")
                continue

            if kind == "system":
                subtype = event.get("subtype")
                if subtype == "init":
                    note("system/init", event.get("session_id"))
            elif kind == "text":
                text = event.get("text")
                if not isinstance(text, str):
                    collector.malformed.append("text event without string text")
                    continue
                if text:
                    deltas.append(text)
            elif kind == "result":
                if result_event is not None:
                    collector.malformed.append("more than one result event")
                    continue
                result_event = event
                note("result", event.get("session_id"))
            elif kind == "error":
                message = event.get("message")
                if isinstance(message, str) and message.strip():
                    notices.append(message.strip())

        delta_text = "".join(deltas).strip() or None

        if collector.malformed:
            return self._result(
                STATUS_FAILED,
                session_id=session_id,
                final_message=delta_text,
                collector=collector,
                error=collector.malformed[0],
                error_code="malformed_stream",
            )

        if conflict is not None:
            return self._result(
                STATUS_FAILED,
                session_id=session_id,
                final_message=delta_text,
                collector=collector,
                error=(
                    "conflicting session ids reported: %s then %s; a replaced "
                    "session cannot be resumed exactly" % conflict
                ),
                error_code="session_conflict",
            )

        if timed_out:
            detail = "hermes worker exceeded its run budget and was stopped"
            if notices:
                detail = "%s (%s)" % (detail, notices[0])
            return self._result(
                STATUS_FAILED,
                session_id=session_id,
                final_message=delta_text,
                collector=collector,
                error=detail,
                error_code="timeout",
            )

        if expected is not None and session_id is not None and session_id != expected:
            return self._result(
                STATUS_FAILED,
                session_id=session_id,
                final_message=delta_text,
                collector=collector,
                error=(
                    "returned session id %s does not equal expected session id %s"
                    % (session_id, expected)
                ),
                error_code="session_mismatch",
            )

        if result_event is None:
            if exit_code != 0:
                detail = (
                    "hermes worker exited with status %d before emitting a result "
                    "event" % (exit_code,)
                )
                if notices:
                    detail = "%s (%s)" % (detail, notices[0])
                return self._result(
                    STATUS_FAILED,
                    session_id=session_id,
                    final_message=delta_text,
                    collector=collector,
                    error=detail,
                    error_code="exit_nonzero",
                )
            detail = (
                "hermes worker stream ended without a result event; completion is "
                "never inferred from process exit or silence"
            )
            if notices:
                detail = "%s (%s)" % (detail, notices[0])
            return self._result(
                STATUS_FAILED,
                session_id=session_id,
                final_message=delta_text,
                collector=collector,
                error=detail,
                error_code="incomplete_stream",
            )

        result_text = result_event.get("text")
        if not isinstance(result_text, str) or not result_text.strip():
            result_text = None
        else:
            result_text = result_text.strip()
        final_message = result_text or delta_text

        result_error = result_event.get("error")
        if isinstance(result_error, dict):
            message = result_error.get("message")
            result_error = message if isinstance(message, str) else None
        if isinstance(result_error, str) and not result_error.strip():
            result_error = None
        if result_error is not None and not isinstance(result_error, str):
            result_error = None

        result_exit = result_event.get("exit_code")
        if isinstance(result_exit, bool) or not isinstance(result_exit, int):
            result_exit = None

        is_error = result_event.get("is_error") is True
        subtype = result_event.get("subtype")

        if result_exit is not None and result_exit != 0:
            detail = (
                clip(result_error)
                or "hermes worker reported exit_code %d in its result" % (result_exit,)
            )
            return self._result(
                STATUS_FAILED,
                session_id=session_id,
                final_message=final_message,
                collector=collector,
                error=detail,
                error_code="exit_nonzero",
            )

        if result_error is not None or is_error or (
            isinstance(subtype, str) and subtype not in {"success", "completed"}
        ):
            detail = (
                clip(result_error)
                or "hermes worker reported a failed result (subtype=%r)" % (subtype,)
            )
            return self._result(
                STATUS_FAILED,
                session_id=session_id,
                final_message=final_message,
                collector=collector,
                error=detail,
                error_code="result_error",
            )

        if exit_code != 0:
            return self._result(
                STATUS_FAILED,
                session_id=session_id,
                final_message=final_message,
                collector=collector,
                error=(
                    "hermes worker exited with status %d after a successful result"
                    % (exit_code,)
                ),
                error_code="exit_nonzero",
            )

        if session_id is None:
            return self._result(
                STATUS_FAILED,
                final_message=final_message,
                collector=collector,
                error=(
                    "result arrived without a session id; no session handle was "
                    "reported, so the worker cannot be resumed exactly"
                ),
                error_code="missing_session_id",
            )

        return self._result(
            STATUS_COMPLETED,
            session_id=session_id,
            final_message=final_message,
            collector=collector,
        )


__all__ = [
    "DEFAULT_MAX_TURNS",
    "DEFAULT_SKILLS",
    "HermesWorkerExecutor",
    "NestedWorkerRefusedError",
    "PROMPT_MODE",
    "PROMPT_TEMPLATE",
    "WORKER_FLAG",
    "WORKER_TASK_ENV",
    "WORKER_TOKEN_ENV",
    "validate_worker_environment",
    "worker_environment",
]
