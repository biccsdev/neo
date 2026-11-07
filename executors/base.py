"""Shared contract for NEO's native coding-CLI executors.

An executor is deliberately pure. It builds argv arrays and parses JSONL
streams; it never spawns subprocesses, never builds a shell string, and never
claims a turn completed because a process exited zero or because a stream went
quiet. Completion requires terminal evidence from the CLI itself, and a session
handle is only ever the identifier the CLI actually reported.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Iterable, Iterator, List, Optional

STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"
STATUS_NEEDS_INPUT = "needs_input"
STATUS_NEEDS_APPROVAL = "needs_approval"
STATUS_QUEUED = "queued"

STATUSES = (
    STATUS_COMPLETED,
    STATUS_FAILED,
    STATUS_NEEDS_INPUT,
    STATUS_NEEDS_APPROVAL,
    STATUS_QUEUED,
)

# A session handle is passed as its own argv token, so it must never be able to
# masquerade as a flag or smuggle separators.
_SAFE_SESSION_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_SUMMARY_LIMIT = 240


class ExecutorConfigError(ValueError):
    """The adapter was asked for something it must refuse."""


class FollowUpNotSupportedError(ExecutorConfigError):
    """This CLI cannot take a follow-up into the exact session right now."""


@dataclass(frozen=True)
class ExecutorResult:
    """The honest outcome of one executor turn.

    ``status`` is one of :data:`STATUSES`. A completed result must carry the
    session id that proves continuity, and a failed result must carry error
    evidence. Both rules are enforced at construction so no code path can
    report an unexplained success or a silent failure.
    """

    status: str
    session_id: Optional[str] = None
    final_message: Optional[str] = None
    summary: Optional[str] = None
    event_count: int = 0
    error: Optional[str] = None
    error_code: Optional[str] = None

    def __post_init__(self) -> None:
        if self.status not in STATUSES:
            raise ValueError("unknown executor status: %r" % (self.status,))
        if not isinstance(self.event_count, int) or isinstance(self.event_count, bool):
            raise ValueError("event_count must be an integer")
        if self.event_count < 0:
            raise ValueError("event_count cannot be negative")
        if self.session_id is not None and not (
            isinstance(self.session_id, str) and self.session_id
        ):
            raise ValueError("session_id must be a non-empty string or None")
        if self.status == STATUS_COMPLETED and not self.session_id:
            raise ValueError("a completed executor result requires a session id")
        if self.status == STATUS_FAILED and not (self.error or self.error_code):
            raise ValueError("a failed executor result requires error evidence")

    @property
    def ok(self) -> bool:
        """True only for a genuinely completed turn."""

        return self.status == STATUS_COMPLETED

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "session_id": self.session_id,
            "final_message": self.final_message,
            "summary": self.summary,
            "event_count": self.event_count,
            "error": self.error,
            "error_code": self.error_code,
        }


def validate_argv(argv) -> List[str]:
    """Return a plain argv list, refusing shell strings and empty tokens."""

    if isinstance(argv, (str, bytes)) or not isinstance(argv, (list, tuple)):
        raise ExecutorConfigError("command must be an argv array, never a shell string")
    parts = list(argv)
    if not parts:
        raise ExecutorConfigError("command argv cannot be empty")
    for part in parts:
        if not isinstance(part, str) or not part:
            raise ExecutorConfigError("every argv token must be a non-empty string")
    return parts


def validate_prompt(prompt, *, label: str = "prompt") -> str:
    if not isinstance(prompt, str) or not prompt.strip():
        raise ExecutorConfigError("%s must be a non-empty string" % label)
    return prompt


def validate_workdir(workdir) -> str:
    if not isinstance(workdir, str) or not workdir:
        raise ExecutorConfigError("workdir must be an absolute path string")
    if not workdir.startswith("/"):
        raise ExecutorConfigError("workdir must be an absolute path: %r" % (workdir,))
    return workdir


def validate_session_id(session_id, *, label: str = "session id") -> str:
    if not isinstance(session_id, str) or not session_id:
        raise ExecutorConfigError("%s must be a non-empty string" % label)
    if not _SAFE_SESSION_ID.fullmatch(session_id):
        raise ExecutorConfigError(
            "%s is not a safe argv token: %r" % (label, session_id)
        )
    return session_id


def validate_exit_code(exit_code) -> int:
    if not isinstance(exit_code, int) or isinstance(exit_code, bool):
        raise ExecutorConfigError("exit_code must be an integer")
    return exit_code


def check_line_source(lines) -> Iterable:
    """Refuse a single blob where a line iterable was promised."""

    if isinstance(lines, (str, bytes)):
        raise ExecutorConfigError(
            "stream must be an iterable of JSONL lines, not one string"
        )
    try:
        iter(lines)
    except TypeError:
        raise ExecutorConfigError("stream must be an iterable of JSONL lines") from None
    return lines


class StreamCollector:
    """Decode JSONL lines, counting valid events and recording bad lines.

    Malformed lines are tolerated in the sense that they never raise, but they
    are always recorded: an adapter that saw a corrupted stream cannot report
    an honest success.
    """

    def __init__(self) -> None:
        self.malformed: List[str] = []
        self.event_count = 0

    def read(self, lines) -> Iterator[dict]:
        for number, raw in enumerate(lines, 1):
            if isinstance(raw, bytes):
                try:
                    raw = raw.decode("utf-8")
                except UnicodeDecodeError:
                    self.malformed.append("line %d: stream line is not UTF-8" % number)
                    continue
            if not isinstance(raw, str):
                self.malformed.append("line %d: stream item is not text" % number)
                continue
            text = raw.strip()
            if not text:
                continue
            try:
                payload = json.loads(text)
            except ValueError as error:
                self.malformed.append("line %d: malformed JSON (%s)" % (number, error))
                continue
            if not isinstance(payload, dict):
                self.malformed.append("line %d: event is not a JSON object" % number)
                continue
            self.event_count += 1
            yield payload


def clip(text: Optional[str], limit: int = _SUMMARY_LIMIT) -> Optional[str]:
    """First line of ``text``, trimmed to a bounded summary."""

    if not isinstance(text, str):
        return None
    first = text.strip().splitlines()[0].strip() if text.strip() else ""
    if not first:
        return None
    if len(first) <= limit:
        return first
    return first[: limit - 3].rstrip() + "..."


class ExecutorAdapter:
    """Base class: argv builders plus a JSONL stream parser."""

    name = "executor"
    binary = ""

    # -- argv -----------------------------------------------------------

    def build_initial_argv(self, prompt, *, workdir: Optional[str] = None) -> List[str]:
        raise NotImplementedError

    def build_resume_argv(
        self, session_id, prompt, *, workdir: Optional[str] = None
    ) -> List[str]:
        raise NotImplementedError

    def build_queue_argv(self, session_id, prompt) -> List[str]:
        raise FollowUpNotSupportedError(
            "%s cannot take an active follow-up without replacing the exact session"
            % self.name
        )

    def queued_follow_up(self, session_id, prompt) -> ExecutorResult:
        """Describe a follow-up handed to the CLI's own queue, not a new run."""

        session_id = validate_session_id(session_id)
        prompt = validate_prompt(prompt, label="follow-up")
        self.build_queue_argv(session_id, prompt)
        return ExecutorResult(
            status=STATUS_QUEUED,
            session_id=session_id,
            summary="follow-up queued for the active session; it is not mid-turn steering",
        )

    # -- parsing --------------------------------------------------------

    def parse_stream(
        self,
        lines,
        *,
        expected_session_id: Optional[str] = None,
        exit_code: int = 0,
    ) -> ExecutorResult:
        raise NotImplementedError

    # -- result helpers -------------------------------------------------

    def _result(
        self,
        status: str,
        *,
        session_id: Optional[str] = None,
        final_message: Optional[str] = None,
        collector: Optional[StreamCollector] = None,
        error: Optional[str] = None,
        error_code: Optional[str] = None,
        summary: Optional[str] = None,
    ) -> ExecutorResult:
        if summary is None:
            if status == STATUS_COMPLETED:
                summary = clip(final_message) or "completed"
            else:
                summary = clip(error)
        return ExecutorResult(
            status=status,
            session_id=session_id,
            final_message=final_message,
            summary=summary,
            event_count=collector.event_count if collector is not None else 0,
            error=error,
            error_code=error_code,
        )


__all__ = [
    "ExecutorAdapter",
    "ExecutorConfigError",
    "ExecutorResult",
    "FollowUpNotSupportedError",
    "STATUSES",
    "STATUS_COMPLETED",
    "STATUS_FAILED",
    "STATUS_NEEDS_APPROVAL",
    "STATUS_NEEDS_INPUT",
    "STATUS_QUEUED",
    "StreamCollector",
    "check_line_source",
    "clip",
    "validate_argv",
    "validate_exit_code",
    "validate_prompt",
    "validate_session_id",
    "validate_workdir",
]
