"""Codex CLI executor: exact-thread runs, resumes, and queued follow-ups.

Initial and resumed runs pin the thread by its reported id, never by
recency. ``--last`` is never emitted: several NEO tasks may run at once and
recency would cross-route them. An active follow-up goes through
``codex queue`` and is reported as ``queued`` because the CLI queues it rather
than steering the turn in progress.
"""

from __future__ import annotations

from typing import List, Optional

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
    validate_prompt,
    validate_session_id,
    validate_workdir,
)

SANDBOX_MODE = "workspace-write"
_ITEM_EVENTS = frozenset({"item.started", "item.updated", "item.completed"})


class CodexExecutor(ExecutorAdapter):
    """Build and interpret ``codex exec --json`` for one exact thread."""

    name = "codex"

    def __init__(self, binary: str = "codex") -> None:
        if not isinstance(binary, str) or not binary:
            raise ExecutorConfigError("codex binary must be a non-empty string")
        self.binary = binary

    # -- argv -----------------------------------------------------------

    def _parent_flags(self, workdir) -> List[str]:
        return [
            self.binary,
            "exec",
            "--json",
            "--sandbox",
            SANDBOX_MODE,
            "--skip-git-repo-check",
            "--cd",
            validate_workdir(workdir),
        ]

    def build_initial_argv(self, prompt, *, workdir: Optional[str] = None) -> List[str]:
        prompt = validate_prompt(prompt)
        argv = self._parent_flags(workdir) + ["--", prompt]
        return validate_argv(argv)

    def build_resume_argv(
        self, session_id, prompt, *, workdir: Optional[str] = None
    ) -> List[str]:
        thread_id = validate_session_id(session_id, label="thread id")
        prompt = validate_prompt(prompt, label="follow-up")
        if prompt.startswith("-"):
            raise ExecutorConfigError(
                "follow-up must not begin with '-' because codex exec resume takes "
                "the prompt positionally without a terminator"
            )
        argv = self._parent_flags(workdir) + ["resume", thread_id, prompt]
        return validate_argv(argv)

    def build_queue_argv(self, session_id, prompt) -> List[str]:
        thread_id = validate_session_id(session_id, label="thread id")
        prompt = validate_prompt(prompt, label="follow-up")
        return validate_argv(
            [self.binary, "queue", "--thread", thread_id, "--message", prompt]
        )

    # -- parsing --------------------------------------------------------

    def parse_stream(
        self,
        lines,
        *,
        expected_session_id: Optional[str] = None,
        exit_code: int = 0,
    ) -> ExecutorResult:
        expected = (
            validate_session_id(expected_session_id, label="expected thread id")
            if expected_session_id is not None
            else None
        )
        exit_code = validate_exit_code(exit_code)
        collector = StreamCollector()

        thread_id: Optional[str] = None
        conflict: Optional[tuple] = None
        completed = False
        failure: Optional[tuple] = None
        final_message: Optional[str] = None
        notices: List[str] = []

        for event in collector.read(check_line_source(lines)):
            kind = event.get("type")
            if not isinstance(kind, str) or not kind:
                collector.malformed.append("event without a string type")
                continue

            if kind == "thread.started":
                value = event.get("thread_id")
                if not isinstance(value, str) or not value.strip():
                    collector.malformed.append(
                        "thread.started event without a usable thread_id"
                    )
                    continue
                value = value.strip()
                if thread_id is None:
                    thread_id = value
                elif value != thread_id and conflict is None:
                    conflict = (thread_id, value)
            elif kind == "turn.completed":
                completed = True
            elif kind == "turn.failed":
                message = _error_message(event.get("error")) or (
                    "codex reported turn.failed"
                )
                failure = ("turn_failed", message)
            elif kind == "error":
                message = event.get("message")
                if isinstance(message, str) and message.strip():
                    notices.append(message.strip())
            elif kind in _ITEM_EVENTS:
                item = event.get("item")
                if not isinstance(item, dict):
                    collector.malformed.append(
                        "%s event without an item object" % kind
                    )
                    continue
                item_type = item.get("type")
                if item_type == "agent_message":
                    text = item.get("text")
                    if isinstance(text, str) and text.strip():
                        final_message = text.strip()
                elif item_type == "error":
                    message = item.get("message")
                    if isinstance(message, str) and message.strip():
                        notices.append(message.strip())

        if collector.malformed:
            return self._result(
                STATUS_FAILED,
                session_id=thread_id,
                final_message=final_message,
                collector=collector,
                error=collector.malformed[0],
                error_code="malformed_stream",
            )

        if conflict is not None:
            return self._result(
                STATUS_FAILED,
                session_id=thread_id,
                final_message=final_message,
                collector=collector,
                error=(
                    "conflicting thread.started events: %s then %s; "
                    "a run must report exactly one thread" % conflict
                ),
                error_code="thread_conflict",
            )

        if expected is not None and thread_id is not None and thread_id != expected:
            return self._result(
                STATUS_FAILED,
                session_id=thread_id,
                final_message=final_message,
                collector=collector,
                error=(
                    "returned thread id %s does not equal expected thread id %s"
                    % (thread_id, expected)
                ),
                error_code="session_mismatch",
            )

        if failure is not None:
            return self._result(
                STATUS_FAILED,
                session_id=thread_id,
                final_message=final_message,
                collector=collector,
                error=failure[1],
                error_code=failure[0],
            )

        if exit_code != 0:
            detail = "codex exited with status %d before reporting a terminal turn" % (
                exit_code,
            )
            if notices:
                detail = "%s (%s)" % (detail, notices[0])
            return self._result(
                STATUS_FAILED,
                session_id=thread_id,
                final_message=final_message,
                collector=collector,
                error=detail,
                error_code="exit_nonzero",
            )

        if not completed:
            detail = (
                "codex stream ended without turn.completed; completion is never "
                "inferred from exit status or silence"
            )
            if notices:
                detail = "%s (%s)" % (detail, notices[0])
            return self._result(
                STATUS_FAILED,
                session_id=thread_id,
                final_message=final_message,
                collector=collector,
                error=detail,
                error_code="incomplete_stream",
            )

        if thread_id is None:
            return self._result(
                STATUS_FAILED,
                final_message=final_message,
                collector=collector,
                error=(
                    "turn.completed arrived without thread.started; no thread id "
                    "was reported, so the thread cannot be resumed"
                ),
                error_code="missing_session_id",
            )

        return self._result(
            STATUS_COMPLETED,
            session_id=thread_id,
            final_message=final_message,
            collector=collector,
        )


def _error_message(value) -> Optional[str]:
    if isinstance(value, dict):
        message = value.get("message")
        if isinstance(message, str) and message.strip():
            return message.strip()
        return clip(str(value))
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


__all__ = ["CodexExecutor", "SANDBOX_MODE"]
