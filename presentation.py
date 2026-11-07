"""Presentation policy for NEO results.

Presenting a finished result means opening one verified local file for the user.
Nothing else. This module deliberately knows no publish, send, post or browser
surface: the only capability it holds is an ``opener`` callable supplied by the
caller, and it decides whether to call it at all.

The policy, from the implementation plan (section 4.6):

* the task must be ``completed``;
* completion must be backed by persisted evidence — non-empty ``expected_json``
  and an ``artifacts_json`` list that covers every expected name;
* every expected artifact must still verify beneath the recorded workdir at
  presentation time, because a file can move or become a symlink after the
  worker reported completion;
* the primary artifact (the first expected name) is opened, at most once per
  ``(task_id, version)``, including when calls arrive concurrently;
* when a durable ``ledger`` is supplied, the claim is written *before* the
  opener is called, so idempotence survives a restart and two service
  instances; a failed opener releases its claim and a retry can present;
* a failed opener is not recorded, so a retry is allowed;
* a skip is a clean result, never an error and never an implicit success.

Opening a file is presentation, not proof. This module never changes a task's
state, and callers must not treat a successful open as evidence of anything.
The production opener is an argv-only ``open -- <path>`` call with a timeout;
no shell is ever used and no publication, send or approval surface exists.
"""

from __future__ import annotations

import json
import subprocess
import threading
from dataclasses import dataclass
from typing import Callable, Iterable, Mapping, Optional, Tuple

from artifacts import ArtifactError, verify_artifacts
from registry import TaskNotFoundError

Opener = Callable[[str], object]

_COMPLETED = "completed"
_DEFAULT_OPEN_TIMEOUT = 10.0


class PresentationOpenError(RuntimeError):
    """The macOS opener could not open the verified artifact."""


@dataclass(frozen=True)
class PresentationResult:
    """The outcome of one presentation attempt."""

    presented: bool
    path: Optional[str] = None
    reason: str = ""
    task_id: str = ""
    version: int = 0

    def as_dict(self) -> dict:
        return {
            "presented": self.presented,
            "path": self.path,
            "reason": self.reason,
            "task_id": self.task_id,
            "version": self.version,
        }


class MacOSOpener:
    """Open one absolute path with ``open -- <path>``, never through a shell.

    ``run`` is injectable so tests can prove the argv shape without invoking
    the real macOS opener.
    """

    def __init__(self, *, timeout: float = _DEFAULT_OPEN_TIMEOUT, run=None) -> None:
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
            raise ValueError("opener timeout must be a positive number of seconds")
        if timeout <= 0:
            raise ValueError("opener timeout must be a positive number of seconds")
        self._timeout = float(timeout)
        self._run = run or subprocess.run

    def __call__(self, path: str) -> object:
        if not isinstance(path, str) or not path or "\x00" in path:
            raise ValueError("presentation path must be a non-empty string")
        if not path.startswith("/"):
            raise ValueError("presentation path must be absolute")
        try:
            completed = self._run(
                ["open", "--", path],
                shell=False,
                check=False,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                timeout=self._timeout,
            )
        except subprocess.TimeoutExpired as error:
            raise PresentationOpenError("the macOS opener timed out") from error
        except OSError as error:
            raise PresentationOpenError(
                "the macOS opener could not start: %s" % error
            ) from error
        status = getattr(completed, "returncode", None)
        if status != 0:
            raise PresentationOpenError(
                "the macOS opener exited with status %s" % (status,)
            )
        return completed


class PresentationService:
    """Open the primary verified artifact of a completed task, once per version."""

    def __init__(
        self,
        opener: Opener,
        *,
        allowed_extensions: Optional[Iterable[str]] = None,
        ledger: object = None,
    ) -> None:
        if not callable(opener):
            raise TypeError("presentation opener must be a callable")
        if ledger is not None and not all(
            callable(getattr(ledger, method, None))
            for method in ("claim", "finish", "release")
        ):
            raise TypeError(
                "presentation ledger must provide claim(), finish() and release()"
            )
        self._opener = opener
        self._allowed_extensions = (
            None if allowed_extensions is None else tuple(allowed_extensions)
        )
        self._ledger = ledger
        self._lock = threading.Lock()
        self._presented: set = set()

    def present(self, task: Mapping[str, object]) -> PresentationResult:
        """Decide and, when allowed, open one artifact. Never raises for bad tasks."""
        if not isinstance(task, Mapping):
            raise TypeError("task must be a mapping of persisted task fields")

        task_id = _text(task.get("id") or task.get("task_id"))
        version = task.get("version")
        if not task_id:
            return PresentationResult(False, reason="task id is not recorded")
        if (
            not isinstance(version, int)
            or isinstance(version, bool)
            or version < 0
        ):
            return PresentationResult(
                False, reason="task version is not recorded", task_id=task_id
            )

        state = _text(task.get("state"))
        if state != _COMPLETED:
            return PresentationResult(
                False,
                reason="task state is %r, not completed" % (state or "unknown"),
                task_id=task_id,
                version=version,
            )

        try:
            expected = _persisted_names(task.get("expected_json"), "expected_json")
            verified_names = _persisted_names(task.get("artifacts_json"), "artifacts_json")
        except ValueError:
            return PresentationResult(
                False,
                reason="task completion evidence is not readable",
                task_id=task_id,
                version=version,
            )

        if not expected:
            return PresentationResult(
                False,
                reason="task recorded no expected artifacts, so there is nothing to open",
                task_id=task_id,
                version=version,
            )

        unverified = sorted(set(expected) - set(verified_names))
        if unverified:
            return PresentationResult(
                False,
                reason="expected artifacts were not verified by the executor: %s"
                % ", ".join(unverified),
                task_id=task_id,
                version=version,
            )

        try:
            checked = verify_artifacts(
                task.get("workdir"),
                expected,
                allowed_extensions=self._allowed_extensions,
            )
        except ArtifactError as error:
            return PresentationResult(
                False,
                reason="artifact verification failed: %s" % error,
                task_id=task_id,
                version=version,
            )

        primary = checked[0]
        path = str(primary.path)
        key = (task_id, version)
        with self._lock:
            if key in self._presented:
                return PresentationResult(
                    False,
                    reason="already presented for this task version",
                    task_id=task_id,
                    version=version,
                )
            if self._ledger is not None:
                try:
                    claimed = self._ledger.claim(task_id, version, path)
                except Exception as error:
                    return PresentationResult(
                        False,
                        reason="presentation could not be recorded (%s)"
                        % type(error).__name__,
                        task_id=task_id,
                        version=version,
                    )
                if not claimed:
                    return PresentationResult(
                        False,
                        reason="already presented for this task version",
                        task_id=task_id,
                        version=version,
                    )
            try:
                self._opener(path)
            except Exception as error:  # a failed open is a skip, not a presented task
                if self._ledger is not None:
                    try:
                        self._ledger.release(task_id, version)
                    except Exception:
                        # A failed release leaves the claim held, so retries
                        # remain refused rather than risking a second open.
                        pass
                return PresentationResult(
                    False,
                    reason="opener failed (%s)" % type(error).__name__,
                    task_id=task_id,
                    version=version,
                )
            if self._ledger is not None:
                try:
                    self._ledger.finish(task_id, version)
                except Exception:
                    # The file is already open; durability recovery is the
                    # ledger's problem, not a reason to claim failure.
                    pass
            self._presented.add(key)
        return PresentationResult(
            True,
            path=path,
            reason="presented",
            task_id=task_id,
            version=version,
        )


class TaskPresenter:
    """Read a task and present it only when it is verified and in scope.

    The presenter owns the small amount of policy the HTTP surface needs:
    the task is read from durable state, an optional conversation key must
    match the task's immutable key, and the result is a JSON-safe report of
    ``presented`` / ``skipped`` / ``reason``. It never mutates the task.
    """

    def __init__(self, registry: object, presentation: object) -> None:
        if not callable(getattr(registry, "get_task", None)):
            raise TypeError("task presenter requires a registry with get_task()")
        if not callable(getattr(presentation, "present", None)):
            raise TypeError("task presenter requires a presentation service")
        self._registry = registry
        self._presentation = presentation

    def present_task(
        self, task_id: object, conversation_key: Optional[str] = None
    ) -> dict:
        """Return the presentation report, raising ``TaskNotFoundError`` on scope."""
        if not isinstance(task_id, str) or not task_id.strip():
            raise TaskNotFoundError(str(task_id))
        task = self._registry.get_task(task_id)
        if not isinstance(task, Mapping) or not task.get("id"):
            raise TaskNotFoundError(task_id)
        if conversation_key is not None:
            if (
                not isinstance(conversation_key, str)
                or not conversation_key.strip()
                or task.get("conversation_key") != conversation_key
            ):
                raise TaskNotFoundError(task_id)
        result = self._presentation.present(task)
        report = result.as_dict()
        report["skipped"] = not report["presented"]
        return report


def _text(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""


def _persisted_names(raw: object, field: str) -> Tuple[str, ...]:
    """Decode a persisted JSON string list, refusing anything malformed."""
    if raw is None:
        return ()
    if not isinstance(raw, str):
        raise ValueError("%s must be a JSON string" % field)
    try:
        value = json.loads(raw)
    except ValueError as error:
        raise ValueError("%s is not valid JSON: %s" % (field, error)) from None
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item.strip() for item in value
    ):
        raise ValueError("%s must be a JSON list of non-empty strings" % field)
    return tuple(value)


__all__ = [
    "MacOSOpener",
    "PresentationOpenError",
    "PresentationResult",
    "PresentationService",
    "TaskPresenter",
]
