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
* a failed opener is not recorded, so a retry is allowed;
* a skip is a clean result, never an error and never an implicit success.

Opening a file is presentation, not proof. This module never changes a task's
state, and callers must not treat a successful open as evidence of anything.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from typing import Callable, Iterable, Mapping, Optional, Tuple

from artifacts import ArtifactError, verify_artifacts

Opener = Callable[[str], object]

_COMPLETED = "completed"


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


class PresentationService:
    """Open the primary verified artifact of a completed task, once per version."""

    def __init__(
        self,
        opener: Opener,
        *,
        allowed_extensions: Optional[Iterable[str]] = None,
    ) -> None:
        if not callable(opener):
            raise TypeError("presentation opener must be a callable")
        self._opener = opener
        self._allowed_extensions = (
            None if allowed_extensions is None else tuple(allowed_extensions)
        )
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
        key = (task_id, version)
        with self._lock:
            if key in self._presented:
                return PresentationResult(
                    False,
                    reason="already presented for this task version",
                    task_id=task_id,
                    version=version,
                )
            try:
                self._opener(str(primary.path))
            except Exception as error:  # a failed open is a skip, not a presented task
                return PresentationResult(
                    False,
                    reason="opener failed (%s)" % type(error).__name__,
                    task_id=task_id,
                    version=version,
                )
            self._presented.add(key)
        return PresentationResult(
            True,
            path=str(primary.path),
            reason="presented",
            task_id=task_id,
            version=version,
        )


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


__all__ = ["PresentationResult", "PresentationService"]
