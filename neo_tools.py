"""Worker-facing tool boundary for NEO.

A worker process holds ``NEO_TASK_ID`` and a random ``NEO_TASK_TOKEN``. The
plugin stores only a SHA-256 digest of that token, so a database read never
yields a usable credential. Every worker transition verifies the digest inside
the same transaction as its state change, and a rejected token changes nothing.

Two layers live here:

* ``WorkerTokenStore`` issues and verifies per-task tokens;
* ``WorkerTaskTools`` is the original authenticated update call;
* ``WorkerBridge`` is the scoped set of calls a worker may make: update,
  heartbeat, preview publish, completion and a read of its own task.

The bridge authorises before it reads anything, so an unknown task and a wrong
token are indistinguishable. Completion is proved by ``artifacts``-verified
files beneath the task workdir; a worker's assertion or a process exit is never
enough. Preview publication copies one allowlisted image into a task-scoped
directory under a byte cap and records the revision and the copy in one
transaction, so two concurrent publishes cannot claim the same revision.

This module knows nothing about process management or supervisors.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
import re
import secrets
import shutil
import stat
import time
from pathlib import Path
from types import MappingProxyType
from typing import Optional, Union
from uuid import uuid4

from artifacts import (
    ArtifactEscapesWorkdirError,
    ArtifactNotFoundError,
    EmptyExpectedArtifactsError,
    MissingWorkdirError,
    verify_artifact,
    completion_evidence,
)
from events import TaskService
from registry import InvalidWorkerTokenError, Registry, VersionConflictError
from state_machine import Evidence, TaskState

TOKEN_BYTES = 32

WORKER_TOKEN_REJECTED = "worker token rejected"
"""The one message a refused worker call may show: it names no task and no path."""

DEFAULT_PREVIEW_MAX_BYTES = 2 * 1024 * 1024
"""Two mebibytes, the default cap for one published preview image."""

PREVIEW_MIME_TYPES = MappingProxyType(
    {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
    }
)
"""The only image types NEO will copy into a preview directory."""

PREVIEW_EVENT = "task.preview_updated"

_PREVIEW_DIR_ENV = "NEO_PREVIEW_DIR"
_PREVIEW_MAX_BYTES_ENV = "NEO_PREVIEW_MAX_BYTES"
_DEFAULT_PREVIEW_DIR = Path.home() / ".hermes" / "neo" / "previews"
_SAFE_TASK_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

_WORKER_TASK_FIELDS = (
    "id",
    "parent_id",
    "kind",
    "intent",
    "state",
    "capability_id",
    "executor",
    "executor_session_id",
    "workdir",
    "resource_key",
    "preview_path",
    "preview_rev",
    "result_summary",
    "error_code",
    "error_detail",
    "heartbeat_at",
    "created_at",
    "updated_at",
    "started_at",
    "finished_at",
    "version",
)
"""Fields a worker may read back; conversation routing and tokens stay server-side."""


class WorkerToolError(RuntimeError):
    """Base class for a refused worker call."""


class PreviewTooLargeError(WorkerToolError):
    """The candidate preview image exceeds the configured byte cap."""


def _digest(token: str) -> str:
    """Return the stored form of a worker token: its SHA-256 hex digest."""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _safe_task_component(task_id: str) -> str:
    """Return a filesystem-safe directory name for a task id.

    A well-formed id is used as-is. Anything else is replaced by a digest, so a
    hostile id can never name a directory outside the preview root.
    """
    if _SAFE_TASK_ID.fullmatch(task_id):
        return task_id
    return "task-" + hashlib.sha256(task_id.encode("utf-8")).hexdigest()[:32]


def _is_beneath(path: Path, root: Path) -> bool:
    """Return whether ``path`` is ``root`` or inside it, without following links."""
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _stored_expected_names(task: dict) -> tuple:
    """Decode a task's persisted ``expected_json`` into relative names."""
    raw = task.get("expected_json") or "[]"
    try:
        value = json.loads(raw)
    except (TypeError, ValueError) as error:
        raise WorkerToolError(
            "task expected artifacts are not readable: %s" % error
        ) from error
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise WorkerToolError(
            "task expected artifacts must be a list of relative names"
        )
    return tuple(value)


def _supplied_names(expected_artifacts: object) -> tuple:
    """Coerce a caller's optional expectation, treating a bare string as one name."""
    if expected_artifacts is None:
        return ()
    if isinstance(expected_artifacts, str):
        names = (expected_artifacts,)
    else:
        try:
            names = tuple(expected_artifacts)  # type: ignore[arg-type]
        except TypeError:
            raise WorkerToolError(
                "expected_artifacts must be a list of relative names"
            ) from None
    for name in names:
        if not isinstance(name, str) or not name.strip():
            raise WorkerToolError(
                "expected_artifacts entries must be non-empty relative names"
            )
    return names


def _relative_source(workdir: object, source_path: object) -> str:
    """Return a workdir-relative source name, refusing anything outside it.

    An absolute path is accepted only when its parent resolves beneath the task
    workdir; the leaf is left unresolved so a symlink leaf is still refused by
    the artifact walk.
    """
    if not isinstance(source_path, str) or not source_path.strip():
        raise WorkerToolError("source_path must be a non-empty string")
    text = source_path.strip()
    if "\x00" in text:
        raise ArtifactEscapesWorkdirError(
            "preview source contains a null byte and was refused"
        )
    if not os.path.isabs(text):
        return text
    if workdir is None:
        raise MissingWorkdirError(
            "task workdir is missing; a preview source cannot be contained"
        )
    try:
        root = Path(workdir).resolve(strict=True)
        parent = Path(text).parent.resolve(strict=True)
    except (OSError, RuntimeError, ValueError):
        raise MissingWorkdirError(
            "task workdir or preview source cannot be resolved"
        ) from None
    try:
        relative_parent = parent.relative_to(root)
    except ValueError:
        raise ArtifactEscapesWorkdirError(
            "preview source must be a file beneath the task workdir"
        ) from None
    return (relative_parent / Path(text).name).as_posix()


class WorkerTokenStore:
    """Issue and verify per-task worker tokens without persisting them."""

    def __init__(self, registry: Registry) -> None:
        self._registry = registry

    def issue(self, task_id: str) -> str:
        """Return a fresh raw token for ``task_id`` and store only its digest.

        Issuing again rotates the token: the previous one is invalidated.
        """
        token = secrets.token_urlsafe(TOKEN_BYTES)
        self._registry.store_worker_token(task_id, _digest(token))
        return token

    def verify(self, task_id: str, token: object) -> bool:
        """Return whether ``token`` is the current token for ``task_id``."""
        if not isinstance(task_id, str) or not task_id:
            return False
        if not isinstance(token, str) or not token:
            return False
        stored = self._registry.worker_token_hash(task_id)
        if stored is None:
            return False
        return hmac.compare_digest(stored, _digest(token))


class WorkerTaskTools:
    """The tools a worker process is allowed to call."""

    def __init__(self, service: TaskService, tokens: WorkerTokenStore) -> None:
        self._service = service
        self._tokens = tokens

    @staticmethod
    def _validate_text(value: object, name: str) -> Optional[str]:
        if value is not None and (
            not isinstance(value, str) or len(value) > 4000
        ):
            raise WorkerToolError(
                "%s must be a string of at most 4000 characters" % name
            )
        return value

    def neo_task_update(
        self,
        task_id: str,
        token: Optional[str],
        state: Union[TaskState, str],
        expected_version: int,
        progress: object = None,
        detail: object = None,
        summary: object = None,
        evidence: Optional[Evidence] = None,
    ) -> dict:
        """Apply one authenticated worker transition and optional progress event.

        The token digest is checked inside the state-change transaction. A token
        rotated before or during this call therefore cannot commit stale work.
        """
        if not isinstance(token, str) or not token:
            raise InvalidWorkerTokenError(
                "worker token rejected for task %s" % task_id
            )
        detail = self._validate_text(detail, "detail")
        summary = self._validate_text(summary, "summary")
        if progress is not None:
            if isinstance(progress, bool) or not isinstance(progress, (int, float)):
                raise WorkerToolError("progress must be a finite number")
            if isinstance(progress, float) and not math.isfinite(progress):
                raise WorkerToolError("progress must be a finite number")
        event_extra = {}
        if progress is not None:
            event_extra["progress"] = progress
        if detail is not None:
            event_extra["detail"] = detail
        return self._service.transition(
            task_id,
            state,
            expected_version=expected_version,
            summary=summary,
            evidence=evidence,
            event_extra=event_extra or None,
            worker_token_digest=_digest(token),
        )

    def worker_task_update(
        self,
        task_id: str,
        token: Optional[str],
        target: Union[TaskState, str],
        expected_version: int,
        summary: Optional[str] = None,
        evidence: Optional[Evidence] = None,
    ) -> dict:
        """Compatibility wrapper for the original internal worker update name."""
        return self.neo_task_update(
            task_id,
            token,
            state=target,
            expected_version=expected_version,
            summary=summary,
            evidence=evidence,
        )


class WorkerBridge:
    """The scoped tool surface a worker process may call.

    Every method authorises the per-task token before reading any task data and
    raises ``InvalidWorkerTokenError`` with one identical message for a wrong,
    missing, rotated or other-task token and for an unknown task. Refusals never
    create a directory, a file, an event or a row.
    """

    def __init__(
        self,
        service: TaskService,
        tokens: WorkerTokenStore,
        *,
        preview_root: Optional[Union[str, Path]] = None,
        preview_max_bytes: Optional[int] = None,
    ) -> None:
        self._service = service
        self._tokens = tokens
        self._updates = WorkerTaskTools(service, tokens)
        if preview_root is None:
            preview_root = os.environ.get(_PREVIEW_DIR_ENV) or _DEFAULT_PREVIEW_DIR
        self._preview_root = Path(preview_root).expanduser()
        if preview_max_bytes is None:
            preview_max_bytes = _configured_preview_max_bytes()
        if (
            isinstance(preview_max_bytes, bool)
            or not isinstance(preview_max_bytes, int)
            or preview_max_bytes < 1
        ):
            raise ValueError("preview_max_bytes must be a positive integer")
        self._preview_max_bytes = preview_max_bytes

    # -- authorisation ---------------------------------------------------

    def _authorise(self, task_id: object, token: object) -> str:
        """Return the token digest, or refuse with the one indistinguishable error."""
        if not self._tokens.verify(task_id, token):
            raise InvalidWorkerTokenError(WORKER_TOKEN_REJECTED)
        return _digest(token)

    def _own_task(self, task_id: str) -> dict:
        """Return the caller's task row or refuse without revealing existence."""
        task = self._service.get_task(task_id)
        if task is None:
            raise InvalidWorkerTokenError(WORKER_TOKEN_REJECTED)
        return task

    # -- update ----------------------------------------------------------

    def neo_task_update(
        self,
        task_id: str,
        token: Optional[str],
        state: Union[TaskState, str],
        expected_version: Optional[int] = None,
        progress: object = None,
        detail: object = None,
        summary: object = None,
    ) -> dict:
        """Apply one authenticated transition to the caller's own task.

        ``expected_version`` is optional for workers: when omitted the task's
        current version is used, and a concurrent change raises
        ``VersionConflictError`` rather than committing stale work.
        """
        self._authorise(task_id, token)
        if expected_version is None:
            expected_version = self._own_task(task_id)["version"]
        return self._updates.neo_task_update(
            task_id,
            token,
            state=state,
            expected_version=expected_version,
            progress=progress,
            detail=detail,
            summary=summary,
        )

    # -- heartbeat -------------------------------------------------------

    def neo_task_heartbeat(self, task_id: str, token: Optional[str]) -> dict:
        """Advance ``heartbeat_at`` without touching state, events or outputs.

        The row version is advanced by the registry's optimistic update, so a
        supervisor holding a stale version notices, but lifecycle state and the
        event stream are untouched.
        """
        self._authorise(task_id, token)
        last_error = None
        for _ in range(4):
            task = self._own_task(task_id)
            try:
                updated = self._service.registry.update_task(
                    task_id,
                    expected_version=task["version"],
                    heartbeat_at=time.time(),
                )
            except VersionConflictError as error:
                last_error = error
                continue
            return {
                "task_id": updated["id"],
                "state": updated["state"],
                "heartbeat_at": updated["heartbeat_at"],
                "version": updated["version"],
            }
        raise last_error  # type: ignore[misc]

    # -- completion ------------------------------------------------------

    def neo_task_complete(
        self,
        task_id: str,
        token: Optional[str],
        expected_artifacts: object = None,
        summary: Optional[str] = None,
    ) -> dict:
        """Complete the caller's task only after every expected artifact verifies.

        The expectation is the task's persisted ``expected_json`` plus any names
        supplied by the worker. ``artifacts.completion_evidence`` walks each name
        beneath the task workdir and returns canonical relative names, which the
        registry persists in the same transaction as the completion event. With
        no stored and no supplied expectation the call is refused: "nothing to
        present" must be explicit, never inferred.
        """
        digest = self._authorise(task_id, token)
        task = self._own_task(task_id)
        names = tuple(
            dict.fromkeys(
                (*_stored_expected_names(task), *_supplied_names(expected_artifacts))
            )
        )
        evidence = completion_evidence(task.get("workdir"), names)
        summary = WorkerTaskTools._validate_text(summary, "summary")

        last_error = None
        for _ in range(4):
            current = self._own_task(task_id)
            try:
                return self._service.transition(
                    task_id,
                    "completed",
                    expected_version=current["version"],
                    summary=summary,
                    evidence=evidence,
                    worker_token_digest=digest,
                )
            except VersionConflictError as error:
                last_error = error
        raise last_error  # type: ignore[misc]

    # -- preview publish -------------------------------------------------

    def _preview_directory(self, task_id: str) -> Path:
        """Create and return the caller's own preview directory, contained."""
        self._preview_root.mkdir(parents=True, exist_ok=True)
        root = self._preview_root.resolve(strict=True)
        directory = self._preview_root / _safe_task_component(task_id)
        directory.mkdir(parents=True, exist_ok=True)
        resolved = directory.resolve(strict=True)
        if not _is_beneath(resolved, root):
            raise WorkerToolError("preview directory escaped its root and was refused")
        return resolved

    def neo_task_publish_preview(
        self,
        task_id: str,
        token: Optional[str],
        source_path: object,
    ) -> dict:
        """Copy one image into the caller's preview directory as a new revision.

        The revision is allocated, the file copied and the event inserted while
        one write transaction is held, so two concurrent publishes receive
        distinct revisions and no event names a path that does not exist.
        """
        digest = self._authorise(task_id, token)
        task = self._own_task(task_id)
        name = _relative_source(task.get("workdir"), source_path)
        artifact = verify_artifact(
            task.get("workdir"),
            name,
            allowed_extensions=PREVIEW_MIME_TYPES,
        )
        suffix = artifact.path.suffix.lower()
        mime = PREVIEW_MIME_TYPES[suffix]

        try:
            descriptor = os.open(
                str(artifact.path),
                os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
            )
        except OSError as error:
            raise ArtifactNotFoundError(
                "preview source cannot be opened as a regular file"
            ) from error

        preview_root = self._preview_root
        connection = None
        temporary = None
        destination = None
        committed = False
        try:
            info = os.fstat(descriptor)
            if not stat.S_ISREG(info.st_mode):
                raise ArtifactNotFoundError(
                    "preview source is not a regular file"
                )
            if info.st_size > self._preview_max_bytes:
                raise PreviewTooLargeError(
                    "preview image is %d bytes; the limit is %d"
                    % (info.st_size, self._preview_max_bytes)
                )

            connection = self._service.registry._connect()
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM tasks WHERE id = ?", (task_id,)
            ).fetchone()
            if row is None:
                raise InvalidWorkerTokenError(WORKER_TOKEN_REJECTED)
            token_row = connection.execute(
                "SELECT token_hash FROM task_worker_tokens WHERE task_id = ?",
                (task_id,),
            ).fetchone()
            if token_row is None or not hmac.compare_digest(
                token_row["token_hash"], digest
            ):
                raise InvalidWorkerTokenError(WORKER_TOKEN_REJECTED)

            directory = self._preview_directory(task_id)
            revision = int(row["preview_rev"] or 0) + 1
            destination = directory / ("rev-%d%s" % (revision, suffix))
            if not _is_beneath(destination.parent, preview_root.resolve(strict=True)):
                raise WorkerToolError("preview path escaped its root")
            temporary = directory / (".tmp-%s" % uuid4().hex)

            source = os.fdopen(descriptor, "rb")
            descriptor = -1
            with source, temporary.open("wb") as target:
                shutil.copyfileobj(source, target, length=1024 * 1024)
                target.flush()
                os.fsync(target.fileno())
            os.replace(str(temporary), str(destination))
            temporary = None

            now = time.time()
            cursor = connection.execute(
                """
                UPDATE tasks
                SET preview_path = ?, preview_rev = ?, version = version + 1,
                    updated_at = ?
                WHERE id = ? AND version = ?
                """,
                (str(destination), revision, now, task_id, row["version"]),
            )
            if cursor.rowcount != 1:
                raise VersionConflictError(
                    task_id, row["version"], int(row["version"]) + 1
                )
            new_version = int(row["version"]) + 1
            payload = {
                "task_id": task_id,
                "state": row["state"],
                "version": new_version,
                "rev": revision,
                "path": str(destination),
                "mime": mime,
            }
            connection.execute(
                "INSERT INTO task_events (task_id, at, kind, payload_json) "
                "VALUES (?, ?, ?, ?)",
                (task_id, now, PREVIEW_EVENT, json.dumps(payload)),
            )
            connection.execute("COMMIT")
            committed = True
        except Exception:
            if connection is not None:
                try:
                    connection.execute("ROLLBACK")
                except Exception:
                    pass
            for leftover in (temporary, destination if not committed else None):
                if leftover is None:
                    continue
                try:
                    leftover.unlink()
                except OSError:
                    pass
            raise
        finally:
            if descriptor >= 0:
                try:
                    os.close(descriptor)
                except OSError:
                    pass
            if connection is not None:
                connection.close()

        return {
            "task_id": task_id,
            "preview_path": str(destination),
            "preview_rev": revision,
            "mime": mime,
            "bytes": info.st_size,
            "version": new_version,
        }

    # -- read ------------------------------------------------------------

    def neo_task_get(self, task_id: str, token: Optional[str]) -> dict:
        """Return the caller's own task, without routing data or token digests."""
        self._authorise(task_id, token)
        task = self._own_task(task_id)
        view = {field: task.get(field) for field in _WORKER_TASK_FIELDS}
        view["expected_artifacts"] = list(_stored_expected_names(task))
        try:
            artifacts = json.loads(task.get("artifacts_json") or "[]")
        except (TypeError, ValueError):
            artifacts = []
        if not isinstance(artifacts, list):
            artifacts = []
        view["artifacts"] = [
            name for name in artifacts if isinstance(name, str)
        ]
        return view


def _configured_preview_max_bytes() -> int:
    """Read the preview byte cap from the environment, falling back to 2 MiB."""
    raw = os.environ.get(_PREVIEW_MAX_BYTES_ENV)
    if raw is None or not raw.strip():
        return DEFAULT_PREVIEW_MAX_BYTES
    try:
        value = int(raw.strip(), 10)
    except ValueError:
        return DEFAULT_PREVIEW_MAX_BYTES
    return value if value > 0 else DEFAULT_PREVIEW_MAX_BYTES


__all__ = [
    "DEFAULT_PREVIEW_MAX_BYTES",
    "EmptyExpectedArtifactsError",
    "InvalidWorkerTokenError",
    "PREVIEW_EVENT",
    "PREVIEW_MIME_TYPES",
    "PreviewTooLargeError",
    "TOKEN_BYTES",
    "WORKER_TOKEN_REJECTED",
    "WorkerBridge",
    "WorkerTaskTools",
    "WorkerTokenStore",
    "WorkerToolError",
]
