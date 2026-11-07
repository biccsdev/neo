"""Task-scoped preview reading for NEO.

A published preview is a copy of one image, written by the worker bridge into
``<preview_root>/<task>/rev-<n>.<ext>`` and recorded on the task row as
``preview_path`` plus ``preview_rev``. This module turns that record into bytes
for the dashboard, and is the only place that does so.

The rules, from the implementation plan (Task 12, sections 4.1-4.6):

* the preview is looked up by task id and belongs to that task alone; a missing
  or deleted task is unavailable, never another task's image;
* when the caller supplies a conversation key it must equal the task's
  immutable ``conversation_key``;
* the persisted path must be a regular, non-symlink file strictly beneath the
  resolved preview root, with an allowlisted image extension, non-zero size and
  at most ``max_bytes``;
* bytes are read once from an open descriptor and hashed, so a concurrent
  publish can never yield one revision's metadata with another's bytes — the
  read is retried against the newer path, or refused honestly;
* the result is JSON-safe: ``data_base64`` is text, never a binary response
  body, and a refusal never echoes an absolute path. A success names the
  persisted ``preview_rev`` and the task ``version`` it was read at, plus the
  byte count and SHA-256 of exactly the bytes supplied.

No publication, no copy, no state change: reading a preview observes.
"""

from __future__ import annotations

import base64
import hashlib
import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Optional

DEFAULT_PREVIEW_MAX_BYTES = 2 * 1024 * 1024
"""Two mebibytes, the default cap for one preview image."""

PREVIEW_MIME_TYPES = MappingProxyType(
    {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
    }
)
"""The only image types that may be served; mirrors the publish allowlist."""

PREVIEW_DIR_ENV = "NEO_PREVIEW_DIR"
"""Environment override for the preview root, shared with the publish path."""

_DEFAULT_PREVIEW_DIR = Path.home() / ".hermes" / "neo" / "previews"
_CHUNK_BYTES = 64 * 1024
_SNAPSHOT_ATTEMPTS = 3
_REASON_LIMIT = 200
_UNSAFE_REASON = "preview is unavailable"
_PATH_TOKEN = re.compile(r"\S*/[^\s,;)]*")
_WHITESPACE = re.compile(r"\s+")


class PreviewUnavailableError(Exception):
    """An internal refusal with a short, path-free reason."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True)
class _Snapshot:
    """One descriptor read: the bytes asked for, their size and digest."""

    data: Optional[bytes]
    size: int
    sha256: str


def _is_beneath(path: Path, root: Path) -> bool:
    """Containment check that also works on Python versions without is_relative_to."""
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _task_version(task: dict) -> Optional[int]:
    """Return the row's optimistic version as reported with this snapshot."""
    version = task.get("version")
    if isinstance(version, bool) or not isinstance(version, int):
        return None
    return version


class PreviewService:
    """Read one task's published preview as JSON-safe base64."""

    def __init__(
        self,
        registry: Any,
        preview_root: Optional[object] = None,
        *,
        max_bytes: int = DEFAULT_PREVIEW_MAX_BYTES,
    ) -> None:
        if not callable(getattr(registry, "get_task", None)):
            raise TypeError("preview service requires a registry with get_task()")
        if (
            isinstance(max_bytes, bool)
            or not isinstance(max_bytes, int)
            or max_bytes < 1
        ):
            raise ValueError("max_bytes must be a positive integer")
        if preview_root is None:
            preview_root = os.environ.get(PREVIEW_DIR_ENV) or _DEFAULT_PREVIEW_DIR
        if not isinstance(preview_root, (str, os.PathLike)):
            raise ValueError("preview_root must be a filesystem path")
        expanded = Path(preview_root).expanduser()
        self._configured_root = Path(os.path.abspath(str(expanded)))
        self._registry = registry
        self._max_bytes = max_bytes

    # -- configuration ---------------------------------------------------

    @property
    def max_bytes(self) -> int:
        """The largest preview this service will read."""
        return self._max_bytes

    @property
    def preview_root(self) -> Path:
        """The resolved preview root, for diagnostics and containment."""
        return self._resolved_root()

    @property
    def registry(self) -> Any:
        """The registry this service reads tasks from."""
        return self._registry

    def _resolved_root(self) -> Path:
        """Resolve the configured root once per call, following only config links."""
        return Path(self._configured_root).resolve()

    # -- public results --------------------------------------------------

    def unavailable(self, reason: object, *, task_id: Optional[str] = None) -> dict:
        """Build a refusal result that names no path and carries no bytes."""
        return {
            "available": False,
            "task_id": task_id if isinstance(task_id, str) and task_id else None,
            "rev": None,
            "version": None,
            "mime": None,
            "bytes": None,
            "sha256": None,
            "data_base64": None,
            "reason": self._safe_reason(reason),
        }

    def get_preview(
        self, task_id: object, *, conversation_key: Optional[str] = None
    ) -> dict:
        """Return one task's preview as base64, or a clean unavailable result."""
        return self._load(
            task_id, conversation_key=conversation_key, include_data=True
        )

    def preview_descriptor(
        self, task_id: object, *, conversation_key: Optional[str] = None
    ) -> dict:
        """Return the same revision, size and digest, but no image bytes."""
        return self._load(
            task_id, conversation_key=conversation_key, include_data=False
        )

    # -- read ------------------------------------------------------------

    def _load(
        self,
        task_id: object,
        *,
        conversation_key: Optional[str],
        include_data: bool,
    ) -> dict:
        """Read one consistent snapshot, retrying when a publish races us."""
        if not isinstance(task_id, str) or not task_id:
            return self.unavailable("task not found")
        for _ in range(_SNAPSHOT_ATTEMPTS):
            task = self._registry.get_task(task_id)
            if task is None:
                return self.unavailable("task not found")
            refusal = self._scope_refusal(task, conversation_key)
            if refusal is not None:
                return self.unavailable(refusal.reason, task_id=task_id)
            try:
                path, revision, mime = self._validated_preview(task)
                snapshot = self._read_snapshot(path, want_data=include_data)
            except PreviewUnavailableError as refusal:
                return self.unavailable(refusal.reason, task_id=task_id)

            latest = self._registry.get_task(task_id)
            if latest is None:
                return self.unavailable("task not found")
            if (
                latest.get("preview_path"),
                latest.get("preview_rev"),
            ) != (task.get("preview_path"), task.get("preview_rev")):
                # A publish landed mid-read: the bytes we hold belong to the
                # superseded revision, so they must not be labelled with the
                # newer one. Start again from the new record.
                continue
            return self._available(
                task_id,
                revision,
                mime,
                snapshot,
                version=_task_version(task),
                include_data=include_data,
            )
        return self.unavailable(
            "preview changed while it was being read; try again", task_id=task_id
        )

    def _available(
        self,
        task_id: str,
        revision: int,
        mime: str,
        snapshot: _Snapshot,
        *,
        version: Optional[int],
        include_data: bool,
    ) -> dict:
        """Build the one success shape shared by the body and the descriptor."""
        payload = None
        if include_data and snapshot.data is not None:
            payload = base64.b64encode(snapshot.data).decode("ascii")
        return {
            "available": True,
            "task_id": task_id,
            "rev": revision,
            "version": version,
            "mime": mime,
            "bytes": snapshot.size,
            "sha256": snapshot.sha256,
            "data_base64": payload,
            "reason": None,
        }

    # -- scope -----------------------------------------------------------

    @staticmethod
    def _scope_refusal(
        task: dict, conversation_key: Optional[str]
    ) -> Optional[PreviewUnavailableError]:
        """Refuse a preview the caller's conversation does not own."""
        if conversation_key is None:
            return None
        if not isinstance(conversation_key, str) or not conversation_key:
            return PreviewUnavailableError(
                "preview is unavailable without a conversation"
            )
        if task.get("conversation_key") != conversation_key:
            return PreviewUnavailableError(
                "preview is not available in this conversation"
            )
        return None

    # -- path validation -------------------------------------------------

    def _tail(self, raw_path: Path) -> Path:
        """Return the path relative to a preview root, refusing escapes."""
        candidates = []
        for base in (self._resolved_root(), self._configured_root):
            try:
                relative = raw_path.relative_to(base)
            except ValueError:
                continue
            if ".." in relative.parts:
                raise PreviewUnavailableError(
                    "preview path is outside the preview root"
                )
            candidates.append(relative)
        if not candidates:
            raise PreviewUnavailableError(
                "preview path is outside the preview root"
            )
        return candidates[0]

    def _validated_preview(self, task: dict) -> tuple:
        """Validate the persisted path and return (path, revision, mime)."""
        raw = task.get("preview_path")
        if not isinstance(raw, str) or not raw.strip():
            raise PreviewUnavailableError("no preview has been published")
        revision = task.get("preview_rev")
        if (
            isinstance(revision, bool)
            or not isinstance(revision, int)
            or revision < 1
        ):
            raise PreviewUnavailableError("preview revision is missing")
        if "\x00" in raw:
            raise PreviewUnavailableError("preview path is invalid")
        path = Path(raw)
        if not path.is_absolute():
            raise PreviewUnavailableError("preview path is invalid")

        root = self._resolved_root()
        try:
            if not root.is_dir():
                raise PreviewUnavailableError(
                    "preview storage is not available"
                )
        except OSError:
            raise PreviewUnavailableError(
                "preview storage is not available"
            ) from None

        relative = self._tail(path)
        if not relative.parts:
            raise PreviewUnavailableError("preview path is invalid")

        current = root
        leaf = None
        for index, part in enumerate(relative.parts):
            current = current / part
            try:
                info = os.lstat(str(current))
            except FileNotFoundError:
                raise PreviewUnavailableError(
                    "preview file is missing"
                ) from None
            except OSError:
                raise PreviewUnavailableError(
                    "preview file cannot be inspected"
                ) from None
            if stat.S_ISLNK(info.st_mode):
                raise PreviewUnavailableError(
                    "preview path uses a symbolic link and was refused"
                )
            if index < len(relative.parts) - 1:
                if not stat.S_ISDIR(info.st_mode):
                    raise PreviewUnavailableError("preview file is missing")
            else:
                leaf = info
        if leaf is None or not stat.S_ISREG(leaf.st_mode):
            raise PreviewUnavailableError("preview file is not a regular file")

        suffix = current.suffix
        mime = PREVIEW_MIME_TYPES.get(suffix)
        if mime is None:
            raise PreviewUnavailableError(
                "preview file type is not supported"
            )
        if current.stem != "rev-%d" % revision:
            # The publisher names each revision's file after that revision, so
            # a record whose path and revision disagree would hand the caller
            # one revision's bytes under another's number.
            raise PreviewUnavailableError(
                "preview file does not match the recorded revision"
            )

        try:
            resolved = current.resolve(strict=True)
        except (OSError, RuntimeError):
            raise PreviewUnavailableError("preview file is missing") from None
        if resolved != current or not _is_beneath(resolved, root):
            raise PreviewUnavailableError(
                "preview path is outside the preview root"
            )
        return resolved, revision, mime

    # -- bytes -----------------------------------------------------------

    def _read_snapshot(self, path: Path, *, want_data: bool = True) -> _Snapshot:
        """Read a preview once from an open descriptor, sized and hashed.

        The descriptor pins the inode: a concurrent ``os.replace`` at the path
        cannot change what this call reads, so the size and digest always
        describe exactly the bytes returned (or withheld) here.
        """
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        flags |= getattr(os, "O_CLOEXEC", 0)
        try:
            descriptor = os.open(str(path), flags)
        except OSError:
            raise PreviewUnavailableError("preview file cannot be opened") from None
        try:
            info = os.fstat(descriptor)
            if not stat.S_ISREG(info.st_mode):
                raise PreviewUnavailableError(
                    "preview file is not a regular file"
                )
            if info.st_size > self._max_bytes:
                raise PreviewUnavailableError(
                    "preview file exceeds the size limit"
                )
            chunks = []
            digest = hashlib.sha256()
            total = 0
            while True:
                chunk = os.read(descriptor, _CHUNK_BYTES)
                if not chunk:
                    break
                total += len(chunk)
                if total > self._max_bytes:
                    raise PreviewUnavailableError(
                        "preview file exceeds the size limit"
                    )
                digest.update(chunk)
                if want_data:
                    chunks.append(chunk)
            if total == 0:
                raise PreviewUnavailableError("preview file is empty")
            data = b"".join(chunks) if want_data else None
            return _Snapshot(data=data, size=total, sha256=digest.hexdigest())
        except OSError:
            raise PreviewUnavailableError("preview file cannot be read") from None
        finally:
            try:
                os.close(descriptor)
            except OSError:
                pass

    # -- reasons ---------------------------------------------------------

    def _safe_reason(self, reason: object) -> str:
        """Return a short reason that never includes a filesystem path."""
        text = reason if isinstance(reason, str) and reason.strip() else _UNSAFE_REASON
        text = text.replace("\x00", " ").replace("\r", " ").replace("\n", " ")
        for base in (str(self._configured_root), str(self._resolved_root())):
            if base:
                text = text.replace(base, "preview storage")
        text = _PATH_TOKEN.sub("preview file", text)
        text = _WHITESPACE.sub(" ", text).strip()
        if not text:
            return _UNSAFE_REASON
        if len(text) > _REASON_LIMIT:
            text = text[: _REASON_LIMIT - 1].rstrip() + "…"
        return text


__all__ = [
    "DEFAULT_PREVIEW_MAX_BYTES",
    "PREVIEW_DIR_ENV",
    "PREVIEW_MIME_TYPES",
    "PreviewService",
    "PreviewUnavailableError",
]
