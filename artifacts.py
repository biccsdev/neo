"""Artifact verification for NEO.

A task's result is only real when NEO can prove it is a plain file inside the
task's own workdir. This module answers that question once, for the worker
bridge, the API and the presentation policy:

* the workdir must exist and be a directory;
* the name must be relative, with no traversal, scheme, or null byte;
* every path component must be a real component, never a symlink;
* the final component must be a regular file, not a directory, pipe or device;
* the extension must be on an explicit allowlist of types NEO can show.

Refusals raise ``ArtifactError`` subclasses with short messages that name the
artifact by its relative name only. An absolute path is never echoed back, so a
refusal is safe to log or speak.

The module performs no opening, no copying and no publication. ``completion_evidence``
is the one bridge to the state machine: it only returns
``CompletionEvidence(executor_confirmed=True)`` after every expected artifact
verified. An empty expectation is allowed only when the caller says so
explicitly, because "no artifacts expected" and "artifacts went missing" must
never be confused.
"""

from __future__ import annotations

import os
import re
import stat
from collections.abc import Iterable as IterableABC
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

from state_machine import CompletionEvidence

#: Extensions NEO may present, lower-case and dot-prefixed. Grouped for review;
#: the set is flat because lookup is by suffix alone.
_DOCUMENT_EXTENSIONS = (
    ".md",
    ".txt",
    ".rst",
    ".pdf",
    ".doc",
    ".docx",
    ".rtf",
    ".csv",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
)
_SOURCE_EXTENSIONS = (
    ".py",
    ".js",
    ".mjs",
    ".cjs",
    ".ts",
    ".tsx",
    ".jsx",
    ".html",
    ".htm",
    ".css",
    ".scss",
    ".sh",
    ".bash",
    ".rb",
    ".go",
    ".rs",
    ".java",
    ".kt",
    ".swift",
    ".c",
    ".h",
    ".hpp",
    ".cc",
    ".cpp",
    ".sql",
    ".ipynb",
)
_IMAGE_EXTENSIONS = (
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".svg",
    ".webp",
    ".bmp",
    ".tif",
    ".tiff",
    ".heic",
    ".ico",
)
_MEDIA_EXTENSIONS = (
    ".mp3",
    ".wav",
    ".m4a",
    ".aac",
    ".flac",
    ".ogg",
    ".opus",
    ".mp4",
    ".mov",
    ".m4v",
    ".webm",
    ".avi",
    ".mkv",
)
_THREE_D_EXTENSIONS = (".blend", ".glb", ".gltf", ".fbx", ".obj", ".stl")

DEFAULT_ALLOWED_EXTENSIONS = frozenset(
    _DOCUMENT_EXTENSIONS
    + _SOURCE_EXTENSIONS
    + _IMAGE_EXTENSIONS
    + _MEDIA_EXTENSIONS
    + _THREE_D_EXTENSIONS
)
"""Types a NEO coding, browser or Blender task may legitimately produce."""

_SCHEME_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9+.\-]*://")
_DRIVE_PATTERN = re.compile(r"^[A-Za-z]:")


class ArtifactError(ValueError):
    """Base class for a refused artifact."""


class MissingWorkdirError(ArtifactError):
    """The task has no usable workdir, so nothing can be contained."""


class ArtifactNotFoundError(ArtifactError):
    """The named artifact does not exist beneath the workdir."""


class ArtifactNotRegularFileError(ArtifactError):
    """The name exists but is a directory, pipe, socket, device or other special file."""


class ArtifactEscapesWorkdirError(ArtifactError):
    """The name is absolute, traverses, uses a symlink, or leaves the workdir."""


class UnsupportedArtifactTypeError(ArtifactError):
    """The extension is not on the allowlist NEO is willing to present."""


class EmptyExpectedArtifactsError(ArtifactError):
    """Completion was attempted with no expected artifacts and no explicit opt-in."""


@dataclass(frozen=True)
class Artifact:
    """One verified artifact: a resolved absolute path plus its workdir name."""

    name: str
    path: Path


def _kind(mode: int) -> str:
    """Describe a non-regular file in words a message can use."""
    if stat.S_ISDIR(mode):
        return "directory"
    if stat.S_ISFIFO(mode):
        return "named pipe"
    if stat.S_ISSOCK(mode):
        return "socket"
    if stat.S_ISCHR(mode):
        return "character device"
    if stat.S_ISBLK(mode):
        return "block device"
    return "special file"


def _leaf(name: str) -> str:
    """Return only the final segment, so a message never echoes an absolute path."""
    try:
        return Path(name).name or "artifact"
    except (OSError, ValueError):
        return "artifact"


def _normalise_extensions(allowed_extensions: object) -> "frozenset[str]":
    """Return a lower-case, dot-prefixed allowlist or refuse a malformed one."""
    if allowed_extensions is None:
        return DEFAULT_ALLOWED_EXTENSIONS
    if isinstance(allowed_extensions, (str, bytes)):
        raise TypeError(
            "allowed_extensions must be a collection of extensions, not a single string"
        )
    if not isinstance(allowed_extensions, IterableABC):
        raise TypeError("allowed_extensions must be an iterable of extensions")
    normalised = set()
    for entry in allowed_extensions:
        if not isinstance(entry, str) or not entry.strip():
            raise ValueError("artifact extensions must be non-empty strings")
        text = entry.strip().lower()
        if not text.startswith("."):
            text = "." + text
        if text == "." or "/" in text or "\\" in text or os.sep in text:
            raise ValueError("artifact extension %r is not a file suffix" % entry)
        normalised.add(text)
    return frozenset(normalised)


def _resolve_workdir(workdir: object) -> Path:
    """Return the real workdir, or refuse when a task has no usable directory."""
    if workdir is None or isinstance(workdir, bool):
        raise MissingWorkdirError(
            "task workdir is missing; an artifact cannot be verified without one"
        )
    if not isinstance(workdir, (str, os.PathLike)):
        raise MissingWorkdirError("task workdir must be a filesystem path")
    if isinstance(workdir, str) and not workdir.strip():
        raise MissingWorkdirError(
            "task workdir is missing; an artifact cannot be verified without one"
        )
    try:
        resolved = Path(workdir).resolve(strict=True)
    except (OSError, RuntimeError, ValueError):
        raise MissingWorkdirError(
            "task workdir does not exist or cannot be resolved"
        ) from None
    try:
        is_directory = resolved.is_dir()
    except OSError:
        raise MissingWorkdirError("task workdir cannot be inspected") from None
    if not is_directory:
        raise MissingWorkdirError("task workdir is not a directory")
    return resolved


def _name_parts(name: object) -> Tuple[str, ...]:
    """Split a relative artifact name, refusing absolutes, schemes and traversal."""
    if not isinstance(name, str) or not name.strip():
        raise ArtifactError("expected artifact name must be a non-empty string")
    if "\x00" in name:
        raise ArtifactEscapesWorkdirError(
            "expected artifact name contains a null byte and was refused"
        )
    if _SCHEME_PATTERN.match(name.strip()):
        raise ArtifactEscapesWorkdirError(
            "expected artifact is a URL, not a file beneath the task workdir"
        )
    if Path(name).is_absolute() or os.path.isabs(name) or _DRIVE_PATTERN.match(name):
        raise ArtifactEscapesWorkdirError(
            "expected artifact %r must be relative to the task workdir"
            % _leaf(name)
        )
    parts = tuple(part for part in Path(name).parts if part not in ("", "."))
    if not parts:
        raise ArtifactEscapesWorkdirError(
            "expected artifact %r names the workdir itself, not a file" % name
        )
    if ".." in parts:
        raise ArtifactEscapesWorkdirError(
            "expected artifact %r traverses outside the task workdir" % name
        )
    return parts


def _walk(root: Path, parts: Sequence[str], name: str) -> Path:
    """Walk every component without following symlinks and return the final path."""
    current = root
    last_index = len(parts) - 1
    for index, part in enumerate(parts):
        current = current / part
        try:
            info = current.lstat()
        except FileNotFoundError:
            raise ArtifactNotFoundError(
                "expected artifact %r does not exist beneath the task workdir" % name
            ) from None
        except OSError:
            raise ArtifactNotFoundError(
                "expected artifact %r cannot be inspected" % name
            ) from None
        if stat.S_ISLNK(info.st_mode):
            raise ArtifactEscapesWorkdirError(
                "expected artifact %r uses a symbolic link and was refused" % name
            )
        if index != last_index:
            if not stat.S_ISDIR(info.st_mode):
                raise ArtifactNotFoundError(
                    "expected artifact %r has a path component that is not a directory"
                    % name
                )
        elif not stat.S_ISREG(info.st_mode):
            raise ArtifactNotRegularFileError(
                "expected artifact %r is not a regular file (%s)"
                % (name, _kind(info.st_mode))
            )
    return current


def _check_extension(name: str, allowed_extensions: "frozenset[str]") -> None:
    """Refuse a file whose suffix NEO has no policy for opening."""
    suffix = Path(name).suffix.lower()
    if not suffix:
        raise UnsupportedArtifactTypeError(
            "expected artifact %r has no file extension; NEO refuses to open unknown "
            "file types" % name
        )
    if suffix not in allowed_extensions:
        raise UnsupportedArtifactTypeError(
            "expected artifact %r has extension %r, which is not on NEO's allowlist"
            % (name, suffix)
        )


def verify_artifact(
    workdir: object,
    name: str,
    *,
    allowed_extensions: Optional[Iterable[str]] = None,
) -> Artifact:
    """Return the verified artifact or raise an ``ArtifactError`` subclass.

    The returned path is resolved and strictly beneath the resolved workdir; the
    returned name is the canonical posix-style name relative to that workdir.
    """
    root = _resolve_workdir(workdir)
    parts = _name_parts(name)
    extensions = _normalise_extensions(allowed_extensions)
    path = _walk(root, parts, name)
    _check_extension(parts[-1], extensions)
    try:
        resolved = path.resolve(strict=True)
    except (OSError, RuntimeError):
        raise ArtifactNotFoundError(
            "expected artifact %r cannot be resolved" % name
        ) from None
    if resolved == root or not _is_beneath(resolved, root):
        raise ArtifactEscapesWorkdirError(
            "expected artifact %r escapes the task workdir" % name
        )
    return Artifact(name=resolved.relative_to(root).as_posix(), path=resolved)


def _is_beneath(path: Path, root: Path) -> bool:
    """Containment check that also works on Python versions without is_relative_to."""
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def verify_artifacts(
    workdir: object,
    names: Iterable[str],
    *,
    allowed_extensions: Optional[Iterable[str]] = None,
) -> Tuple[Artifact, ...]:
    """Verify every name in order; the first refusal aborts the whole batch."""
    extensions = _normalise_extensions(allowed_extensions)
    return tuple(
        verify_artifact(workdir, name, allowed_extensions=extensions) for name in names
    )


def _expected_names(expected_artifacts: object) -> Tuple[str, ...]:
    """Coerce an expectation into a tuple of names, treating a bare string as one."""
    if expected_artifacts is None:
        return ()
    if isinstance(expected_artifacts, str):
        return (expected_artifacts,)
    try:
        return tuple(expected_artifacts)  # type: ignore[arg-type]
    except TypeError:
        raise TypeError("expected artifacts must be an iterable of relative names") from None


def completion_evidence(
    workdir: object,
    expected_artifacts: Iterable[str],
    *,
    allow_empty: bool = False,
    allowed_extensions: Optional[Iterable[str]] = None,
) -> CompletionEvidence:
    """Build completion evidence from verified expected artifacts.

    ``executor_confirmed`` is always ``True`` here because the caller must only
    reach this point holding the executor's own terminal result. A process exit
    code is not terminal evidence and must never be routed through this helper.

    An empty expectation is refused unless ``allow_empty=True``; NEO must opt in
    to "nothing to present" rather than infer it from a missing list.
    """
    names = _expected_names(expected_artifacts)
    if not names:
        if not allow_empty:
            raise EmptyExpectedArtifactsError(
                "completion evidence requires at least one expected artifact; "
                "an empty expectation is allowed only when the caller requests it "
                "explicitly"
            )
        return CompletionEvidence(executor_confirmed=True)
    verified = verify_artifacts(
        workdir, names, allowed_extensions=allowed_extensions
    )
    canonical = tuple(artifact.name for artifact in verified)
    return CompletionEvidence(
        executor_confirmed=True,
        expected_artifacts=canonical,
        verified_artifacts=canonical,
    )


__all__ = [
    "DEFAULT_ALLOWED_EXTENSIONS",
    "Artifact",
    "ArtifactError",
    "ArtifactEscapesWorkdirError",
    "ArtifactNotFoundError",
    "ArtifactNotRegularFileError",
    "EmptyExpectedArtifactsError",
    "MissingWorkdirError",
    "UnsupportedArtifactTypeError",
    "completion_evidence",
    "verify_artifact",
    "verify_artifacts",
]
