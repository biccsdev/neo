"""NEO dashboard backend API.

The dashboard is a thin client over NEO's durable state. This module is the
only HTTP surface NEO exposes, and it keeps that surface narrow:

* every response is JSON; previews and artifacts travel as base64 text, so a
  route can never become a file server that ignores NEO's own containment
  rules;
* task ids are validated before they reach the registry, so a malformed or
  traversing id is *not found*, never a probe;
* the task body is an allowlist, so internal columns (a worker token digest,
  for instance) cannot ride along by accident;
* blocking SQLite and file work runs in the threadpool, never on the event
  loop;
* cancel, retry and approval are idempotent: repeating a completed decision
  returns the same row rather than mutating it again.

Service wiring is injected: ``build_router(services_factory)`` binds the routes
to whatever object the factory returns, and the factory is only consulted by a
route that needs durable state. ``GET /health`` never resolves it, so the
dashboard can probe a plugin whose services are not ready yet.

The module-level ``router`` is what Hermes mounts under ``/api/plugins/neo``;
its services are built lazily on first use and then reused.
"""

from __future__ import annotations

import base64
import hashlib
import json
import mimetypes
import os
import re
import stat
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import Any, Callable, List, Literal, Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

# -- import bootstrap ---------------------------------------------------------
#
# Hermes loads this file with ``spec_from_file_location`` and does not extend
# ``sys.path``, while the modules it depends on live one level up (the plugin
# root). Adding the root here keeps ``plugin_api`` importable both as a mounted
# backend and as a plain module in tests.

_DASHBOARD_DIR = Path(__file__).resolve().parent
_PLUGIN_ROOT = _DASHBOARD_DIR.parent
if str(_PLUGIN_ROOT) not in sys.path:
    sys.path.insert(0, str(_PLUGIN_ROOT))

from artifacts import (  # noqa: E402  (path bootstrap above)
    ArtifactError,
    ArtifactEscapesWorkdirError,
    ArtifactNotFoundError,
    ArtifactNotRegularFileError,
    MissingWorkdirError,
    UnsupportedArtifactTypeError,
    verify_artifact,
)
from capabilities import (  # noqa: E402
    CapabilityDiscovery,
    ConsentRequiredError,
    InstallRefusedError,
    JsonConsentStore,
    validate_capability,
)
from events import TaskService  # noqa: E402
from presentation import (  # noqa: E402
    MacOSOpener,
    PresentationResult,
    PresentationService,
    TaskPresenter,
)
from previews import PreviewService  # noqa: E402
from registry import Registry, TaskNotFoundError  # noqa: E402
from state_machine import TERMINAL_STATES, TaskState  # noqa: E402

PLUGIN_VERSION = "0.1.0"
"""Version reported by ``GET /health``; mirrors the package and the manifest."""

NEO_KNOWN_CAPABILITIES = ("blender", "browser")
"""Capabilities NEO routes to (plan §4.7). Installed discovery asks about these
names only, because a capability NEO cannot route is not NEO's business."""

_TASK_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
"""Ids NEO itself generates. Anything else is not found before any lookup."""

_KIND_RE = r"^[A-Za-z][A-Za-z0-9_-]{0,31}$"
_QUERY_RE = re.compile(r"^[^\x00-\x1f\x7f]*$")

_ARTIFACT_MIME = MappingProxyType(
    {
        ".md": "text/markdown",
        ".markdown": "text/markdown",
        ".rst": "text/x-rst",
        ".txt": "text/plain",
        ".json": "application/json",
        ".yaml": "application/yaml",
        ".yml": "application/yaml",
        ".toml": "application/toml",
        ".csv": "text/csv",
        ".pdf": "application/pdf",
        ".doc": "application/msword",
        ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ".rtf": "application/rtf",
        ".html": "text/html",
        ".htm": "text/html",
        ".css": "text/css",
        ".js": "text/javascript",
        ".mjs": "text/javascript",
        ".cjs": "text/javascript",
        ".jsx": "text/javascript",
        ".ts": "text/plain",
        ".tsx": "text/plain",
        ".py": "text/x-python",
        ".sh": "text/x-shellscript",
        ".bash": "text/x-shellscript",
        ".sql": "application/sql",
        ".svg": "image/svg+xml",
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
        ".tif": "image/tiff",
        ".tiff": "image/tiff",
        ".ico": "image/x-icon",
        ".mp3": "audio/mpeg",
        ".wav": "audio/wav",
        ".m4a": "audio/mp4",
        ".mp4": "video/mp4",
        ".mov": "video/quicktime",
        ".webm": "video/webm",
        ".glb": "model/gltf-binary",
        ".gltf": "model/gltf+json",
        ".obj": "text/plain",
        ".stl": "model/stl",
        ".blend": "application/octet-stream",
        ".ipynb": "application/x-ipynb+json",
    }
)
"""Mime labels for artefacts NEO may present; anything else is octet-stream."""

_TASK_FIELDS = (
    "id",
    "parent_id",
    "kind",
    "intent",
    "state",
    "capability_id",
    "executor",
    "workdir",
    "resource_key",
    "preview_rev",
    "version",
    "created_at",
    "updated_at",
    "started_at",
    "finished_at",
    "result_summary",
    "error_code",
    "error_detail",
)
"""Task columns the dashboard may see, in dashboard order."""

_TERMINAL = frozenset(state.value for state in TERMINAL_STATES)

_DEFAULT_ARTIFACT_MAX_BYTES = 1024 * 1024
_DEFAULT_LOG_MAX_BYTES = 256 * 1024

_HOME_ENV = "NEO_HOME"
_DB_ENV = "NEO_DB"
_LOG_ENV = "NEO_LOG_DIR"
_NOT_FOUND = "task not found"


# -- request bodies -----------------------------------------------------------


class ApprovalBody(BaseModel):
    """One explicit approval decision; anything else is a validation error."""

    decision: Literal["approve", "deny"]
    note: Optional[str] = Field(None, max_length=2000)


class CapabilityBody(BaseModel):
    """A catalog candidate exactly as the dashboard displayed it."""

    name: str = Field(..., min_length=1, max_length=64)
    repo: str = Field(..., min_length=1, max_length=2048)
    sha: str = Field(..., min_length=1, max_length=64)
    tier: str = Field(..., min_length=1, max_length=32)
    description: str = Field("", max_length=2000)
    version: str = Field("", max_length=64)
    subdir: str = Field("", max_length=256)
    dependencies: List[str] = Field(default_factory=list)


class EnableBody(BaseModel):
    """A capability NEO should hand to Hermes to enable."""

    name: str = Field(..., min_length=1, max_length=64, pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class PresentBody(BaseModel):
    """Optional conversation binding for one presentation request."""

    conversation_key: Optional[str] = Field(None, min_length=1, max_length=128)


# -- small helpers ------------------------------------------------------------


def _services(factory: Callable[[], Any]) -> Any:
    """Resolve the injected services, or refuse politely."""
    try:
        services = factory()
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=503, detail="neo services are unavailable") from None
    if services is None:
        raise HTTPException(status_code=503, detail="neo services are unavailable")
    return services


def _registry_of(services: Any) -> Any:
    registry = getattr(services, "registry", None)
    if registry is None or not callable(getattr(registry, "get_task", None)):
        raise HTTPException(status_code=503, detail="neo registry is unavailable")
    return registry


def _check_task_id(task_id: str) -> str:
    """Refuse anything NEO could not have issued, before any lookup."""
    if not isinstance(task_id, str) or not _TASK_ID_RE.match(task_id):
        raise HTTPException(status_code=404, detail=_NOT_FOUND)
    return task_id


async def _load_task(services: Any, task_id: str) -> dict:
    """Read one task off the event loop, or 404 without probing storage."""
    _check_task_id(task_id)
    registry = _registry_of(services)
    task = await run_in_threadpool(registry.get_task, task_id)
    if not isinstance(task, dict) or not task.get("id"):
        raise HTTPException(status_code=404, detail=_NOT_FOUND)
    return task


def _decode_list(raw: Any) -> list:
    """Decode a JSON string list column, treating malformed text as empty."""
    if isinstance(raw, list):
        value = raw
    elif isinstance(raw, (str, bytes)):
        try:
            value = json.loads(raw)
        except (TypeError, ValueError):
            return []
    else:
        return []
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str)]


def _decode_dict(raw: Any) -> Optional[dict]:
    """Decode a JSON object column, refusing anything that is not an object."""
    if isinstance(raw, dict):
        return dict(raw)
    if not isinstance(raw, (str, bytes)):
        return None
    try:
        value = json.loads(raw)
    except (TypeError, ValueError):
        return None
    return dict(value) if isinstance(value, dict) else None


def _public_task(task: dict) -> dict:
    """Project one task row onto the dashboard allowlist.

    Only named columns leave; the row may hold token digests, worker paths and
    other internals, and none of them are reachable from this projection.
    """
    body = {field: task.get(field) for field in _TASK_FIELDS}
    body["expected"] = _decode_list(task.get("expected_json"))
    body["artifacts"] = _decode_list(task.get("artifacts_json"))
    body["approval"] = _decode_dict(task.get("approval_json"))
    body["preview"] = {
        "published": bool(task.get("preview_path")),
        "rev": task.get("preview_rev"),
    }
    body["has_log"] = bool(task.get("log_path"))
    return body


def _now_iso() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )


def _mime_for(name: str) -> str:
    suffix = Path(name).suffix.lower()
    if suffix in _ARTIFACT_MIME:
        return _ARTIFACT_MIME[suffix]
    guessed = mimetypes.guess_type(name)[0]
    return guessed or "application/octet-stream"


def _artifact_name_or_400(raw: str) -> str:
    """Refuse an absolute, Windows, NUL-bearing or scheme-bearing name."""
    if not isinstance(raw, str) or not raw:
        raise HTTPException(status_code=400, detail="artifact name is malformed")
    if "\x00" in raw or "\\" in raw:
        raise HTTPException(status_code=400, detail="artifact name is malformed")
    if raw.startswith("/"):
        raise HTTPException(status_code=400, detail="artifact name must be relative")
    if re.match(r"^[A-Za-z]:", raw):
        raise HTTPException(status_code=400, detail="artifact name must be relative")
    if "://" in raw:
        raise HTTPException(status_code=400, detail="artifact name must be relative")
    return raw


def _artifact_beneath_404(detail: str = "no workdir is recorded for this task") -> HTTPException:
    return HTTPException(status_code=404, detail=detail)


def _read_artifact(services: Any, task: dict, name: str) -> dict:
    """Verify and read one artifact as base64, entirely inside the workdir."""
    workdir = task.get("workdir")
    if not isinstance(workdir, str) or not workdir.strip():
        raise _artifact_beneath_404()
    cap = getattr(services, "artifact_max_bytes", _DEFAULT_ARTIFACT_MAX_BYTES)
    if isinstance(cap, bool) or not isinstance(cap, int) or cap < 1:
        cap = _DEFAULT_ARTIFACT_MAX_BYTES
    try:
        artifact = verify_artifact(workdir, name)
    except ArtifactEscapesWorkdirError:
        raise HTTPException(status_code=400, detail="artifact escapes the task workdir") from None
    except ArtifactNotFoundError:
        raise HTTPException(status_code=404, detail="artifact not found") from None
    except MissingWorkdirError:
        raise _artifact_beneath_404() from None
    except UnsupportedArtifactTypeError:
        raise HTTPException(status_code=415, detail="artifact type is not supported") from None
    except ArtifactNotRegularFileError:
        raise HTTPException(status_code=400, detail="artifact is not a regular file") from None
    except ArtifactError:
        raise HTTPException(status_code=400, detail="artifact was refused") from None
    try:
        size = artifact.path.stat().st_size
    except OSError:
        raise HTTPException(status_code=404, detail="artifact not found") from None
    if size > cap:
        raise HTTPException(status_code=413, detail="artifact is too large")
    try:
        with artifact.path.open("rb") as handle:
            data = handle.read(cap + 1)
    except OSError:
        raise HTTPException(status_code=404, detail="artifact not found") from None
    if len(data) > cap:
        raise HTTPException(status_code=413, detail="artifact is too large")
    return {
        "name": artifact.name,
        "mime": _mime_for(artifact.name),
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "data_base64": base64.b64encode(data).decode("ascii"),
    }


def _log_root(services: Any) -> Path:
    raw = getattr(services, "log_root", None)
    if isinstance(raw, (str, os.PathLike)):
        return Path(os.path.abspath(str(raw))).resolve()
    raise HTTPException(status_code=404, detail="no log directory is configured")


def _read_log_tail(services: Any, task: dict, tail: int) -> dict:
    """Return the last ``tail`` lines of this task's log, if it is inside the root."""
    raw_path = task.get("log_path")
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise HTTPException(status_code=404, detail="task has no log")
    root = _log_root(services)
    path = Path(raw_path)
    if not path.is_absolute():
        raise HTTPException(status_code=404, detail="task has no log")
    try:
        info = path.lstat()
    except OSError:
        raise HTTPException(status_code=404, detail="task has no log") from None
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise HTTPException(status_code=404, detail="task has no log")
    try:
        resolved = path.resolve(strict=True)
    except OSError:
        raise HTTPException(status_code=404, detail="task has no log") from None
    try:
        resolved.relative_to(root)
    except ValueError:
        raise HTTPException(status_code=404, detail="task has no log") from None
    cap = getattr(services, "log_max_bytes", _DEFAULT_LOG_MAX_BYTES)
    if isinstance(cap, bool) or not isinstance(cap, int) or cap < 1:
        cap = _DEFAULT_LOG_MAX_BYTES
    size = info.st_size
    start = max(0, size - cap)
    try:
        with resolved.open("rb") as handle:
            if start:
                handle.seek(start)
            chunk = handle.read(cap)
    except OSError:
        raise HTTPException(status_code=404, detail="task has no log") from None
    lines = chunk.decode("utf-8", "replace").splitlines()
    drained = start > 0
    if drained and lines:
        # The first line began before the window, so it is a fragment.
        lines = lines[1:]
    truncated = drained or len(lines) > tail
    return {"lines": lines[-tail:] if tail else lines, "tail": tail, "truncated": truncated}


def _find_retry_child(registry: Any, task_id: str) -> Optional[dict]:
    """Return the child a previous retry already created, newest last."""
    list_tasks = getattr(registry, "list_tasks", None)
    if not callable(list_tasks):
        return None
    children = [
        row
        for row in list_tasks()
        if isinstance(row, dict) and row.get("parent_id") == task_id
    ]
    if not children:
        return None
    children.sort(key=lambda row: (row.get("created_at") or 0, row.get("id") or ""))
    return children[0]


def _capability_or_422(body: CapabilityBody) -> Any:
    """Validate a dashboard candidate with the same rules the installer uses."""
    candidate = {
        "name": body.name,
        "repo": body.repo,
        "sha": body.sha,
        "tier": body.tier,
        "description": body.description,
        "version": body.version,
        "subdir": body.subdir,
        "dependencies": list(body.dependencies),
    }
    try:
        return validate_capability(candidate)
    except InstallRefusedError as refusal:
        raise HTTPException(status_code=422, detail=str(refusal)) from None


def _capability_service(services: Any, method: str) -> Any:
    discovery = getattr(services, "capabilities", None)
    if discovery is None or not callable(getattr(discovery, method, None)):
        raise HTTPException(status_code=409, detail="capability service is unavailable")
    return discovery


# -- route factory ------------------------------------------------------------


def build_router(services_factory: Callable[[], Any]) -> APIRouter:
    """Return the NEO router bound to ``services_factory``.

    The factory is called by the routes that need durable state, never by
    ``/health``, and never at import time.
    """
    router = APIRouter()

    @router.get("/health")
    def health() -> dict:
        """Liveness only: no services, no paths, no filesystem."""
        return {"status": "ok", "plugin": "neo", "version": PLUGIN_VERSION}

    # -- tasks -----------------------------------------------------------

    @router.get("/tasks")
    async def list_tasks(
        state: Optional[TaskState] = None,
        kind: Optional[str] = Query(None, pattern=_KIND_RE),
        since: Optional[float] = Query(None, ge=0),
    ) -> dict:
        """List tasks, filtered by state, kind and creation time."""
        services = _services(services_factory)
        registry = _registry_of(services)
        lister = getattr(registry, "list_tasks", None)
        if not callable(lister):
            raise HTTPException(status_code=503, detail="neo registry is unavailable")
        rows = await run_in_threadpool(lister)
        selected = []
        for row in rows or ():
            if not isinstance(row, dict):
                continue
            if state is not None and row.get("state") != state.value:
                continue
            if kind is not None and row.get("kind") != kind:
                continue
            if since is not None:
                created = row.get("created_at")
                if not isinstance(created, (int, float)) or float(created) < since:
                    continue
            selected.append(_public_task(row))
        return {"tasks": selected, "count": len(selected)}

    @router.get("/tasks/{task_id}")
    async def task_detail(task_id: str) -> dict:
        """One task, projected onto the dashboard allowlist."""
        services = _services(services_factory)
        task = await _load_task(services, task_id)
        return _public_task(task)

    @router.post("/tasks/{task_id}/cancel")
    async def cancel_task(task_id: str) -> dict:
        """Cancel a live task once; a terminal task is returned unchanged."""
        services = _services(services_factory)
        task = await _load_task(services, task_id)
        if task.get("state") in _TERMINAL:
            return _public_task(task)
        supervisor = getattr(services, "supervisor", None)
        if supervisor is None or not callable(getattr(supervisor, "cancel", None)):
            raise HTTPException(status_code=409, detail="no supervisor is attached to this task")
        try:
            outcome = await run_in_threadpool(supervisor.cancel, task_id)
        except Exception:
            raise HTTPException(
                status_code=409, detail="the task could not be cancelled"
            ) from None
        row = outcome if isinstance(outcome, dict) and outcome.get("id") == task_id else None
        if row is None:
            registry = _registry_of(services)
            row = await run_in_threadpool(registry.get_task, task_id) or task
        return _public_task(row)

    @router.post("/tasks/{task_id}/retry")
    async def retry_task(task_id: str) -> dict:
        """Create one child task for a terminal task; repeats return it again."""
        services = _services(services_factory)
        task = await _load_task(services, task_id)
        if task.get("state") not in _TERMINAL:
            raise HTTPException(
                status_code=409, detail="only a terminal task can be retried"
            )
        registry = _registry_of(services)
        existing = await run_in_threadpool(_find_retry_child, registry, task_id)
        if existing is not None:
            return _public_task(existing)
        retry = getattr(services, "retry", None)
        if not callable(retry):
            raise HTTPException(status_code=409, detail="retry is not available")
        try:
            child = await run_in_threadpool(retry, task)
        except Exception:
            raise HTTPException(
                status_code=409, detail="the retry could not be created"
            ) from None
        if not isinstance(child, dict) or not child.get("id"):
            child = await run_in_threadpool(registry.get_task, task_id)
            child = _find_retry_child(registry, task_id) or child or task
        return _public_task(child)

    @router.get("/tasks/{task_id}/events")
    async def task_events(task_id: str, since: int = Query(0, ge=0)) -> dict:
        """This task's events only, after the caller's cursor."""
        services = _services(services_factory)
        task = await _load_task(services, task_id)
        service = getattr(services, "task_service", None)
        if service is not None and callable(getattr(service, "task_events", None)):
            conversation_key = task.get("conversation_key")
            rows = await run_in_threadpool(
                service.task_events, conversation_key, task_id, since
            )
        else:
            registry = _registry_of(services)
            reader = getattr(registry, "events_since", None)
            if not callable(reader):
                raise HTTPException(status_code=503, detail="neo registry is unavailable")
            rows = await run_in_threadpool(reader, since, task_id=task_id)
        events = []
        for row in rows or ():
            if not isinstance(row, dict):
                continue
            payload = row.get("payload")
            if not isinstance(payload, dict):
                payload = _decode_dict(row.get("payload_json")) or {}
            events.append(
                {
                    "id": row.get("id"),
                    "task_id": row.get("task_id"),
                    "at": row.get("at"),
                    "kind": row.get("kind"),
                    "payload": payload,
                }
            )
        return {"since": since, "events": events}

    @router.get("/tasks/{task_id}/preview")
    async def task_preview(task_id: str) -> dict:
        """This task's published preview as base64, or an honest unavailable body."""
        services = _services(services_factory)
        await _load_task(services, task_id)
        preview = getattr(services, "preview_service", None)
        if preview is None or not callable(getattr(preview, "get_preview", None)):
            raise HTTPException(status_code=409, detail="preview service is unavailable")
        return await run_in_threadpool(preview.get_preview, task_id)

    @router.get("/tasks/{task_id}/artifact")
    async def task_artifact(
        task_id: str, name: str = Query(..., min_length=1, max_length=1024)
    ) -> dict:
        """One verified artifact of this task, as base64 JSON."""
        services = _services(services_factory)
        task = await _load_task(services, task_id)
        relative = _artifact_name_or_400(name)
        return await run_in_threadpool(_read_artifact, services, task, relative)

    @router.get("/tasks/{task_id}/log")
    async def task_log(task_id: str, tail: int = Query(100, ge=1, le=1000)) -> dict:
        """The bounded tail of this task's log; never a path, never a file body."""
        services = _services(services_factory)
        task = await _load_task(services, task_id)
        return await run_in_threadpool(_read_log_tail, services, task, tail)

    @router.get("/tasks/{task_id}/capabilities")
    async def task_capabilities(task_id: str) -> dict:
        """Installed capabilities for this task's kind, without touching the network."""
        services = _services(services_factory)
        task = await _load_task(services, task_id)
        discovery = _capability_service(services, "list_installed")
        snapshot = await run_in_threadpool(discovery.list_installed, NEO_KNOWN_CAPABILITIES)
        payload = snapshot.to_dict()
        return {
            "task_id": task.get("id"),
            "kind": task.get("kind"),
            "capability_id": task.get("capability_id"),
            "installed": payload["installed"],
            "unavailable": payload["unavailable"],
            "network_used": False,
        }

    @router.post("/tasks/{task_id}/approve")
    async def approve_task(task_id: str, body: ApprovalBody) -> dict:
        """Record one approval decision; repeating the same decision is a no-op."""
        services = _services(services_factory)
        task = await _load_task(services, task_id)
        existing = _decode_dict(task.get("approval_json")) or {}
        decision = body.decision
        if task.get("state") != TaskState.NEEDS_APPROVAL.value:
            if existing.get("decision") == decision:
                return _public_task(task)
            raise HTTPException(
                status_code=409, detail="task is not waiting for approval"
            )
        service = getattr(services, "task_service", None)
        if service is None or not callable(getattr(service, "transition", None)):
            raise HTTPException(status_code=503, detail="task service is unavailable")
        target = "running" if decision == "approve" else "cancelled"
        approval = dict(existing)
        approval.update({"decision": decision, "note": body.note, "at": _now_iso()})
        try:
            updated = await run_in_threadpool(
                service.transition,
                task_id,
                target,
                expected_version=task.get("version"),
                event_extra={"decision": decision},
            )
        except Exception:
            raise HTTPException(
                status_code=409, detail="the decision was not accepted"
            ) from None
        registry = _registry_of(services)
        try:
            final = await run_in_threadpool(
                registry.update_task,
                task_id,
                expected_version=updated["version"],
                approval_json=json.dumps(approval, sort_keys=True),
            )
        except Exception:
            raise HTTPException(
                status_code=409, detail="the decision was not recorded"
            ) from None
        return _public_task(final)

    @router.post("/tasks/{task_id}/present")
    async def present_task(
        task_id: str, body: Optional[PresentBody] = None
    ) -> dict:
        """Present a completed, verified task once; repeats are clean skips.

        The presenter reads the task itself, so this route never reaches for
        another task's state. An optional conversation key binds the call to
        the conversation that owns the task.
        """
        services = _services(services_factory)
        _check_task_id(task_id)
        presenter = getattr(services, "presenter", None)
        if presenter is None or not callable(getattr(presenter, "present_task", None)):
            raise HTTPException(status_code=409, detail="presentation is not available")
        conversation_key = None if body is None else body.conversation_key
        try:
            outcome = await run_in_threadpool(
                presenter.present_task, task_id, conversation_key
            )
        except TaskNotFoundError:
            raise HTTPException(status_code=404, detail=_NOT_FOUND) from None
        except Exception:
            raise HTTPException(
                status_code=409, detail="the result could not be presented"
            ) from None
        if isinstance(outcome, PresentationResult):
            payload = outcome.as_dict()
        elif isinstance(outcome, dict):
            payload = dict(outcome)
        else:
            raise HTTPException(
                status_code=409, detail="the result could not be presented"
            )
        payload.setdefault("skipped", not bool(payload.get("presented")))
        return payload

    # -- capabilities ----------------------------------------------------

    @router.get("/capabilities")
    async def capabilities(
        q: str = Query("", max_length=120)
    ) -> dict:
        """Installed capabilities; a query is required before any catalog search."""
        if not isinstance(q, str) or not _QUERY_RE.match(q):
            raise HTTPException(status_code=422, detail="catalog query is malformed")
        services = _services(services_factory)
        discovery = _capability_service(services, "list_installed")
        installed = await run_in_threadpool(discovery.list_installed, NEO_KNOWN_CAPABILITIES)
        payload = installed.to_dict()
        body = {
            "query": q,
            "installed": payload["installed"],
            "catalog": [],
            "unavailable": payload["unavailable"],
            "network_used": False,
            "catalog_requested": False,
        }
        if q:
            searcher = getattr(discovery, "search_catalog", None)
            if not callable(searcher):
                raise HTTPException(
                    status_code=409, detail="catalog search is not available"
                )
            found = await run_in_threadpool(searcher, q)
            found_payload = found.to_dict()
            body["catalog"] = found_payload["catalog"]
            body["unavailable"] = body["unavailable"] + found_payload["unavailable"]
            body["network_used"] = bool(found_payload["network_used"])
            body["catalog_requested"] = True
        return body

    @router.post("/capabilities/consent")
    async def capability_consent(body: CapabilityBody) -> dict:
        """Record consent for exactly this name and SHA; never installs."""
        services = _services(services_factory)
        discovery = _capability_service(services, "grant_consent")
        capability = _capability_or_422(body)
        try:
            record = await run_in_threadpool(discovery.grant_consent, capability)
        except InstallRefusedError as refusal:
            raise HTTPException(status_code=422, detail=str(refusal)) from None
        payload = record.to_dict()
        payload["ok"] = True
        return payload

    @router.post("/capabilities/install")
    async def capability_install(body: CapabilityBody) -> dict:
        """Install only a candidate whose exact name and SHA were consented to."""
        services = _services(services_factory)
        discovery = _capability_service(services, "install_capability")
        capability = _capability_or_422(body)
        try:
            result = await run_in_threadpool(discovery.install_capability, capability)
        except ConsentRequiredError:
            raise HTTPException(
                status_code=409, detail="consent is required for this exact commit"
            ) from None
        except InstallRefusedError as refusal:
            raise HTTPException(status_code=422, detail=str(refusal)) from None
        payload = result.to_dict()
        payload["ok"] = bool(getattr(result, "installed", False))
        return payload

    @router.post("/capabilities/enable")
    async def capability_enable(body: EnableBody) -> dict:
        """Ask Hermes to enable an installed capability; NEO never enables alone."""
        services = _services(services_factory)
        discovery = _capability_service(services, "enable_capability")
        try:
            outcome = await run_in_threadpool(discovery.enable_capability, body.name)
        except Exception:
            raise HTTPException(
                status_code=409, detail="the capability could not be enabled"
            ) from None
        if not isinstance(outcome, dict):
            raise HTTPException(status_code=409, detail="the capability could not be enabled")
        return outcome

    return router


# -- production wiring --------------------------------------------------------

_DEFAULT_CACHE: dict = {}


def _neo_home() -> Path:
    raw = os.environ.get(_HOME_ENV)
    base = Path(raw).expanduser() if raw else Path.home() / ".hermes" / "neo"
    return Path(os.path.abspath(str(base)))


def _hermes_cli(argv: list) -> Any:
    """Run one read-only Hermes CLI command without a shell."""
    return subprocess.run(
        list(argv),
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


def _default_retry(service: TaskService) -> Callable[[dict], dict]:
    """Create a fresh child task for a terminal parent."""

    def retry(task: dict) -> dict:
        child = service.create_task(
            conversation_key=task["conversation_key"],
            kind=task["kind"],
            intent=task["intent"],
            workdir=task.get("workdir"),
            resource_key=task.get("resource_key"),
        )
        return service.registry.update_task(
            child["id"], expected_version=child["version"], parent_id=task["id"]
        )

    return retry


def _int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return default
    return value if value > 0 else default


def _default_services() -> Any:
    """Build the plugin's own services once, on first request."""
    cached = _DEFAULT_CACHE.get("services")
    if cached is not None:
        return cached
    home = _neo_home()
    log_root = Path(os.environ.get(_LOG_ENV) or (home / "logs"))
    home.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)
    registry = Registry(
        path=os.environ.get(_DB_ENV) or str(home / "neo.db"),
        max_parallel=_int_env("NEO_MAX_PARALLEL", 4),
    )
    service = TaskService(registry)
    preview = PreviewService(
        registry,
        preview_root=os.environ.get("NEO_PREVIEW_DIR") or str(home / "previews"),
    )
    presenter = TaskPresenter(
        registry,
        PresentationService(MacOSOpener(), ledger=registry),
    )
    supervisor = None
    try:
        from tools import WorkerTokenStore
        from workers import ProcessSupervisor

        supervisor = ProcessSupervisor(
            service, WorkerTokenStore(registry), log_dir=log_root
        )
    except Exception:
        supervisor = None
    try:
        discovery = CapabilityDiscovery(
            run_command=_hermes_cli,
            plugin_names=NEO_KNOWN_CAPABILITIES,
            consent_store=JsonConsentStore(home / "capability-consents.json"),
        )
    except Exception:
        discovery = None
    services = SimpleNamespace(
        registry=registry,
        task_service=service,
        preview_service=preview,
        presenter=presenter,
        supervisor=supervisor,
        retry=_default_retry(service),
        capabilities=discovery,
        installer=None,
        log_root=log_root,
        artifact_max_bytes=_int_env("NEO_ARTIFACT_MAX_BYTES", _DEFAULT_ARTIFACT_MAX_BYTES),
        log_max_bytes=_int_env("NEO_LOG_MAX_BYTES", _DEFAULT_LOG_MAX_BYTES),
    )
    _DEFAULT_CACHE["services"] = services
    return services


router = build_router(_default_services)
"""Router Hermes mounts at ``/api/plugins/neo``; services stay lazy."""

__all__ = [
    "NEO_KNOWN_CAPABILITIES",
    "PLUGIN_VERSION",
    "build_router",
    "router",
]
