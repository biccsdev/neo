"""The Blender vertical slice: probe, readiness, consent, task, verification.

Blender is not a normal capability. Its tools execute arbitrary Python, its
session is online, and its local bridge on ``127.0.0.1:9876`` has no
authentication while Blender is open. So this module keeps three facts apart
that are easy to blur:

* **App presence.** ``/Applications/Blender.app`` either exists and reports a
  version or it does not. That is a fact about the Mac and nothing more.
* **Route readiness.** A Blender task may only start when the official
  ``blender`` plugin is installed *and* enabled, the app is present, and the
  bridge port is not held by a process NEO did not start. Discovery is never
  treated as scene readiness: ``scene_ready`` stays ``False`` until a worker has
  connected and read the scene with ``get_objects_summary``.
* **Completion.** A task completes only with a real ``.blend`` inside its own
  workdir (plus any expected screenshot), verified by :mod:`artifacts`. A path,
  a process exit or a claim is not a result.

The slice is injected with its runtime, discovery, consent store and installer.
It never installs implicitly, never spawns a process, and never retries a busy
port into an existing Blender instance. Task preparation returns *fields*; the
caller owns task creation so the registry keeps its single-writer rules and
queues a second Blender task on the same ``blender:9876`` resource key.

Sections 4.2, 4.7 and 5 of the implementation plan (Task 19).
"""

from __future__ import annotations

import json
import os
import re
import socket
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence, Tuple

from artifacts import (
    DEFAULT_ALLOWED_EXTENSIONS,
    ArtifactError,
    completion_evidence,
    verify_artifact,
)
from capabilities import (
    OFFICIAL_BLENDER,
    Capability,
    CapabilityDiscovery,
    ConsentRecord,
    ConsentRequiredError,
    ConsentStore,
    blender_disclosure,
)
from state_machine import CompletionEvidence

BLENDER_APP_PATH = "/Applications/Blender.app"
"""The macOS application bundle this slice probes by default."""

BLENDER_BINARY = "Contents/MacOS/Blender"
BLENDER_HOST = "127.0.0.1"
BLENDER_PORT = 9876
RESOURCE_KEY = "blender:9876"
"""The one resource a Blender task holds: the unauthenticated bridge port."""

WORKER_SKILLS: Tuple[str, ...] = ("blender:blender", "neo-worker")
SCENE_NAME = "scene.blend"
"""The file name the worker is told to save; completion requires a .blend."""

WORKER_EXECUTOR = "hermes_worker"

ACTIVE_RESOURCE_STATES = (
    "starting",
    "running",
    "waiting",
    "needs_approval",
    "needs_input",
    "cancelling",
    "unknown",
)

_VERSION_RE = re.compile(r"Blender\s+([0-9][0-9A-Za-z.+-]*)")
_BLEND_MAGIC = (b"BLENDER", b"\x1f\x8b", b"\x28\xb5\x2f\xfd")
"""Blender saves start with BLENDER, gzip or zstd depending on compression."""

_MAX_INTENT = 8000
_SCENE_NOTE = (
    "App presence is not a live scene. Scene readiness is only known after the "
    "worker connects and get_objects_summary returns."
)


# -- errors -------------------------------------------------------------------


class BlenderSliceError(Exception):
    """Base class for an honest Blender refusal."""


class BlenderNotReadyError(BlenderSliceError):
    """A Blender task was requested before the route was ready."""

    def __init__(self, decision: Mapping[str, Any]) -> None:
        self.decision = dict(decision)
        super().__init__(str(self.decision.get("reason") or "Blender is not ready"))


class BlenderPortBusyError(BlenderSliceError):
    """Port 9876 is held by a process NEO did not start."""

    def __init__(self, detail: str, status: Mapping[str, Any]) -> None:
        self.status = dict(status)
        super().__init__(detail)


class BlenderCompletionError(BlenderSliceError):
    """A Blender task cannot be completed from the evidence it offers."""


# -- probe --------------------------------------------------------------------


@dataclass(frozen=True)
class BlenderProbe:
    """The observed state of the Blender application bundle."""

    path: str
    present: bool
    version: str = ""
    detail: str = ""

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "present": self.present,
            "version": self.version,
            "detail": self.detail,
        }


def parse_blender_version(text: Any) -> str:
    """Return ``5.1.2`` from Blender's ``--version`` output, else ``""``."""

    if not isinstance(text, str):
        return ""
    match = _VERSION_RE.search(text)
    return match.group(1) if match is not None else ""


def _default_probe(path: str) -> BlenderProbe:
    """Run ``Blender --version`` as an argv array; never through a shell."""

    binary = Path(path) / BLENDER_BINARY
    if not binary.is_file():
        return BlenderProbe(
            path=path,
            present=False,
            version="",
            detail="Blender.app was not found at %s" % path,
        )
    try:
        completed = subprocess.run(
            [str(binary), "--version"],
            shell=False,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=20,
        )
    except (OSError, subprocess.SubprocessError) as error:
        return BlenderProbe(
            path=path,
            present=True,
            version="",
            detail="Blender did not report a version (%s)" % type(error).__name__,
        )
    output = (completed.stdout or "") + "\n" + (completed.stderr or "")
    version = parse_blender_version(output)
    return BlenderProbe(
        path=path,
        present=True,
        version=version,
        detail="" if version else "Blender ran but did not report a version",
    )


def _coerce_probe(value: Any, path: str) -> BlenderProbe:
    if isinstance(value, BlenderProbe):
        return value
    if isinstance(value, Mapping):
        return BlenderProbe(
            path=str(value.get("path") or path),
            present=bool(value.get("present")),
            version=str(value.get("version") or ""),
            detail=str(value.get("detail") or ""),
        )
    raise TypeError("probe must return a BlenderProbe or a mapping")


def _default_port_check(host: str, port: int) -> bool:
    """Return whether something is listening; a local connect, no traffic."""

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(0.25)
    try:
        return sock.connect_ex((host, int(port))) == 0
    except OSError:
        return False
    finally:
        sock.close()


# -- prompt -------------------------------------------------------------------


def build_worker_prompt(intent: str, workdir: str) -> str:
    """Render the Blender work order: steps, boundaries, verification."""

    text = intent.strip() if isinstance(intent, str) else ""
    if not text or len(text) > _MAX_INTENT:
        raise ValueError("a Blender task needs a non-empty intent")
    if not isinstance(workdir, str) or not workdir.strip():
        raise ValueError("a Blender task needs an absolute workdir")
    return "\n".join(
        (
            "## Blender work order",
            "",
            "Intent: %s" % text,
            "",
            "Work only inside %s. Drive Blender through the official `blender` "
            "plugin's MCP tools (skill `blender:blender`). Its bridge is the "
            "local, unauthenticated endpoint 127.0.0.1:9876 and accepts one "
            "task at a time." % workdir,
            "",
            "### Steps",
            "1. Read the current scene with `get_objects_summary` before you "
            "change anything. Report what is already there.",
            "2. Build or edit the object with low-poly primitives: no "
            "subdivision or smooth-by-default modifiers, keep the polycount "
            "deliberately low, and name every object meaningfully.",
            "3. After each modelling phase, capture a viewport screenshot "
            "inside %s and publish it with `neo_task_publish_preview`, so the "
            "preview updates phase by phase." % workdir,
            "4. Verify the result with `get_objects_summary` again: object "
            "names, counts and transforms must match what you intended. A "
            "screenshot alone is not verification.",
            "5. Save the scene as `%s` inside %s. Completion requires that "
            "real file; a saved path or a claim is not a result."
            % (SCENE_NAME, workdir),
            "",
            "### Approval boundaries",
            "- Ask before installing or enabling any Blender add-on, and stop "
            "until the user approves.",
            "- Ask before publishing, uploading, posting or sending anything "
            "outside this Mac.",
            "- Never widen your own permissions or change the user's "
            "configuration.",
            "",
            "### Recursion boundary",
            "Do not start a nested NEO task, a second worker or another Hermes "
            "session. One task, one worker. Report the need and stop instead.",
            "",
            "### Reporting",
            "Use the NEO bridge tools for progress, previews, completion and "
            "failure. Never print, paste or write the task token anywhere.",
        )
    )


# -- the slice ----------------------------------------------------------------


class BlenderSlice:
    """Probe, route, install and verify the official Blender capability."""

    def __init__(
        self,
        *,
        runtime: Any = None,
        discovery: Optional[CapabilityDiscovery] = None,
        consent: Optional[ConsentStore] = None,
        installer: Any = None,
        probe: Optional[Callable[[str], Any]] = None,
        port_check: Optional[Callable[[str, int], Any]] = None,
        capability: Optional[Capability] = None,
        app_path: str = BLENDER_APP_PATH,
        host: str = BLENDER_HOST,
        port: int = BLENDER_PORT,
    ) -> None:
        self.runtime = runtime
        self.capability = capability or OFFICIAL_BLENDER
        self.app_path = app_path
        self.host = host
        self.port = int(port)
        self.consent = consent
        self.discovery = discovery if discovery is not None else CapabilityDiscovery(
            consent_store=consent,
            installer=installer,
        )
        self._probe = probe if probe is not None else _default_probe
        self._port_check = port_check if port_check is not None else _default_port_check

    # -- probe and readiness ----------------------------------------------

    def probe(self) -> BlenderProbe:
        """Return what the Mac reports about Blender; never scene readiness."""

        try:
            return _coerce_probe(self._probe(self.app_path), self.app_path)
        except BlenderSliceError:
            raise
        except Exception as error:
            return BlenderProbe(
                path=self.app_path,
                present=False,
                version="",
                detail="Blender could not be probed (%s)" % type(error).__name__,
            )

    def port_status(self) -> dict:
        """Whether the bridge port is listening; never whether it is ours."""

        listening = False
        try:
            listening = bool(self._port_check(self.host, self.port))
        except Exception:
            listening = False
        return {
            "host": self.host,
            "port": self.port,
            "listening": listening,
            "authenticated": False,
        }

    def readiness(self) -> dict:
        """Report route readiness: app, plugin install/enable, port, scene."""

        app = self.probe()
        decision = self.discovery.route_blender()
        installed = bool(decision.get("installed"))
        enabled = bool(decision.get("enabled"))
        disclosure = decision.get("disclosure")
        if disclosure is None and decision.get("status") != "ready":
            disclosure = blender_disclosure(self.capability)

        if decision.get("status") == "ready" and app.present:
            status = "ready"
            reason = ""
        elif not app.present:
            status = "needs_approval"
            reason = "Blender.app was not found at %s; NEO cannot open a scene." % (
                self.app_path,
            )
            if disclosure is None:
                disclosure = blender_disclosure(self.capability)
        elif not installed:
            status = "needs_approval"
            reason = "the official blender plugin is not installed"
        else:
            status = "needs_approval"
            reason = "the official blender plugin is installed but not enabled"

        ready = status == "ready"
        return {
            "capability": self.capability.name,
            "status": status,
            "ready": ready,
            "route": "hermes_worker" if ready else None,
            "executor": WORKER_EXECUTOR if ready else None,
            "skill": "blender:blender" if ready else None,
            "skills": list(WORKER_SKILLS) if ready else [],
            "resource_key": RESOURCE_KEY,
            "plugin": {
                "name": self.capability.name,
                "installed": installed,
                "enabled": enabled,
            },
            "app": app.to_dict(),
            "scene_ready": False,
            "scene_note": _SCENE_NOTE,
            "port": self.port_status(),
            "disclosure": None if ready else disclosure,
            "reason": reason,
        }

    # -- install ------------------------------------------------------------

    def install(self, consent: Any) -> dict:
        """Install the pinned capability with exact consent, then verify it.

        No installer flag, no repository URL and no dependency is ever passed:
        :meth:`CapabilityDiscovery.install_capability` builds the one pinned
        request shape. A caller cannot hand this method extra arguments.
        """

        capability = self.capability
        if not isinstance(consent, ConsentRecord):
            raise ConsentRequiredError(
                "installing blender requires an exact ConsentRecord for %s at %s"
                % (capability.name, capability.sha)
            )
        if not consent.matches(capability.name, capability.sha):
            raise ConsentRequiredError(
                "consent for %s at %s does not authorise %s at %s"
                % (consent.capability, consent.sha, capability.name, capability.sha)
            )

        store = getattr(self.discovery, "consent_store", None) or self.consent
        if store is not None and store.find(capability.name, capability.sha) is None:
            store.save(consent)

        result = self.discovery.install_capability(capability)
        decision = self.discovery.route_blender()
        enabled = decision.get("status") == "ready"
        ready = bool(result.installed) and bool(result.verified) and enabled
        reason = result.reason
        if not reason and not enabled:
            reason = "the official blender plugin is installed but not enabled"
        return {
            "installed": bool(result.installed),
            "verified": bool(result.verified),
            "enabled": enabled,
            "status": "ready" if ready else "needs_approval",
            "name": result.name,
            "version": capability.version,
            "sha": capability.sha,
            "reason": reason,
            "disclosure": None if ready else blender_disclosure(capability),
        }

    # -- task preparation ---------------------------------------------------

    def prepare_task(
        self,
        intent: str,
        conversation_key: str,
        workdir: Optional[str] = None,
        *,
        task_id: Optional[str] = None,
    ) -> dict:
        """Return the task fields, but only after readiness and consent.

        Raises :class:`BlenderNotReadyError` with the pinned disclosure when the
        route is not ready, and :class:`BlenderPortBusyError` when port 9876 is
        held by a process NEO did not start. Nothing is created or spawned here.
        """

        text = intent.strip() if isinstance(intent, str) else ""
        if not text or len(text) > _MAX_INTENT:
            raise ValueError("a Blender task needs a non-empty intent")
        if not isinstance(conversation_key, str) or not conversation_key.strip():
            raise ValueError("a Blender task needs a conversation key")
        directory = self._validated_workdir(workdir)

        decision = self.readiness()
        if not decision["ready"]:
            raise BlenderNotReadyError(decision)

        conflict = self._conflict(decision["port"])
        if conflict is not None:
            raise BlenderPortBusyError(str(conflict["detail"]), conflict)

        prompt = build_worker_prompt(text, directory)
        fields: dict = {
            "kind": "blender",
            "capability": self.capability.name,
            "executor": WORKER_EXECUTOR,
            "resource_key": RESOURCE_KEY,
            "skills": tuple(WORKER_SKILLS),
            "intent": text,
            "prompt": prompt,
            "workdir": directory,
            "expected_artifacts": (SCENE_NAME,),
            "scene_ready": False,
            "app": decision["app"],
        }
        if task_id is not None and self.runtime is not None:
            prompt_path = self.runtime.prompt_file(task_id, prompt, directory)
            executor = self.runtime.worker_executor("blender")
            fields["prompt_path"] = prompt_path
            fields["argv"] = tuple(executor.build_initial_argv(prompt_path, workdir=directory))
        else:
            fields["prompt_path"] = None
            fields["argv"] = ()
        return fields

    def _validated_workdir(self, workdir: Optional[str]) -> str:
        if not isinstance(workdir, str) or not workdir.strip():
            raise ValueError("a Blender task needs an absolute existing workdir")
        text = workdir.strip()
        if "\x00" in text or not os.path.isabs(text):
            raise ValueError("a Blender task needs an absolute existing workdir")
        try:
            resolved = Path(text).resolve(strict=True)
        except (OSError, RuntimeError, ValueError):
            raise ValueError("a Blender task needs an absolute existing workdir") from None
        if not resolved.is_dir():
            raise ValueError("a Blender task workdir must be a directory")
        return str(resolved)

    def port_conflict(self) -> Optional[dict]:
        """Return an honest conflict only when NEO does not own the listener.

        A port held by a NEO Blender task is not a conflict: the registry queues
        the second task on ``blender:9876``. Any other listener is reported and
        never retried, because NEO cannot authenticate or clear it.
        """

        return self._conflict(self.port_status())

    def _conflict(self, status: Mapping[str, Any]) -> Optional[dict]:
        if not status.get("listening"):
            return None
        if self._owns_resource():
            return None
        return {
            **dict(status),
            "detail": (
                "%s:%d is already listening and NEO did not start it; NEO will "
                "not reuse or retry into an existing Blender session."
                % (self.host, self.port)
            ),
        }

    def _owns_resource(self) -> bool:
        registry = getattr(self.runtime, "registry", None)
        if registry is None:
            return False
        try:
            rows = registry.list_tasks()
        except Exception:
            return False
        for row in rows or ():
            if not isinstance(row, Mapping):
                continue
            if row.get("resource_key") == RESOURCE_KEY and row.get("state") in ACTIVE_RESOURCE_STATES:
                return True
        return False

    # -- completion ---------------------------------------------------------

    def verify_completion(self, task: Mapping[str, Any]) -> CompletionEvidence:
        """Return completion evidence, or refuse when the result is not real.

        A ``.blend`` must exist inside the task's own workdir, be a non-empty
        regular file and start like a Blender save. Every artifact the task
        expected (typically a phase screenshot) must verify too.
        """

        if not isinstance(task, Mapping):
            raise BlenderCompletionError("a Blender task row is required to verify completion")
        workdir = task.get("workdir")
        expected = _decoded_expected(task.get("expected_json"))
        blend = self._choose_blend(workdir, expected)

        ordered = list(expected)
        if blend not in ordered:
            ordered.append(blend)
        try:
            evidence = completion_evidence(
                workdir, ordered, allowed_extensions=DEFAULT_ALLOWED_EXTENSIONS
            )
            artifact = verify_artifact(
                workdir, blend, allowed_extensions=DEFAULT_ALLOWED_EXTENSIONS
            )
        except ArtifactError as error:
            raise BlenderCompletionError(str(error)) from error

        path = artifact.path
        try:
            size = path.stat().st_size
        except OSError:
            raise BlenderCompletionError("the saved .blend cannot be read") from None
        if size <= 0:
            raise BlenderCompletionError("the saved .blend is empty; nothing was saved")
        try:
            with path.open("rb") as handle:
                header = handle.read(8)
        except OSError:
            raise BlenderCompletionError("the saved .blend cannot be read") from None
        if not any(header.startswith(magic) for magic in _BLEND_MAGIC):
            raise BlenderCompletionError(
                "the saved .blend does not start like a Blender file"
            )
        return evidence

    def _choose_blend(self, workdir: Any, expected: Sequence[str]) -> str:
        named = [name for name in expected if name.lower().endswith(".blend")]
        for name in named:
            try:
                verify_artifact(workdir, name, allowed_extensions=DEFAULT_ALLOWED_EXTENSIONS)
                return name
            except ArtifactError:
                continue
        try:
            root = Path(str(workdir)).resolve(strict=True)
        except (OSError, RuntimeError, ValueError):
            raise BlenderCompletionError(
                "the task workdir is not readable; no .blend can be verified"
            ) from None
        for candidate in sorted(root.iterdir()):
            if candidate.suffix.lower() != ".blend":
                continue
            if candidate.is_symlink() or not candidate.is_file():
                continue
            return candidate.name
        raise BlenderCompletionError(
            "completion requires a real .blend inside the task workdir; none was found"
        )


def _decoded_expected(raw: Any) -> Tuple[str, ...]:
    """Decode ``expected_json`` into names without trusting its shape."""

    if isinstance(raw, (list, tuple)):
        values = raw
    elif isinstance(raw, (str, bytes)):
        try:
            values = json.loads(raw)
        except (TypeError, ValueError):
            return ()
    else:
        return ()
    if not isinstance(values, (list, tuple)):
        return ()
    return tuple(item for item in values if isinstance(item, str) and item.strip())


__all__ = [
    "BLENDER_APP_PATH",
    "BLENDER_HOST",
    "BLENDER_PORT",
    "RESOURCE_KEY",
    "SCENE_NAME",
    "WORKER_SKILLS",
    "BlenderCompletionError",
    "BlenderNotReadyError",
    "BlenderPortBusyError",
    "BlenderProbe",
    "BlenderSlice",
    "BlenderSliceError",
    "build_worker_prompt",
    "parse_blender_version",
]
