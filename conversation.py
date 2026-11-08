"""Conversational task tools for NEO: submit, inspect, refine, cancel, open.

This module is the talking half of NEO. It turns a natural request into one
durable task, answers questions from durable state, and routes a follow-up into
the exact native session that did the original work.

Rules it enforces:

* **Reads never write.** ``neo_task_status`` and ``neo_task_list`` create no
  task, emit no event and change no row; an unknown or ambiguous reference is
  reported, not resolved by guessing.
* **One question, then candidates.** An ambiguous reference returns exactly one
  clarifying question plus the candidate ids, because guessing corrupts work.
* **Exact sessions only.** A refinement is resumed on the session id the
  executor actually reported. A missing id is a refusal; a fresh-context worker
  is never substituted for a continuation.
* **No nested spawns.** A process marked ``NEO_WORKER=1`` cannot submit,
  refine or frame another NEO task.
* **Contained workspaces.** A requested workdir must already exist, be
  absolute and be a directory; otherwise the task runs in NEO's own
  plugin-data workspace. NEO never adopts an arbitrary process cwd.

``NeoRuntime`` is the composition root. It is built lazily on first use — from
``plugins.plugin_storage.plugin_data_dir("neo")`` — so importing this module
touches no profile. Tests inject a runtime whose ``launch`` records argv.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shlex  # noqa: F401  (kept for callers building their own argv lists)
import subprocess
import threading
import time
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional, Tuple
from uuid import uuid4

import schemas
from blender import (
    BlenderNotReadyError,
    BlenderPortBusyError,
    BlenderSlice,
    BlenderSliceError,
)
from browser import BrowserSlice, BrowserSliceError
from events import TaskService
from executors.base import ExecutorConfigError
from executors.claude import ClaudeExecutor
from executors.codex import CodexExecutor
from executors.hermes_worker import HermesWorkerExecutor, NestedWorkerRefusedError
from presentation import MacOSOpener, PresentationService, TaskPresenter
from registry import DuplicateTaskError, Registry, TaskNotFoundError
from state_machine import TERMINAL_STATES
from neo_tools import WorkerTokenStore, scrub_paths
from workers import ProcessSupervisor, worker_environment

TOOLSET = "neo"
WORKER_ENV = "NEO_WORKER"
WORKER_FLAG_VALUE = "1"
BLENDER_RESOURCE_KEY = "blender:9876"

KINDS = ("coding", "blender", "browser", "general")
ACTIVE_STATES = (
    "queued",
    "starting",
    "running",
    "waiting",
    "needs_approval",
    "needs_input",
    "cancelling",
    "unknown",
)
NEEDS_YOU_STATES = ("needs_approval", "needs_input")
TERMINAL_VALUES = frozenset(state.value for state in TERMINAL_STATES)
ALL_STATES = frozenset(ACTIVE_STATES) | TERMINAL_VALUES

CODING_EXECUTORS = ("codex", "claude")
WORKER_EXECUTORS = ("hermes_worker",)

SUMMARY_LIMIT = 2000
SUMMARY_MAX_TASKS = 5
INTENT_SUMMARY_CHARS = 48

_TASK_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_CAPABILITY_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,63}$")
_SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

_KIND_HINTS = (
    (
        "blender",
        (
            "blender",
            "3d model",
            "3d scene",
            "low-poly",
            "low poly",
            "render",
            "mesh",
            "rig",
        ),
    ),
    (
        "browser",
        (
            "browser",
            "web page",
            "webpage",
            "website",
            "navigate",
            "web form",
        ),
    ),
    (
        "coding",
        (
            "code",
            "coding",
            "implement",
            "refactor",
            "function",
            "bug",
            "fix",
            "test",
            "tests",
            "script",
            "parser",
            "api",
            "repository",
        ),
    ),
)


# -- payload helpers ----------------------------------------------------------


def _error(code: str, detail: str, **extra: object) -> dict:
    """Return one JSON-safe refusal with absolute paths removed."""
    payload = {
        "ok": False,
        "error": {"code": code, "detail": scrub_paths(detail)},
    }
    payload.update(extra)
    return payload


def _dump(payload: object) -> str:
    """Serialise a handler result; a serialisation failure is still JSON."""
    try:
        return json.dumps(payload, ensure_ascii=False, default=str)
    except Exception:
        return json.dumps(
            {
                "ok": False,
                "error": {
                    "code": "internal_error",
                    "detail": "NEO could not serialise that response.",
                },
            }
        )


def _guard(build: Callable[[], object]) -> str:
    """Run one handler body, converting any failure into honest JSON."""
    try:
        payload = build()
    except Exception as error:
        payload = _error(
            "internal_error",
            "NEO could not complete that request (%s)." % type(error).__name__,
        )
    return _dump(payload)


def _args(value: object) -> dict:
    return value if isinstance(value, dict) else {}


def _text(value: object, *, limit: int = 8000) -> Optional[str]:
    """Return trimmed text within ``limit``, or ``None`` for anything else."""
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text or len(text) > limit:
        return None
    return text


def _short(value: object, limit: int = INTENT_SUMMARY_CHARS) -> str:
    if not isinstance(value, str):
        return ""
    first = value.strip().splitlines()[0].strip() if value.strip() else ""
    if len(first) <= limit:
        return first
    return first[: limit - 3].rstrip() + "..."


def _safe_component(task_id: str) -> str:
    if _SAFE_COMPONENT.fullmatch(task_id):
        return task_id
    return "task-" + hashlib.sha256(task_id.encode("utf-8")).hexdigest()[:32]


def _decoded_list(raw: object) -> list:
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


def _age(seconds: float) -> str:
    total = int(max(0.0, seconds))
    if total < 60:
        return "%ds" % total
    if total < 3600:
        return "%dm" % (total // 60)
    if total < 86400:
        return "%dh" % (total // 3600)
    return "%dd" % (total // 86400)


def _worker_process() -> bool:
    return os.environ.get(WORKER_ENV) == WORKER_FLAG_VALUE


def _recursion_refusal() -> Optional[dict]:
    if not _worker_process():
        return None
    return _error(
        "recursion_refused",
        "NEO workers cannot start another NEO task; finish this task instead.",
    )


def conversation_key_from(kwargs: Mapping[str, object]) -> str:
    """Return the conversation key: the session id, then the task id context."""
    for field in ("session_id", "task_id"):
        value = kwargs.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def normalize_kind(raw: object, intent: str) -> str:
    """Return one of :data:`KINDS`; anything else is inferred from the intent."""
    if isinstance(raw, str):
        candidate = raw.strip().lower()
        if candidate in KINDS:
            return candidate
    text = intent.lower()
    for kind, hints in _KIND_HINTS:
        for hint in hints:
            pattern = r"(?<![a-z0-9])" + re.escape(hint) + r"(?![a-z0-9])"
            if re.search(pattern, text):
                return kind
    return "general"


def _safe_workdir(value: object) -> Optional[str]:
    """Return an existing absolute directory, or ``None`` when it is not safe."""
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text or "\x00" in text or not os.path.isabs(text):
        return None
    try:
        resolved = Path(text).resolve(strict=True)
    except (OSError, RuntimeError, ValueError):
        return None
    if not resolved.is_dir():
        return None
    return str(resolved)


def _artifact_names(value: object) -> Tuple[str, ...]:
    """Validate a task's optional expected artifact names."""
    if value is None:
        return ()
    if isinstance(value, str):
        names = (value,)
    elif isinstance(value, (list, tuple)):
        names = tuple(value)
    else:
        raise ValueError("expected_artifacts must be a list of relative names")
    clean: List[str] = []
    for name in names:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("every expected artifact must be a non-empty relative name")
        text = name.strip()
        if "\x00" in text or text.startswith("/") or ".." in Path(text).parts:
            raise ValueError("expected artifacts must stay inside the task workdir")
        clean.append(text)
    return tuple(clean)


def _dedupe_key(kind: str, intent: str) -> str:
    normalized = " ".join(intent.split()).lower()
    digest = hashlib.sha256(("%s\n%s" % (kind, normalized)).encode("utf-8")).hexdigest()
    return "%s:%s" % (kind, digest)


def _merge_expected(
    requested: Tuple[str, ...], defaults: object
) -> Tuple[str, ...]:
    """Keep the caller's expectations first, then the slice's fixed defaults."""

    merged: List[str] = list(requested)
    if isinstance(defaults, (list, tuple)):
        for name in defaults:
            if isinstance(name, str) and name.strip() and name not in merged:
                merged.append(name.strip())
    return tuple(merged)


def _prepare_blender(
    runtime: "NeoRuntime",
    intent: str,
    conversation: str,
    workdir: str,
    task_id: str,
) -> Tuple[Optional[dict], Optional[dict]]:
    """Ask the injected Blender slice for task fields, or refuse honestly.

    A runtime without the slice keeps the legacy worker route: the same
    ``hermes_worker`` submission this module used before Task 19. With the slice
    present, a missing or disabled plugin returns ``needs_approval`` plus the
    pinned disclosure and nothing is created or launched.
    """

    slice_ = getattr(runtime, "blender", None)
    if slice_ is None:
        return None, None
    try:
        return (
            slice_.prepare_task(intent, conversation, workdir, task_id=task_id),
            None,
        )
    except BlenderNotReadyError as error:
        decision = getattr(error, "decision", {}) or {}
        reason = decision.get("reason") or "Blender is not ready."
        return None, _error(
            "approval_required",
            "Blender needs approval before NEO can start it: %s" % reason,
            state="needs_approval",
            requires_approval=True,
            capability="blender",
            reason=reason,
            disclosure=decision.get("disclosure"),
            plugin=decision.get("plugin"),
            app=decision.get("app"),
        )
    except BlenderPortBusyError as error:
        return None, _error(
            "blender_port_busy",
            str(error),
            port=getattr(error, "status", None),
        )
    except BlenderSliceError as error:
        return None, _error("blender_unavailable", str(error))
    except (ExecutorConfigError, OSError, ValueError) as error:
        return None, _error("invalid_blender_task", str(error))


def _prepare_browser(
    runtime: "NeoRuntime",
    intent: str,
    conversation: str,
    workdir: str,
    task_id: str,
) -> Tuple[Optional[dict], Optional[dict]]:
    """Ask the injected browser slice for task fields, or refuse honestly.

    A runtime without the slice keeps the legacy worker route. With the slice
    present, preparation supplies the browser work order, the navigation
    artifacts and the worker argv; nothing is launched or published here.
    """

    slice_ = getattr(runtime, "browser", None)
    if slice_ is None:
        return None, None
    try:
        return (
            slice_.prepare_task(intent, conversation, workdir, task_id=task_id),
            None,
        )
    except BrowserSliceError as error:
        return None, _error("browser_unavailable", str(error))
    except (ExecutorConfigError, OSError, ValueError) as error:
        return None, _error("invalid_browser_task", str(error))


def _conversation_tasks(registry: Registry, conversation: str) -> List[dict]:
    """Return this conversation's tasks, newest first."""
    rows = [
        row
        for row in registry.list_tasks()
        if isinstance(row, dict) and row.get("conversation_key") == conversation
    ]
    rows.sort(key=lambda row: (row.get("created_at") or 0.0, row.get("id") or ""))
    rows.reverse()
    return rows


def _active_duplicate(
    runtime: "NeoRuntime", conversation: str, dedupe: str
) -> Optional[dict]:
    for row in _conversation_tasks(runtime.registry, conversation):
        if row.get("dedupe_key") == dedupe and row.get("state") in ACTIVE_STATES:
            return row
    return None


def _own_task(
    runtime: "NeoRuntime", conversation: str, task_id: object
) -> Tuple[Optional[dict], Optional[dict]]:
    """Return ``(task, None)`` or ``(None, refusal)`` without probing other rows."""
    if not isinstance(task_id, str) or not _TASK_ID.fullmatch(task_id):
        return None, _error("task_not_found", "No task with that id is in this conversation.")
    task = runtime.registry.get_task(task_id)
    if task is None or task.get("conversation_key") != conversation:
        return None, _error("task_not_found", "No task with that id is in this conversation.")
    return task, None


def _last_progress(
    service: TaskService, conversation: str, task_id: str
) -> Optional[float]:
    try:
        events = service.task_events(conversation, task_id)
    except Exception:
        return None
    for event in reversed(events):
        payload = event.get("payload")
        if not isinstance(payload, dict):
            continue
        value = payload.get("progress")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        return float(value)
    return None


def _fail(runtime: "NeoRuntime", task: dict, code: str, detail: str) -> dict:
    """Record an honest failure on a task that could not be started."""
    current = runtime.service.get_task(task["id"]) or task
    try:
        if current["state"] not in TERMINAL_VALUES:
            failed = runtime.service.transition(
                current["id"],
                "failed",
                expected_version=current["version"],
                summary=_short(detail, 240),
                event_extra={"error_code": code},
            )
            try:
                failed = runtime.registry.update_task(
                    failed["id"],
                    expected_version=failed["version"],
                    error_code=code,
                    error_detail=detail[:500],
                )
            except Exception:
                pass
            return runtime.service.get_task(failed["id"]) or failed
    except Exception:
        pass
    return runtime.service.get_task(current["id"]) or current


# -- composition root ---------------------------------------------------------


def _plugin_data_dir() -> Path:
    """Resolve NEO's durable data root through Hermes' public plugin storage."""
    from plugins.plugin_storage import plugin_data_dir

    return Path(plugin_data_dir("neo"))


def _int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return default
    return value if value > 0 else default


class NeoRuntime:
    """NEO's composed services plus the two injection points tests need.

    ``launch`` starts an executor for a task and is the only way this module
    reaches a process. A test supplies a recorder; production goes through the
    process supervisor for a new task and a detached spawn for a CLI follow-up.
    """

    def __init__(
        self,
        *,
        registry: Registry,
        service: TaskService,
        supervisor: object,
        presenter: object,
        data_dir: Path,
        codex: CodexExecutor,
        claude: ClaudeExecutor,
        hermes_worker: HermesWorkerExecutor,
        launch: Optional[Callable[..., object]] = None,
        coding_executor: str = "codex",
        preview_root: Optional[Path] = None,
        clock: Callable[[], float] = time.time,
        blender: object = None,
        browser: object = None,
        capabilities: object = None,
    ) -> None:
        self.registry = registry
        self.service = service
        self.supervisor = supervisor
        self.presenter = presenter
        self.data_dir = Path(data_dir)
        self.codex = codex
        self.claude = claude
        self.hermes_worker = hermes_worker
        self.coding_executor = coding_executor
        self.blender = blender
        """The optional Blender slice; ``None`` keeps the legacy worker route."""
        self.browser = browser
        """The optional browser slice; ``None`` keeps the legacy worker route."""
        self.capabilities = capabilities
        """Captured public capability lookups; ``None`` means fail closed."""
        self.browser_authorized = False
        """Explicit test/embedding opt-in; production checks the live bridge."""
        self.preview_root = (
            Path(preview_root) if preview_root is not None else self.data_dir / "previews"
        )
        self.clock = clock
        self._launch = launch
        self._workers: Dict[str, HermesWorkerExecutor] = {}

    # -- construction ----------------------------------------------------

    @classmethod
    def build(
        cls,
        *,
        data_dir: Optional[object] = None,
        launch: Optional[Callable[..., object]] = None,
        supervisor: object = None,
        presenter: object = None,
        coding_executor: Optional[str] = None,
        max_parallel: Optional[int] = None,
        clock: Callable[[], float] = time.time,
        blender: object = None,
        browser: object = None,
        capabilities: object = None,
    ) -> "NeoRuntime":
        """Build the real service graph under NEO's plugin data directory."""
        data = Path(data_dir) if data_dir is not None else _plugin_data_dir()
        data = Path(os.path.abspath(str(data)))
        data.mkdir(parents=True, exist_ok=True)
        data = data.resolve()
        log_dir = data / "logs"
        preview_root = data / "previews"
        for directory in (log_dir, preview_root, data / "tasks", data / "prompts"):
            directory.mkdir(parents=True, exist_ok=True)

        registry = Registry(
            path=data / "neo.db",
            max_parallel=max_parallel if max_parallel is not None else _int_env("NEO_MAX_PARALLEL", 4),
        )
        service = TaskService(registry)
        tokens = WorkerTokenStore(registry)
        supervisor = supervisor or ProcessSupervisor(service, tokens, log_dir=log_dir)
        # Restart recovery: record an honest outcome for work a previous run left
        # behind. It creates, launches and resumes nothing, so startup stays fast.
        reconcile = getattr(supervisor, "reconcile", None)
        if callable(reconcile):
            reconcile()
        presenter = presenter or TaskPresenter(
            registry, PresentationService(MacOSOpener(), ledger=registry)
        )
        selected = (
            coding_executor
            or os.environ.get("NEO_CODING_EXECUTOR")
            or "codex"
        )
        if selected not in CODING_EXECUTORS:
            selected = "codex"
        capabilities = capabilities or current_capability_context()
        runtime = cls(
            registry=registry,
            service=service,
            supervisor=supervisor,
            presenter=presenter,
            data_dir=data,
            codex=CodexExecutor(),
            claude=ClaudeExecutor(),
            hermes_worker=HermesWorkerExecutor(),
            launch=launch,
            coding_executor=selected,
            preview_root=preview_root,
            clock=clock,
            blender=blender,
            browser=browser,
            capabilities=capabilities,
        )
        if runtime.blender is None:
            runtime.blender = BlenderSlice(
                runtime=runtime,
                discovery=capabilities.discovery() if capabilities else None,
            )
        # Do not cache production authorization: a plugin can be disabled
        # between submission and follow-up. Tests may opt in explicitly.
        runtime.browser_authorized = False
        if runtime.browser is None:
            runtime.browser = BrowserSlice(runtime=runtime, clock=clock)
        return runtime

    # -- workspace and prompts -------------------------------------------

    def workspace(self, task_id: str) -> str:
        """Create and return this task's own workspace under plugin data."""
        directory = self.data_dir / "tasks" / _safe_component(task_id)
        directory.mkdir(parents=True, exist_ok=True)
        return str(directory.resolve())

    def prompt_file(self, task_id: str, text: str, workdir: str) -> str:
        """Write a worker prompt file for ``task_id`` and return its path."""
        prompts = self.data_dir / "prompts"
        prompts.mkdir(parents=True, exist_ok=True)
        path = str(prompts / ("%s.md" % _safe_component(task_id)))
        return self.hermes_worker.write_prompt_file(
            path, task_id=task_id, intent=text, workdir=workdir
        )

    def worker_executor(self, kind: str) -> HermesWorkerExecutor:
        """Return the Hermes worker executor for one kind of work."""
        cached = self._workers.get(kind)
        if cached is not None:
            return cached
        if kind == "blender":
            executor = HermesWorkerExecutor(skills=("blender:blender", "neo-worker"))
        else:
            executor = self.hermes_worker
        self._workers[kind] = executor
        return executor

    # -- launch ----------------------------------------------------------

    def launch(self, task_id: str, argv: object, *, role: str = "task") -> object:
        """Start ``argv`` for ``task_id``; the single process boundary."""
        argv = list(argv)  # type: ignore[arg-type]
        if self._launch is not None:
            return self._launch(task_id, argv, role=role)
        if role == "task":
            return self.supervisor.submit(task_id, argv)
        return self._spawn_follow_up(task_id, argv)

    def _spawn_follow_up(self, task_id: str, argv: List[str]) -> object:
        log_dir = self.data_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / ("%s-follow-up.log" % _safe_component(task_id))
        environment = worker_environment(task_id, None)
        environment[WORKER_ENV] = WORKER_FLAG_VALUE
        with log_path.open("ab", buffering=0) as log:
            return subprocess.Popen(
                argv,
                shell=False,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=environment,
                start_new_session=True,
                close_fds=True,
            )


_RUNTIME: Optional[NeoRuntime] = None
_RUNTIME_LOCK = threading.Lock()
# Registration and runtime publication are one profile-switch transaction.
_CAPABILITY_LOCK = _RUNTIME_LOCK
_CAPABILITY_CONTEXT = None


class CapabilityBridge:
    """Snapshot of the host's public capability lookups for this profile."""

    def __init__(
        self,
        *,
        has_plugin=None,
        is_plugin_enabled=None,
        find_skills=None,
        install_plugin=None,
        installed_sha=None,
    ) -> None:
        self._has_plugin = has_plugin if callable(has_plugin) else (lambda name: False)
        self._is_enabled = (
            is_plugin_enabled
            if callable(is_plugin_enabled)
            else (lambda name: bool(self._has_plugin(name)))
        )
        self._find_skills = find_skills if callable(find_skills) else (lambda: ())
        self._install_plugin = install_plugin if callable(install_plugin) else None
        self._installed_sha = installed_sha if callable(installed_sha) else (lambda name: None)

    def has_plugin(self, name: str) -> bool:
        try:
            return bool(self._has_plugin(name))
        except Exception:
            return False

    def is_plugin_enabled(self, name: str) -> bool:
        try:
            return bool(self._is_enabled(name))
        except Exception:
            return False

    def find_skills(self) -> list:
        try:
            return list(self._find_skills())
        except Exception:
            return []

    def browser_ready(self) -> bool:
        """A browser plugin/skill must be visibly present; core is not assumed."""
        try:
            for row in self.find_skills():
                if isinstance(row, Mapping):
                    name = row.get("name")
                    enabled = row.get("enabled", True)
                else:
                    name = getattr(row, "name", None)
                    enabled = getattr(row, "enabled", True)
                if name == "browser" and enabled:
                    return True
        except Exception:
            pass
        for name in ("browser", "browser-browse"):
            if self.is_plugin_enabled(name):
                return True
        return False

    def installed_sha(self, name: str):
        try:
            return self._installed_sha(name)
        except Exception:
            return None

    def _discovery_has_plugin(self, name: str) -> bool:
        if name == "blender":
            from capabilities import OFFICIAL_BLENDER

            if self.installed_sha(name) != OFFICIAL_BLENDER.sha:
                return False
        return self.has_plugin(name)

    def discovery(self):
        from capabilities import CapabilityDiscovery

        return CapabilityDiscovery(
            has_plugin=self._discovery_has_plugin,
            is_plugin_enabled=self.is_plugin_enabled,
            find_skills=self.find_skills,
            installer=self._install_plugin,
        )

    def list_installed(self, names=None):
        """Delegate installed capability reporting to a fresh discovery."""
        return self.discovery().list_installed(names)


def register_capability_context(ctx) -> Optional[CapabilityBridge]:
    """Capture public lookups from a plugin context; replace any old profile."""
    global _CAPABILITY_CONTEXT, _RUNTIME
    bridge = CapabilityBridge(
        has_plugin=getattr(ctx, "has_plugin", None),
        is_plugin_enabled=getattr(ctx, "is_plugin_enabled", None),
        find_skills=getattr(ctx, "find_skills", None),
        install_plugin=getattr(ctx, "install_plugin", None),
        installed_sha=(
            getattr(ctx, "get_installed_sha", None)
            or getattr(ctx, "installed_sha", None)
        ),
    )
    with _CAPABILITY_LOCK:
        _CAPABILITY_CONTEXT = bridge
        _RUNTIME = None
    return bridge


def current_capability_context() -> Optional[CapabilityBridge]:
    with _CAPABILITY_LOCK:
        return _CAPABILITY_CONTEXT


def reset_capability_context() -> None:
    global _CAPABILITY_CONTEXT, _RUNTIME
    with _CAPABILITY_LOCK:
        _CAPABILITY_CONTEXT = None
        _RUNTIME = None


def get_runtime() -> NeoRuntime:
    """Return the process-wide runtime, building it on first use."""
    global _RUNTIME
    if _RUNTIME is None:
        with _RUNTIME_LOCK:
            if _RUNTIME is None:
                _RUNTIME = NeoRuntime.build()
    return _RUNTIME


def reset_runtime() -> None:
    """Drop the cached runtime (tests and profile switches)."""
    global _RUNTIME
    with _RUNTIME_LOCK:
        _RUNTIME = None


# -- the tool surface ---------------------------------------------------------


class ConversationTools:
    """JSON handlers behind the seven conversational tools."""

    def __init__(self, runtime_provider: Optional[Callable[[], NeoRuntime]] = None) -> None:
        self._provider = runtime_provider or get_runtime

    def _runtime(self) -> NeoRuntime:
        return self._provider()

    # -- submit ----------------------------------------------------------

    def neo_submit_task(self, args=None, **kwargs) -> str:
        return _guard(lambda: self._submit(_args(args), kwargs))

    def _submit(self, args: dict, kwargs: Mapping[str, object]) -> dict:
        refusal = _recursion_refusal()
        if refusal is not None:
            return refusal
        intent = _text(args.get("intent"))
        if intent is None:
            return _error("invalid_intent", "A task needs a non-empty intent.")
        conversation = conversation_key_from(kwargs)
        if not conversation:
            return _error(
                "no_conversation",
                "NEO could not tell which conversation this belongs to.",
            )
        try:
            expected = _artifact_names(args.get("expected_artifacts"))
        except ValueError as error:
            return _error("invalid_artifacts", str(error))
        raw_capability = args.get("capability_id")
        capability = None
        if raw_capability is not None:
            if not isinstance(raw_capability, str) or not _CAPABILITY_ID.fullmatch(
                raw_capability.strip()
            ):
                return _error("invalid_capability", "That capability id is not one NEO accepts.")
            capability = raw_capability.strip()

        kind = normalize_kind(args.get("kind"), intent)
        runtime = self._runtime()
        dedupe = _dedupe_key(kind, intent)

        existing = _active_duplicate(runtime, conversation, dedupe)
        if existing is not None:
            return _submitted(runtime, existing, deduplicated=True)

        task_id = uuid4().hex
        requested = _safe_workdir(args.get("workdir"))
        workdir = requested or runtime.workspace(task_id)

        prepared: Optional[dict] = None
        if kind == "blender":
            prepared, refusal = _prepare_blender(
                runtime, intent, conversation, workdir, task_id
            )
            if refusal is not None:
                return refusal
            if prepared is not None:
                expected = _merge_expected(expected, prepared.get("expected_artifacts"))
        elif kind == "browser":
            authorized = bool(getattr(runtime, "browser_authorized", False)) or bool(
                getattr(runtime, "capabilities", None)
                and runtime.capabilities.browser_ready()
            )
            if not authorized:
                return _error(
                    "browser_unavailable",
                    "No enabled browser capability is available in this Hermes profile.",
                )
            prepared, refusal = _prepare_browser(
                runtime, intent, conversation, workdir, task_id
            )
            if refusal is not None:
                return refusal
            if prepared is not None:
                expected = _merge_expected(expected, prepared.get("expected_artifacts"))
        try:
            task = runtime.service.create_task(
                conversation_key=conversation,
                kind=kind,
                intent=intent,
                workdir=workdir,
                task_id=task_id,
                dedupe_key=dedupe,
                resource_key=(
                    prepared["resource_key"]
                    if prepared is not None
                    else (BLENDER_RESOURCE_KEY if kind == "blender" else None)
                ),
            )
        except DuplicateTaskError:
            existing = _active_duplicate(runtime, conversation, dedupe)
            if existing is None:
                return _error(
                    "duplicate_task",
                    "An equivalent task is already active but could not be read back.",
                )
            return _submitted(runtime, existing, deduplicated=True)
        if task["id"] != task_id:
            # The registry deduplicated inside its own transaction.
            return _submitted(runtime, task, deduplicated=True)

        executor_name = (
            runtime.coding_executor if kind == "coding" else "hermes_worker"
        )
        executor = (
            getattr(runtime, executor_name)
            if executor_name in CODING_EXECUTORS
            else runtime.worker_executor(kind)
        )
        session_id: Optional[str] = None
        try:
            if prepared is not None:
                executor_name = prepared["executor"]
                executor = runtime.worker_executor(kind)
                argv = list(prepared["argv"])
                if not argv:
                    raise ValueError(
                        "the %s slice did not build a worker argv" % kind
                    )
            elif executor_name == "claude":
                session_id = executor.new_session_id()
                argv = executor.build_initial_argv(
                    intent, workdir=workdir, session_id=session_id
                )
            elif executor_name == "hermes_worker":
                prompt = runtime.prompt_file(task["id"], intent, workdir)
                argv = executor.build_initial_argv(prompt, workdir=workdir)
            else:
                argv = executor.build_initial_argv(intent, workdir=workdir)
        except (ExecutorConfigError, OSError, ValueError) as error:
            failed = _fail(runtime, task, "executor_config", str(error))
            return _error(
                "executor_unavailable",
                "The task could not be prepared, so nothing was started.",
                task_id=failed["id"],
                state=failed["state"],
            )

        changes: dict = {"executor": executor_name}
        if session_id is not None:
            changes["executor_session_id"] = session_id
        if expected:
            changes["expected_json"] = json.dumps(list(expected))
        if capability is not None:
            changes["capability_id"] = capability
        try:
            row = runtime.registry.update_task(
                task["id"], expected_version=task["version"], **changes
            )
        except Exception:
            failed = _fail(runtime, task, "executor_config", "task preparation could not be recorded")
            return _error(
                "executor_unavailable",
                "The task could not be prepared, so nothing was started.",
                task_id=failed["id"],
                state=failed["state"],
            )

        try:
            runtime.launch(row["id"], argv, role="task")
        except NestedWorkerRefusedError:
            _fail(runtime, row, "recursion_refused", "a worker cannot spawn another worker")
            return _error(
                "recursion_refused",
                "NEO workers cannot start another NEO task; finish this task instead.",
                task_id=row["id"],
            )
        except Exception as error:
            failed = _fail(runtime, row, "spawn_failed", str(error))
            return _error(
                "executor_unavailable",
                "The task was recorded but its executor could not start.",
                task_id=failed["id"],
                state=failed["state"],
            )

        final = runtime.service.get_task(row["id"]) or row
        if final.get("state") == "failed":
            return _error(
                "executor_unavailable",
                "The task was recorded but its executor failed to start.",
                task_id=final["id"],
                state=final["state"],
            )
        payload = _submitted(runtime, final, deduplicated=False)
        payload["expected_artifacts"] = list(expected)
        if prepared is not None:
            payload["capability"] = prepared["capability"]
            payload["resource_key"] = prepared["resource_key"]
        return payload

    # -- status ----------------------------------------------------------

    def neo_task_status(self, args=None, **kwargs) -> str:
        return _guard(lambda: self._status(_args(args), kwargs))

    def _status(self, args: dict, kwargs: Mapping[str, object]) -> dict:
        conversation = conversation_key_from(kwargs)
        if not conversation:
            return _error(
                "no_conversation",
                "NEO could not tell which conversation this belongs to.",
            )
        runtime = self._runtime()
        task, refusal = _resolve_task(runtime, conversation, args)
        if refusal is not None:
            return refusal

        state = task["state"]
        error = None
        if task.get("error_code") or task.get("error_detail"):
            error = {
                "code": task.get("error_code"),
                "detail": scrub_paths(_short(task.get("error_detail"), 400) or ""),
            }
        created = task.get("created_at")
        age = runtime.clock() - created if isinstance(created, (int, float)) else 0.0
        return {
            "ok": True,
            "task_id": task["id"],
            "parent_id": task.get("parent_id"),
            "kind": task.get("kind"),
            "intent": task.get("intent"),
            "state": state,
            "version": task.get("version"),
            "executor": task.get("executor"),
            "progress": _last_progress(runtime.service, conversation, task["id"]),
            "needs": {
                "approval": state == "needs_approval",
                "input": state == "needs_input",
            },
            "error": error,
            "artifacts": _decoded_list(task.get("artifacts_json")),
            "expected_artifacts": _decoded_list(task.get("expected_json")),
            "preview": {
                "available": bool(task.get("preview_path"))
                or int(task.get("preview_rev") or 0) > 0,
                "rev": int(task.get("preview_rev") or 0),
            },
            "summary": scrub_paths(task.get("result_summary") or ""),
            "age_seconds": round(float(age), 3),
            "created_at": created,
            "updated_at": task.get("updated_at"),
        }

    # -- list ------------------------------------------------------------

    def neo_task_list(self, args=None, **kwargs) -> str:
        return _guard(lambda: self._list(_args(args), kwargs))

    def _list(self, args: dict, kwargs: Mapping[str, object]) -> dict:
        conversation = conversation_key_from(kwargs)
        if not conversation:
            return _error(
                "no_conversation",
                "NEO could not tell which conversation this belongs to.",
            )
        state = args.get("state")
        if state is not None and (
            not isinstance(state, str) or state not in ALL_STATES
        ):
            return _error("invalid_filter", "That task state is not one NEO records.")
        kind = args.get("kind")
        if kind is not None and (not isinstance(kind, str) or kind not in KINDS):
            return _error("invalid_filter", "That task kind is not one NEO records.")
        limit = args.get("limit", 10)
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 50:
            return _error("invalid_filter", "limit must be an integer from 1 to 50.")

        runtime = self._runtime()
        rows = _conversation_tasks(runtime.registry, conversation)
        selected = [
            row
            for row in rows
            if (state is None or row.get("state") == state)
            and (kind is None or row.get("kind") == kind)
        ][:limit]
        return {
            "ok": True,
            "count": len(selected),
            "total": len(rows),
            "tasks": [_list_entry(runtime, row) for row in selected],
        }

    # -- refine ----------------------------------------------------------

    def neo_task_refine(self, args=None, **kwargs) -> str:
        return _guard(lambda: self._refine(_args(args), kwargs))

    def _refine(self, args: dict, kwargs: Mapping[str, object]) -> dict:
        refusal = _recursion_refusal()
        if refusal is not None:
            return refusal
        conversation = conversation_key_from(kwargs)
        if not conversation:
            return _error(
                "no_conversation",
                "NEO could not tell which conversation this belongs to.",
            )
        follow_up = _text(args.get("follow_up"))
        if follow_up is None:
            return _error("invalid_follow_up", "A refinement needs a non-empty follow-up.")

        runtime = self._runtime()
        task, refusal = _own_task(runtime, conversation, args.get("task_id"))
        if refusal is not None:
            return refusal

        executor = task.get("executor")
        session = task.get("executor_session_id")
        if executor not in CODING_EXECUTORS + WORKER_EXECUTORS:
            return _error(
                "missing_session_id",
                "This task has no recorded executor session, so a follow-up would "
                "start from scratch. NEO will not substitute a fresh session.",
            )
        if not isinstance(session, str) or not session.strip():
            return _error(
                "missing_session_id",
                "This task has no recorded session id, so its context cannot be "
                "resumed. NEO will not substitute a fresh session.",
            )
        state = task["state"]
        active = state in ACTIVE_STATES

        try:
            if executor == "codex" and active:
                argv = runtime.codex.build_queue_argv(session, follow_up)
                runtime.launch(task["id"], argv, role="follow_up")
                return {
                    "ok": True,
                    "task_id": task["id"],
                    "executor": "codex",
                    "queued": True,
                    "state": "queued",
                    "task_state": state,
                    "session_id": session,
                    "note": (
                        "Codex queues this follow-up into the same thread; it runs "
                        "after the current turn rather than steering it."
                    ),
                }
            if executor == "claude" and active:
                return _error(
                    "active_session_refused",
                    "Claude Code cannot take a follow-up while this turn is active "
                    "without copying the session. Wait for it to finish, then ask "
                    "again; NEO will not fork a replacement session.",
                    task_id=task["id"],
                )
            if executor == "hermes_worker" and active:
                return _error(
                    "active_session_refused",
                    "This worker is still running. Wait for its turn to finish, then "
                    "ask again so NEO can resume the exact session.",
                    task_id=task["id"],
                )
        except ExecutorConfigError as error:
            return _error("invalid_follow_up", str(error))
        except Exception as error:
            return _error(
                "follow_up_failed",
                "The follow-up could not be started (%s)." % type(error).__name__,
                task_id=task["id"],
            )

        child_id = uuid4().hex
        workdir = _safe_workdir(task.get("workdir")) or runtime.workspace(child_id)
        try:
            if executor == "codex":
                argv = runtime.codex.build_resume_argv(session, follow_up, workdir=workdir)
            elif executor == "claude":
                argv = runtime.claude.build_resume_argv(session, follow_up, workdir=workdir)
            else:
                prompt = runtime.prompt_file(child_id, follow_up, workdir)
                argv = runtime.hermes_worker.build_resume_argv(
                    session, prompt, workdir=workdir
                )
        except (ExecutorConfigError, OSError, ValueError) as error:
            return _error("invalid_follow_up", str(error), task_id=task["id"])

        child = runtime.service.create_task(
            conversation_key=conversation,
            kind=task["kind"],
            intent=follow_up,
            workdir=workdir,
            task_id=child_id,
            resource_key=BLENDER_RESOURCE_KEY if task["kind"] == "blender" else None,
        )
        child = runtime.registry.update_task(
            child["id"],
            expected_version=child["version"],
            parent_id=task["id"],
            executor=executor,
            executor_session_id=session,
        )
        try:
            runtime.launch(child["id"], argv, role="task")
        except Exception as error:
            failed = _fail(runtime, child, "launch_failed", str(error))
            return _error(
                "launch_failed",
                "The refinement was recorded but its executor could not start.",
                task_id=failed["id"],
                state=failed["state"],
            )
        final = runtime.service.get_task(child["id"]) or child
        return {
            "ok": True,
            "task_id": final["id"],
            "parent_id": task["id"],
            "kind": final["kind"],
            "executor": executor,
            "resumed": True,
            "session_id": session,
            "state": final["state"],
        }

    # -- Blender frame follow-up -----------------------------------------

    def neo_task_frame_follow_up(self, args=None, **kwargs) -> str:
        return _guard(lambda: self._frame_follow_up(_args(args), kwargs))

    def _frame_follow_up(self, args: dict, kwargs: Mapping[str, object]) -> dict:
        refusal = _recursion_refusal()
        if refusal is not None:
            return refusal
        conversation = conversation_key_from(kwargs)
        if not conversation:
            return _error(
                "no_conversation",
                "NEO could not tell which conversation this belongs to.",
            )
        follow_up = _text(args.get("follow_up"))
        if follow_up is None:
            return _error("invalid_follow_up", "A Blender follow-up needs a non-empty prompt.")

        runtime = self._runtime()
        parent, refusal = _own_task(runtime, conversation, args.get("task_id"))
        if refusal is not None:
            return refusal
        if parent.get("kind") != "blender":
            return _error(
                "not_blender",
                "Frame follow-up only continues a Blender task; this task is %r."
                % (parent.get("kind"),),
                task_id=parent["id"],
            )

        prior_intent = parent.get("intent") or ""
        session = parent.get("executor_session_id")
        session = session.strip() if isinstance(session, str) else ""
        child_id = uuid4().hex
        workdir = _safe_workdir(parent.get("workdir")) or runtime.workspace(child_id)
        prompt_text = (
            "Original Blender task:\n%s\n\nFollow-up:\n%s\n\n"
            "Continue the same scene; keep the work inside the task workdir."
            % (prior_intent.strip(), follow_up)
        )
        executor = runtime.worker_executor("blender")
        try:
            prompt = runtime.prompt_file(child_id, prompt_text, workdir)
            if session:
                argv = executor.build_resume_argv(session, prompt, workdir=workdir)
                resumed = True
            else:
                argv = executor.build_initial_argv(prompt, workdir=workdir)
                resumed = False
        except (ExecutorConfigError, OSError, ValueError) as error:
            return _error("invalid_follow_up", str(error), task_id=parent["id"])

        try:
            child = runtime.service.create_task(
                conversation_key=conversation,
                kind="blender",
                intent=prompt_text,
                workdir=workdir,
                task_id=child_id,
                resource_key=BLENDER_RESOURCE_KEY,
            )
            changes = {
                "parent_id": parent["id"],
                "executor": "hermes_worker",
            }
            if session:
                changes["executor_session_id"] = session
            child = runtime.registry.update_task(
                child["id"], expected_version=child["version"], **changes
            )
            runtime.launch(child["id"], argv, role="task")
        except Exception as error:
            failed = _fail(runtime, locals().get("child", parent), "launch_failed", str(error))
            return _error(
                "launch_failed",
                "The Blender follow-up was recorded but its worker could not start.",
                task_id=failed.get("id", parent["id"]),
                state=failed.get("state", parent["state"]),
            )

        final = runtime.service.get_task(child["id"]) or child
        note = (
            "Resumed the recorded Blender session."
            if resumed
            else "No prior Blender session id was recorded, so this is a fresh worker "
            "carrying the previous intent; it is not a resumed Blender session."
        )
        return {
            "ok": True,
            "task_id": final["id"],
            "parent_id": parent["id"],
            "kind": "blender",
            "executor": "hermes_worker",
            "resumed": resumed,
            "prior_session_id": session or None,
            "state": final["state"],
            "note": note,
        }

    # -- cancel ----------------------------------------------------------

    def neo_task_cancel(self, args=None, **kwargs) -> str:
        return _guard(lambda: self._cancel(_args(args), kwargs))

    def _cancel(self, args: dict, kwargs: Mapping[str, object]) -> dict:
        conversation = conversation_key_from(kwargs)
        if not conversation:
            return _error(
                "no_conversation",
                "NEO could not tell which conversation this belongs to.",
            )
        runtime = self._runtime()
        task, refusal = _own_task(runtime, conversation, args.get("task_id"))
        if refusal is not None:
            return refusal
        if task["state"] in TERMINAL_VALUES:
            return {
                "ok": True,
                "task_id": task["id"],
                "state": task["state"],
                "version": task["version"],
                "cancelled": task["state"] == "cancelled",
                "already_terminal": True,
            }
        try:
            outcome = runtime.supervisor.cancel(task["id"])
        except KeyError:
            return _error("task_not_found", "No task with that id is in this conversation.")
        except Exception as error:
            return _error(
                "cancel_failed",
                "The task could not be cancelled (%s)." % type(error).__name__,
                task_id=task["id"],
            )
        row = outcome if isinstance(outcome, dict) and outcome.get("id") == task["id"] else None
        if row is None:
            row = runtime.service.get_task(task["id"]) or task
        return {
            "ok": row["state"] == "cancelled",
            "task_id": row["id"],
            "state": row["state"],
            "version": row["version"],
            "cancelled": row["state"] == "cancelled",
            "already_terminal": row["state"] in TERMINAL_VALUES,
        }

    # -- open ------------------------------------------------------------

    def neo_task_open(self, args=None, **kwargs) -> str:
        return _guard(lambda: self._open(_args(args), kwargs))

    def _open(self, args: dict, kwargs: Mapping[str, object]) -> dict:
        conversation = conversation_key_from(kwargs)
        if not conversation:
            return _error(
                "no_conversation",
                "NEO could not tell which conversation this belongs to.",
            )
        task_id = args.get("task_id")
        runtime = self._runtime()
        task, refusal = _own_task(runtime, conversation, task_id)
        if refusal is not None:
            return refusal
        try:
            outcome = runtime.presenter.present_task(task["id"], conversation)
        except TaskNotFoundError:
            return _error("task_not_found", "No task with that id is in this conversation.")
        except Exception as error:
            return _error(
                "presentation_failed",
                "The result could not be presented (%s)." % type(error).__name__,
                task_id=task["id"],
            )
        presented = bool(outcome.get("presented"))
        return {
            "ok": presented,
            "task_id": task["id"],
            "presented": presented,
            "skipped": bool(outcome.get("skipped", not presented)),
            "reason": scrub_paths(outcome.get("reason") or ""),
        }


# -- resolution and projection helpers ---------------------------------------


def _resolve_task(
    runtime: NeoRuntime, conversation: str, args: dict
) -> Tuple[Optional[dict], Optional[dict]]:
    """Resolve a task by id or unambiguous partial intent, else explain why not."""
    raw = args.get("task_id")
    if raw is not None and raw != "":
        return _own_task(runtime, conversation, raw)

    tasks = _conversation_tasks(runtime.registry, conversation)
    query = args.get("intent")
    if isinstance(query, str) and query.strip():
        needle = query.strip().lower()
        matches = [
            row
            for row in tasks[:20]
            if needle in (row.get("intent") or "").lower()
        ]
    else:
        matches = [row for row in tasks if row.get("state") in ACTIVE_STATES][:20]

    if len(matches) == 1:
        return matches[0], None
    if not matches:
        return None, _error(
            "task_not_found",
            "No task in this conversation matches that reference.",
        )
    return None, {
        "ok": False,
        "status": "ambiguous",
        "question": "Which task did you mean?",
        "candidates": [
            {
                "task_id": row["id"],
                "intent": _short(row.get("intent"), 80),
                "state": row.get("state"),
            }
            for row in matches[:5]
        ],
    }


def _submitted(runtime: NeoRuntime, task: dict, *, deduplicated: bool) -> dict:
    workdir = task.get("workdir")
    source = "plugin_data"
    if isinstance(workdir, str):
        try:
            Path(workdir).resolve().relative_to(runtime.data_dir.resolve())
        except ValueError:
            source = "requested"
    return {
        "ok": True,
        "task_id": task["id"],
        "state": task["state"],
        "executor": task.get("executor"),
        "kind": task.get("kind"),
        "deduplicated": deduplicated,
        "workdir_source": source,
    }


def _list_entry(runtime: NeoRuntime, task: dict) -> dict:
    created = task.get("created_at")
    age = runtime.clock() - created if isinstance(created, (int, float)) else 0.0
    return {
        "task_id": task["id"],
        "parent_id": task.get("parent_id"),
        "kind": task.get("kind"),
        "intent": _short(task.get("intent"), 80),
        "state": task.get("state"),
        "executor": task.get("executor"),
        "version": task.get("version"),
        "needs": {
            "approval": task.get("state") == "needs_approval",
            "input": task.get("state") == "needs_input",
        },
        "age_seconds": round(float(age), 3),
    }


# -- pre_llm_call middleware --------------------------------------------------


def _summary_priority(task: dict) -> tuple:
    state = task.get("state")
    return (
        0 if state in NEEDS_YOU_STATES else 1,
        -(task.get("created_at") or 0.0),
        task.get("id") or "",
    )


def _summary_line(runtime: NeoRuntime, task: dict, now: float) -> str:
    created = task.get("created_at")
    age = now - created if isinstance(created, (int, float)) else 0.0
    flags = []
    if task.get("state") == "needs_approval":
        flags.append("needs approval")
    elif task.get("state") == "needs_input":
        flags.append("needs your input")
    suffix = " [%s]" % ", ".join(flags) if flags else ""
    return "- %s %s - %s - %s ago%s" % (
        task["id"],
        _short(task.get("intent")),
        task.get("state"),
        _age(age),
        suffix,
    )


def build_task_summary(
    runtime: NeoRuntime, session_id: str, *, limit: int = SUMMARY_LIMIT
) -> Optional[str]:
    """Return the bounded active-task summary for one conversation, or ``None``."""
    tasks = [
        row
        for row in _conversation_tasks(runtime.registry, session_id)
        if row.get("state") in ACTIVE_STATES
    ]
    if not tasks:
        return None
    tasks.sort(key=_summary_priority)
    now = runtime.clock()
    lines = [
        _summary_line(runtime, task, now) for task in tasks[:SUMMARY_MAX_TASKS]
    ]
    header = "NEO active tasks in this conversation (context only, do not re-announce):"
    text = "\n".join([header, *lines])
    if len(text) > limit:
        text = text[: max(0, limit - 3)].rstrip() + "..."
    return text


def make_pre_llm_call(
    runtime_provider: Optional[Callable[[], NeoRuntime]] = None,
):
    """Return the ``pre_llm_call`` hook bound to a runtime provider."""
    provider = runtime_provider or get_runtime

    def pre_llm_call(
        session_id=None,
        user_message=None,
        conversation_history=None,
        **kwargs,
    ):
        try:
            if _worker_process():
                return None
            if not isinstance(session_id, str) or not session_id.strip():
                return None
            runtime = provider()
            text = build_task_summary(runtime, session_id.strip())
            if not text:
                return None
            return {"context": text}
        except Exception:
            # A middleware must never break the turn it annotates.
            return None

    return pre_llm_call


pre_llm_call = make_pre_llm_call()
"""The hook Hermes registers; it resolves NEO's runtime only on a live turn."""


# -- registration -------------------------------------------------------------


def register_conversation_tools(
    ctx, runtime_provider: Optional[Callable[[], NeoRuntime]] = None
) -> ConversationTools:
    """Register the seven conversational tools on ``ctx`` under one toolset."""
    tools = ConversationTools(runtime_provider)
    for name in schemas.CONVERSATION_TOOL_NAMES:
        ctx.register_tool(
            name=name,
            toolset=TOOLSET,
            schema=schemas.conversation_schema_for(name),
            handler=getattr(tools, name),
        )
    return tools


__all__ = [
    "ACTIVE_STATES",
    "BLENDER_RESOURCE_KEY",
    "CODING_EXECUTORS",
    "ConversationTools",
    "KINDS",
    "NEEDS_YOU_STATES",
    "NeoRuntime",
    "SUMMARY_LIMIT",
    "SUMMARY_MAX_TASKS",
    "TOOLSET",
    "build_task_summary",
    "conversation_key_from",
    "get_runtime",
    "make_pre_llm_call",
    "normalize_kind",
    "pre_llm_call",
    "register_conversation_tools",
    "reset_runtime",
]
