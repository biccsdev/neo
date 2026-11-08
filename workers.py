"""Detached subprocess supervision for NEO tasks.

The supervisor owns processes, not lifecycle truth. It persists receipts and
worker metadata, cancels by process group, and lets the task service decide
valid state changes. A detached worker outlives the viewer.

Restart recovery lives here too. :meth:`ProcessSupervisor.reconcile` reads the
durable task rows once, checks each recorded PID against the operating system,
and records an honest outcome. It never launches, resumes or kills anything: a
live worker is adopted by a polling monitor because a process this interpreter
did not spawn cannot be waited on, and a dead worker is already dead.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import signal
import subprocess
import threading
import time
from pathlib import Path
from typing import Callable, Optional, Union
from uuid import uuid4

from events import TaskService
from registry import (
    CapacityExceededError,
    RegistryError,
    ResourceBusyError,
    VersionConflictError,
)
from state_machine import (
    InvalidTransitionError,
    StateMachineError,
    TerminalStateError,
)
from neo_tools import WorkerTokenStore

TERMINAL_STATES = frozenset({"completed", "failed", "cancelled"})
_UNCLAIMED_EXIT_STATES = frozenset(
    {"queued", "starting", "running", "waiting", "cancelling", "unknown"}
)
_SAFE_TASK_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

RECOVERABLE_STATES = frozenset(
    {
        "queued",
        "starting",
        "running",
        "waiting",
        "needs_approval",
        "needs_input",
        "cancelling",
        "unknown",
    }
)
"""Every state a restart passes over; queued work is unfinished but processless."""

_PROCESSLESS_STATES = frozenset({"queued"})
_ZOMBIE_STATE_PREFIX = "Z"
_PS_TIMEOUT = 2.0
_ADOPT_POLL_INTERVAL = 0.05


def _state_is_zombie(pid: int) -> bool:
    """Return whether ``ps`` reports ``pid`` as a reaped-but-unwaited process.

    ``os.kill(pid, 0)`` succeeds for a zombie, so a signal probe alone would
    call a dead worker alive. ``ps`` is POSIX-portable and bounded by a
    timeout. A ``ps`` that cannot answer is read as "not a zombie"; the caller
    still never treats the process as finished work, so the error stays
    conservative.
    """
    try:
        completed = subprocess.run(
            ["ps", "-o", "stat=", "-p", str(pid)],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=_PS_TIMEOUT,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    state = completed.stdout.strip()
    return state.upper().startswith(_ZOMBIE_STATE_PREFIX)


def process_alive(pid: int) -> bool:
    """Return whether ``pid`` names a live, non-zombie process on this host.

    This is a read-only probe: signalling with ``0`` performs no delivery, and
    the ``ps`` call only reads state. A process this interpreter did not spawn
    can be observed but never waited on, which is why reconciliation polls.
    """
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        raise ValueError("pid must be a positive integer")
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        pass
    except OSError:
        return False
    return not _state_is_zombie(pid)


class RecursionRefusedError(RuntimeError):
    """A NEO worker attempted to create another NEO worker."""


class ProcessSupervisor:
    """Supervise one service's task subprocesses."""

    def __init__(
        self,
        service: TaskService,
        tokens: WorkerTokenStore,
        *,
        log_dir: Union[str, Path],
        alive_fn: Optional[Callable[[int], bool]] = None,
        poll_interval: float = _ADOPT_POLL_INTERVAL,
    ) -> None:
        self._service = service
        self._registry = service.registry
        self._tokens = tokens
        self._log_dir = Path(log_dir)
        self._lock = threading.RLock()
        self._processes = {}
        self._pending = {}
        self._watchers = {}
        self._adopted = {}
        self._stop = threading.Event()
        self._maintenance = None
        self._alive = alive_fn or process_alive
        self._poll_interval = max(0.01, float(poll_interval))

    # -- submission -----------------------------------------------------

    def submit(self, task_id: str, argv: list) -> dict:
        if (
            isinstance(argv, (str, bytes))
            or not isinstance(argv, (list, tuple))
            or not argv
            or any(not isinstance(part, str) or not part for part in argv)
        ):
            raise ValueError("worker command must be a non-empty argv array")
        argv = list(argv)
        if os.environ.get("NEO_WORKER") == "1":
            raise RecursionRefusedError("NEO workers cannot spawn nested NEO tasks")
        with self._lock:
            task = self._service.get_task(task_id)
            if task is None:
                raise KeyError(task_id)
            if task["state"] == "queued":
                self._pending[task_id] = argv
                self._ensure_maintenance()
                self.drain_pending()
                return self._service.get_task(task_id) or task
            if task["state"] != "starting":
                raise RuntimeError("task %s is not startable" % task_id)
            return self._spawn(task, argv)

    def _log_path(self, task_id: str) -> Path:
        safe = task_id if _SAFE_TASK_ID.fullmatch(task_id) else (
            "task-" + hashlib.sha256(task_id.encode("utf-8")).hexdigest()[:32]
        )
        return self._log_dir / ("%s.log" % safe)

    def _spawn(self, task: dict, argv: list) -> dict:
        task_id = task["id"]
        log_path = self._log_path(task_id)
        process = None
        pgid = None
        try:
            token = self._tokens.issue(task_id)
            self._log_dir.mkdir(parents=True, exist_ok=True)
            environment = os.environ.copy()
            environment.update(
                {
                    "NEO_WORKER": "1",
                    "NEO_TASK_ID": task_id,
                    "NEO_TASK_TOKEN": token,
                }
            )
            with log_path.open("ab", buffering=0) as log:
                process = subprocess.Popen(
                    argv,
                    shell=False,
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env=environment,
                    cwd=task.get("workdir") or None,
                    start_new_session=True,
                    close_fds=True,
                )
            now = time.time()
            pgid = os.getpgid(process.pid)
            self._record_process(task, process, pgid, log_path, now)
        except Exception as error:
            if process is not None:
                self._kill_group(pgid or process.pid)
                try:
                    process.wait(timeout=2)
                except (subprocess.TimeoutExpired, OSError):
                    pass
            return self._spawn_failed(task_id, log_path, error)

        with self._lock:
            self._processes[task_id] = process
        watcher = threading.Thread(
            target=self._watch,
            args=(task_id, process, log_path),
            name="neo-worker-%s" % task_id,
            daemon=True,
        )
        with self._lock:
            self._watchers[task_id] = watcher
        watcher.start()
        try:
            return self._service.transition(
                task_id, "running", expected_version=task["version"] + 1
            )
        except VersionConflictError:
            # Another writer may already have moved this logically-owned task.
            return self._service.get_task(task_id) or task
        except Exception:
            # The watcher already owns the child, so no process is orphaned.
            return self._service.get_task(task_id) or task

    def _record_process(
        self, task: dict, process: subprocess.Popen, pgid: int, log_path: Path, now: float
    ) -> None:
        connection = self._registry._connect()
        started = False
        try:
            connection.execute("BEGIN IMMEDIATE")
            started = True
            connection.execute(
                """
                INSERT INTO workers (
                    id, task_id, pid, pgid, state, started_at, heartbeat_at, log_path
                ) VALUES (?, ?, ?, ?, 'running', ?, ?, ?)
                """,
                (uuid4().hex, task["id"], process.pid, pgid, now, now, str(log_path)),
            )
            cursor = connection.execute(
                """
                UPDATE tasks
                SET pid = ?, pgid = ?, log_path = ?, heartbeat_at = ?,
                    version = version + 1, updated_at = ?
                WHERE id = ? AND version = ?
                """,
                (
                    process.pid,
                    pgid,
                    str(log_path),
                    now,
                    now,
                    task["id"],
                    task["version"],
                ),
            )
            if cursor.rowcount != 1:
                current = connection.execute(
                    "SELECT version FROM tasks WHERE id = ?", (task["id"],)
                ).fetchone()
                actual = current["version"] if current is not None else task["version"]
                raise VersionConflictError(task["id"], task["version"], actual)
            connection.execute("COMMIT")
        except Exception:
            if started:
                connection.execute("ROLLBACK")
            raise
        finally:
            connection.close()

    def _spawn_failed(self, task_id: str, log_path: Path, error: Exception) -> dict:
        """Best-effort failure record, then release the task's capacity slot."""
        now = time.time()
        try:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("a", encoding="utf-8") as log:
                log.write("spawn failed: %s\n" % error)
        except OSError:
            pass
        receipt_path = log_path.with_suffix(".receipt.json")
        try:
            receipt_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = receipt_path.with_suffix(".receipt.json.tmp")
            temporary.write_text(
                json.dumps(
                    {
                        "task_id": task_id,
                        "pid": None,
                        "pgid": None,
                        "exit_code": None,
                        "finished_at": now,
                        "error": "spawn_failed",
                    }
                )
            )
            os.replace(temporary, receipt_path)
        except OSError:
            pass
        try:
            connection = self._registry._connect()
            try:
                connection.execute("BEGIN IMMEDIATE")
                connection.execute(
                    """
                    INSERT INTO workers (
                        id, task_id, pid, pgid, state, started_at, heartbeat_at, log_path
                    )
                    SELECT ?, ?, NULL, NULL, 'failed', ?, ?, ?
                    WHERE NOT EXISTS (SELECT 1 FROM workers WHERE task_id = ?)
                    """,
                    (
                        uuid4().hex,
                        task_id,
                        now,
                        now,
                        str(log_path),
                        task_id,
                    ),
                )
                connection.execute("COMMIT")
            except Exception:
                connection.execute("ROLLBACK")
            finally:
                connection.close()
        except Exception:
            pass

        result = self._service.get_task(task_id)
        if result is not None and result["state"] in {"queued", "starting"}:
            try:
                result = self._service.transition(
                    task_id,
                    "failed",
                    expected_version=result["version"],
                    summary="worker failed to start: %s" % type(error).__name__,
                )
            except Exception:
                result = self._service.get_task(task_id) or result
        self.drain_pending()
        return result

    # -- supervision ----------------------------------------------------

    def _watch(self, task_id: str, process: subprocess.Popen, log_path: Path) -> None:
        try:
            exit_code = process.wait()
            with self._lock:
                task = self._service.get_task(task_id)
                worker = self.get_worker(task_id)
                if (
                    task is not None
                    and task["state"] == "cancelled"
                    and worker is not None
                    and worker["state"] == "cancelled"
                ):
                    return
                self._write_receipt(task_id, process, log_path, exit_code)
                if task is not None:
                    worker_state = self._worker_state(task["state"], exit_code)
                    self._update_worker(task_id, process.pid, worker_state)
                    if exit_code != 0 and task["state"] in _UNCLAIMED_EXIT_STATES:
                        self._transition_best_effort(
                            task,
                            "failed",
                            "worker exited with status %s" % exit_code,
                        )
                    elif exit_code == 0 and task["state"] in {
                        "starting",
                        "running",
                        "waiting",
                    }:
                        self._transition_best_effort(
                            task,
                            "failed",
                            "worker exited without reporting completion",
                        )
        except Exception:
            # A watcher must never die silently holding the capacity slot.
            try:
                task = self._service.get_task(task_id)
                if task is not None and task["state"] in _UNCLAIMED_EXIT_STATES:
                    self._transition_best_effort(
                        task, "failed", "worker supervision failed"
                    )
                self._update_worker(task_id, process.pid, "failed")
            except Exception:
                pass
        finally:
            with self._lock:
                if self._processes.get(task_id) is process:
                    self._processes.pop(task_id, None)
                if self._watchers.get(task_id) is threading.current_thread():
                    self._watchers.pop(task_id, None)
                self.drain_pending()

    @staticmethod
    def _worker_state(task_state: str, exit_code: int) -> str:
        if task_state == "cancelled":
            return "cancelled"
        if task_state == "completed":
            return "completed"
        if task_state == "failed":
            return "failed"
        return "exited" if exit_code == 0 else "failed"

    def _transition_best_effort(self, task: dict, target: str, summary: str) -> None:
        try:
            self._service.transition(
                task["id"], target, expected_version=task["version"], summary=summary
            )
        except (VersionConflictError, InvalidTransitionError, TerminalStateError):
            current = self._service.get_task(task["id"])
            if current is not None and current["state"] in _UNCLAIMED_EXIT_STATES:
                try:
                    self._service.transition(
                        task["id"],
                        target,
                        expected_version=current["version"],
                        summary=summary,
                    )
                except Exception:
                    pass
        except Exception:
            pass

    def _write_receipt(
        self,
        task_id: str,
        process: subprocess.Popen,
        log_path: Path,
        exit_code: int,
        *,
        term_sent: bool = False,
        kill_sent: bool = False,
    ) -> None:
        receipt_path = log_path.with_suffix(".receipt.json")
        existing = {}
        if receipt_path.exists():
            try:
                existing = json.loads(receipt_path.read_text())
            except (OSError, ValueError):
                existing = {}
        receipt = {
            "task_id": task_id,
            "pid": process.pid,
            "pgid": process.pid,
            "exit_code": exit_code,
            "finished_at": time.time(),
            "term_sent": bool(existing.get("term_sent")) or term_sent,
            "kill_sent": bool(existing.get("kill_sent")) or kill_sent,
        }
        receipt_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = receipt_path.with_suffix(".receipt.json.tmp")
        temporary.write_text(json.dumps(receipt))
        os.replace(temporary, receipt_path)

    def _update_worker(
        self, task_id: str, pid: Optional[int], state: str
    ) -> None:
        statement = "UPDATE workers SET state = ?, heartbeat_at = ? WHERE task_id = ?"
        parameters: list = [state, time.time(), task_id]
        if pid is not None:
            statement += " AND pid = ?"
            parameters.append(pid)
        connection = self._registry._connect()
        started = False
        try:
            connection.execute("BEGIN IMMEDIATE")
            started = True
            connection.execute(statement, parameters)
            connection.execute("COMMIT")
        except Exception:
            if started:
                connection.execute("ROLLBACK")
            raise
        finally:
            connection.close()

    def _update_worker_best_effort(
        self, task_id: str, pid: Optional[int], state: str
    ) -> None:
        """Worker bookkeeping must never undo a task outcome already recorded."""
        try:
            self._update_worker(task_id, pid, state)
        except (RegistryError, ValueError, TypeError, OSError):
            pass

    @staticmethod
    def _kill_group(pgid: int) -> None:
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass

    # -- restart recovery ------------------------------------------------

    def reconcile(self) -> dict:
        """Record an honest outcome for every row a previous run left behind.

        The pass is read-mostly and bounded by the number of unfinished rows. It
        creates no task, launches no process and signals no worker. A live PID
        is adopted by a polling monitor; a dead or missing one becomes ``failed``
        (or ``unknown`` when a cancellation cannot be confirmed). Completion is
        never inferred here: only the executor can complete a task.

        One corrupt row is reported in ``errors`` and skipped. Every other row
        is still reconciled in its own transaction, so no unrelated task is
        rolled back.
        """
        summary = {
            "scanned": 0,
            "adopted": [],
            "failed": [],
            "unknown": [],
            "queued": [],
            "watching": [],
            "errors": [],
        }
        with self._lock:
            try:
                rows = self._registry.unfinished_tasks()
            except Exception as error:  # a damaged registry must not stop startup
                summary["errors"].append(
                    {"task_id": None, "error": type(error).__name__}
                )
                return summary
            for row in rows:
                task = dict(row)
                if task.get("state") not in RECOVERABLE_STATES:
                    continue
                summary["scanned"] += 1
                try:
                    self._reconcile_row(task, summary)
                except (
                    RegistryError,
                    StateMachineError,
                    KeyError,
                    ValueError,
                    TypeError,
                    OSError,
                ) as error:
                    summary["errors"].append(
                        {
                            "task_id": task.get("id"),
                            "error": type(error).__name__,
                            "detail": str(error)[:200],
                        }
                    )
            try:
                self.drain_pending()
            except Exception:
                pass
        return summary

    def _reconcile_row(self, task: dict, summary: dict) -> None:
        task_id = task["id"]
        state = task["state"]
        if state in _PROCESSLESS_STATES:
            # A queued task has no process yet; it is not stale, just waiting.
            summary["queued"].append(task_id)
            return
        with self._lock:
            if task_id in self._watchers:
                summary["watching"].append(task_id)
                return
        log_path = task.get("log_path") or str(self._log_path(task_id))
        pid = self._recorded_pid(task)
        if pid is None:
            self._record_outcome(
                task_id,
                pid=None,
                receipt=None,
                summary=summary,
                reason="worker_missing",
                detail="no worker process was recorded for this task; "
                "it never reported completion",
            )
            return
        if self._alive(pid):
            if self._adopt(task_id, pid, log_path):
                summary["adopted"].append(task_id)
            else:
                summary["watching"].append(task_id)
            return
        receipt = self._read_receipt(task_id, log_path)
        reason, detail = self._exit_outcome(state, receipt)
        self._record_outcome(
            task_id,
            pid=pid,
            receipt=receipt,
            summary=summary,
            reason=reason,
            detail=detail,
        )

    @staticmethod
    def _recorded_pid(task: dict) -> Optional[int]:
        """Return the durable PID, or ``None`` when the row never had one."""
        pid = task.get("pid")
        if pid is None:
            return None
        if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
            raise ValueError(
                "task %s records a pid that is not a positive integer"
                % task.get("id")
            )
        return pid

    @staticmethod
    def _exit_outcome(state: str, receipt: Optional[dict]) -> tuple:
        """Map a finished process plus its receipt to an honest reason."""
        if receipt is None:
            if state == "cancelling":
                return (
                    "cancellation_unconfirmed",
                    "worker vanished before its cancellation could be confirmed",
                )
            return (
                "worker_lost",
                "worker process is gone and no receipt was recorded; "
                "it never reported completion",
            )
        exit_code = receipt.get("exit_code")
        if exit_code is None:
            return (
                "worker_lost",
                "worker receipt records no exit status; "
                "it never reported completion",
            )
        if exit_code == 0:
            return (
                "worker_exit_zero",
                "worker exited with status 0 without reporting completion",
            )
        return (
            "worker_exit_nonzero",
            "worker exited with status %s without reporting completion"
            % exit_code,
        )

    def _receipt_path(
        self, task_id: str, log_path: Union[str, Path, None]
    ) -> Path:
        path = Path(log_path) if log_path else self._log_path(task_id)
        return path.with_suffix(".receipt.json")

    def _read_receipt(
        self, task_id: str, log_path: Union[str, Path, None]
    ) -> Optional[dict]:
        """Read this task's receipt, refusing anything unreadable or foreign."""
        try:
            receipt = json.loads(self._receipt_path(task_id, log_path).read_text())
        except (OSError, ValueError):
            return None
        if not isinstance(receipt, dict):
            return None
        recorded_task = receipt.get("task_id")
        if recorded_task not in (None, task_id):
            return None
        exit_code = receipt.get("exit_code")
        if exit_code is not None and (
            isinstance(exit_code, bool) or not isinstance(exit_code, int)
        ):
            return None
        return receipt

    def _record_outcome(
        self,
        task_id: str,
        *,
        pid: Optional[int],
        receipt: Optional[dict],
        summary: dict,
        reason: str,
        detail: str,
    ) -> None:
        """Persist one recovery outcome with its audit payload."""
        current = self._service.get_task(task_id)
        if current is None or current["state"] in TERMINAL_STATES:
            return
        target = "unknown" if reason == "cancellation_unconfirmed" else "failed"
        resource_key = current.get("resource_key")
        try:
            updated = self._service.transition(
                task_id,
                target,
                expected_version=current["version"],
                summary=detail,
                event_extra={
                    "recovered": True,
                    "error_code": reason,
                    "recovery": {
                        "pid": pid,
                        "process_alive": False,
                        "receipt": receipt is not None,
                        "exit_code": receipt.get("exit_code") if receipt else None,
                        "reason": reason,
                        "resource_key": resource_key,
                        "resource_released": bool(resource_key),
                    },
                },
            )
        except VersionConflictError:
            # Another writer owns this row; its version wins.
            return
        except (InvalidTransitionError, TerminalStateError, KeyError):
            # An edge this recovery cannot take; leave the row reported as is.
            return
        try:
            # Error columns are metadata; the task's state stays terminal.
            updated = self._registry.update_task(
                task_id,
                expected_version=updated["version"],
                error_code=reason,
                error_detail=detail[:500],
            )
        except (RegistryError, KeyError, ValueError, TypeError):
            pass
        self._update_worker_best_effort(task_id, pid, target)
        summary[target].append(task_id)

    def _adopt(
        self, task_id: str, pid: int, log_path: Union[str, Path]
    ) -> bool:
        """Start one monitor for an inherited worker, or report it already held."""
        with self._lock:
            if task_id in self._watchers:
                return False
            stop = threading.Event()
            watcher = threading.Thread(
                target=self._monitor_adopted,
                args=(task_id, pid, log_path, stop),
                name="neo-adopt-%s" % task_id,
                daemon=True,
            )
            self._watchers[task_id] = watcher
            self._adopted[task_id] = stop
        watcher.start()
        return True

    def _monitor_adopted(
        self,
        task_id: str,
        pid: int,
        log_path: Union[str, Path],
        stop: threading.Event,
    ) -> None:
        """Poll an inherited worker until it exits; never signal, never wait."""
        try:
            while not stop.wait(self._poll_interval):
                try:
                    alive = self._alive(pid)
                except (OSError, ValueError):
                    alive = False
                if not alive:
                    break
            else:
                # Closing the viewer abandons the monitor, not the task.
                return
            receipt = self._read_receipt(task_id, log_path)
            task = self._service.get_task(task_id)
            state = task["state"] if task is not None else "running"
            reason, detail = self._exit_outcome(state, receipt)
            self._record_outcome(
                task_id,
                pid=pid,
                receipt=receipt,
                summary={"failed": [], "unknown": []},
                reason=reason,
                detail=detail,
            )
        except Exception:
            try:
                self._record_outcome(
                    task_id,
                    pid=pid,
                    receipt=None,
                    summary={"failed": [], "unknown": []},
                    reason="worker_lost",
                    detail="worker supervision failed after restart; "
                    "the task never reported completion",
                )
            except Exception:
                pass
        finally:
            with self._lock:
                if self._watchers.get(task_id) is threading.current_thread():
                    self._watchers.pop(task_id, None)
                self._adopted.pop(task_id, None)
                try:
                    self.drain_pending()
                except Exception:
                    pass

    # -- cancellation ---------------------------------------------------

    def cancel(self, task_id: str, *, grace: float = 1.0) -> dict:
        with self._lock:
            task = self._service.get_task(task_id)
            if task is None:
                raise KeyError(task_id)
            if task["state"] in TERMINAL_STATES:
                return task
            if task["state"] == "queued":
                self._pending.pop(task_id, None)
                return self._service.transition(
                    task_id, "cancelled", expected_version=task["version"]
                )
            process = self._processes.get(task_id)
            worker = self.get_worker(task_id)
            if process is None or worker is None:
                raise RuntimeError("active task has no supervised process")
            if task["state"] == "starting":
                task = self._service.transition(
                    task_id, "running", expected_version=task["version"]
                )
            if task["state"] == "running":
                task = self._service.transition(
                    task_id, "cancelling", expected_version=task["version"]
                )
            self._update_worker(task_id, process.pid, "cancelling")
            term_sent = False
            kill_sent = False
            if process.poll() is None:
                try:
                    os.killpg(worker["pgid"], signal.SIGTERM)
                    term_sent = True
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=grace)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(worker["pgid"], signal.SIGKILL)
                        kill_sent = True
                    except ProcessLookupError:
                        pass
                    process.wait()
            exit_code = process.wait()
            log_path = Path(worker["log_path"])
            self._write_receipt(
                task_id,
                process,
                log_path,
                exit_code,
                term_sent=term_sent,
                kill_sent=kill_sent,
            )
            current = self._service.get_task(task_id)
            if current is not None and current["state"] not in TERMINAL_STATES:
                current = self._service.transition(
                    task_id, "cancelled", expected_version=current["version"]
                )
            if current is not None:
                self._update_worker(task_id, process.pid, "cancelled")
            self._processes.pop(task_id, None)
            return current

    # -- queue ----------------------------------------------------------

    def _ensure_maintenance(self) -> None:
        thread = self._maintenance
        if thread is not None and thread.is_alive():
            return
        thread = threading.Thread(
            target=self._maintain,
            name="neo-queue-maintenance",
            daemon=True,
        )
        self._maintenance = thread
        thread.start()

    def _maintain(self) -> None:
        while not self._stop.wait(0.1):
            if not self._pending:
                continue
            try:
                self.drain_pending()
            except Exception:
                continue

    def drain_pending(self) -> list:
        """Start queued argv while registry capacity and resources permit."""
        started = []
        with self._lock:
            for task_id in list(self._pending):
                task = self._service.get_task(task_id)
                if task is None or task["state"] != "queued":
                    self._pending.pop(task_id, None)
                    continue
                try:
                    starting = self._service.transition(
                        task_id, "starting", expected_version=task["version"]
                    )
                except (CapacityExceededError, ResourceBusyError, VersionConflictError):
                    continue
                argv = self._pending.pop(task_id)
                started.append(self._spawn(starting, argv))
        return started

    def get_worker(self, task_id: str) -> Optional[dict]:
        connection = self._registry._connect()
        try:
            row = connection.execute(
                "SELECT * FROM workers WHERE task_id = ? ORDER BY rowid DESC LIMIT 1",
                (task_id,),
            ).fetchone()
            return dict(row) if row is not None else None
        finally:
            connection.close()

    def close(self) -> None:
        """Stop queue maintenance, adopted monitors and finished watchers.

        Detached workers are never killed. Adopted monitors stop polling; the
        task itself is left to the next restart, which adopts the same PID again.
        Watchers still waiting on a live child are left running; a process exit
        does not wait for the viewer.
        """
        self._stop.set()
        with self._lock:
            adopted = list(self._adopted.values())
        for stop in adopted:
            stop.set()
        thread = self._maintenance
        if thread is not None:
            thread.join(timeout=1.0)
        deadline = time.monotonic() + 2.0
        with self._lock:
            watchers = list(self._watchers.values())
        for watcher in watchers:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            watcher.join(timeout=remaining)


__all__ = ["ProcessSupervisor", "RecursionRefusedError", "process_alive"]
