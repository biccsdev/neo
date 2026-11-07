"""Detached subprocess supervision for NEO tasks.

The supervisor owns processes, not lifecycle truth. It persists receipts and
worker metadata, cancels by process group, and lets the task service decide
valid state changes. A detached worker outlives the viewer; restart
reconciliation is a later task and never guesses completion from a PID.
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
from typing import Optional, Union
from uuid import uuid4

from events import TaskService
from registry import CapacityExceededError, ResourceBusyError, VersionConflictError
from state_machine import InvalidTransitionError, TerminalStateError
from tools import WorkerTokenStore

TERMINAL_STATES = frozenset({"completed", "failed", "cancelled"})
_UNCLAIMED_EXIT_STATES = frozenset(
    {"queued", "starting", "running", "waiting", "cancelling", "unknown"}
)
_SAFE_TASK_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


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
    ) -> None:
        self._service = service
        self._registry = service.registry
        self._tokens = tokens
        self._log_dir = Path(log_dir)
        self._lock = threading.RLock()
        self._processes = {}
        self._pending = {}
        self._stop = threading.Event()
        self._maintenance = None

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

    def _update_worker(self, task_id: str, pid: int, state: str) -> None:
        connection = self._registry._connect()
        started = False
        try:
            connection.execute("BEGIN IMMEDIATE")
            started = True
            connection.execute(
                "UPDATE workers SET state = ?, heartbeat_at = ? "
                "WHERE task_id = ? AND pid = ?",
                (state, time.time(), task_id, pid),
            )
            connection.execute("COMMIT")
        except Exception:
            if started:
                connection.execute("ROLLBACK")
            raise
        finally:
            connection.close()

    @staticmethod
    def _kill_group(pgid: int) -> None:
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
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
        """Stop queue maintenance without killing detached workers."""
        self._stop.set()
        thread = self._maintenance
        if thread is not None:
            thread.join(timeout=1.0)


__all__ = ["ProcessSupervisor", "RecursionRefusedError"]
