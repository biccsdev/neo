"""Detached subprocess supervision for NEO tasks."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import threading
import time
from pathlib import Path
from typing import Union
from uuid import uuid4

from events import TaskService
from registry import CapacityExceededError, ResourceBusyError, VersionConflictError
from tools import WorkerTokenStore


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
                self._pending[task_id] = list(argv)
                return task
            if task["state"] != "starting":
                raise RuntimeError("task %s is not startable" % task_id)
            return self._spawn(task, argv)

    def _spawn(self, task: dict, argv: list) -> dict:
        task_id = task["id"]
        token = self._tokens.issue(task_id)
        self._log_dir.mkdir(parents=True, exist_ok=True)
        log_path = self._log_dir / ("%s.log" % task_id)
        environment = os.environ.copy()
        environment.update(
            {"NEO_WORKER": "1", "NEO_TASK_ID": task_id, "NEO_TASK_TOKEN": token}
        )
        with log_path.open("ab", buffering=0) as log:
            try:
                process = subprocess.Popen(
                    argv,
                    shell=False,
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env=environment,
                    start_new_session=True,
                    close_fds=True,
                )
            except OSError as error:
                return self._spawn_failed(task, log_path, error)
        now = time.time()
        worker_id = uuid4().hex
        pgid = os.getpgid(process.pid)
        connection = self._registry._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """
                INSERT INTO workers (
                    id, task_id, pid, pgid, state, started_at, heartbeat_at, log_path
                ) VALUES (?, ?, ?, ?, 'running', ?, ?, ?)
                """,
                (
                    worker_id,
                    task_id,
                    process.pid,
                    pgid,
                    now,
                    now,
                    str(log_path),
                ),
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
                    task_id,
                    task["version"],
                ),
            )
            if cursor.rowcount != 1:
                current = connection.execute(
                    "SELECT version FROM tasks WHERE id = ?", (task_id,)
                ).fetchone()
                actual = current["version"] if current is not None else task["version"]
                raise VersionConflictError(task_id, task["version"], actual)
            connection.execute("COMMIT")
        except Exception:
            connection.execute("ROLLBACK")
            try:
                os.killpg(pgid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
            raise
        finally:
            connection.close()
        self._processes[task_id] = process
        try:
            updated = self._service.transition(
                task_id, "running", expected_version=task["version"] + 1
            )
        except VersionConflictError:
            updated = self._service.get_task(task_id)
        watcher = threading.Thread(
            target=self._watch,
            args=(task_id, process, log_path),
            name="neo-worker-%s" % task_id,
            daemon=True,
        )
        watcher.start()
        return updated

    def _spawn_failed(self, task: dict, log_path: Path, error: OSError) -> dict:
        task_id = task["id"]
        with log_path.open("a") as log:
            log.write("spawn failed: %s\n" % error)
        now = time.time()
        connection = self._registry._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """
                INSERT INTO workers (
                    id, task_id, pid, pgid, state, started_at, heartbeat_at, log_path
                ) VALUES (?, ?, NULL, NULL, 'failed', ?, ?, ?)
                """,
                (uuid4().hex, task_id, now, now, str(log_path)),
            )
            connection.execute("COMMIT")
        except Exception:
            connection.execute("ROLLBACK")
            raise
        finally:
            connection.close()
        receipt_path = log_path.with_suffix(".receipt.json")
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
        return self._service.transition(
            task_id, "failed", expected_version=task["version"], summary="spawn failed"
        )

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
        temporary = receipt_path.with_suffix(".receipt.json.tmp")
        temporary.write_text(json.dumps(receipt))
        os.replace(temporary, receipt_path)

    def _update_worker(self, task_id: str, pid: int, state: str) -> None:
        connection = self._registry._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                "UPDATE workers SET state = ?, heartbeat_at = ? "
                "WHERE task_id = ? AND pid = ?",
                (state, time.time(), task_id, pid),
            )
            connection.execute("COMMIT")
        except Exception:
            connection.execute("ROLLBACK")
            raise
        finally:
            connection.close()

    def _watch(self, task_id: str, process: subprocess.Popen, log_path: Path) -> None:
        exit_code = process.wait()
        with self._lock:
            self._write_receipt(task_id, process, log_path, exit_code)
            task = self._service.get_task(task_id)
            state = "cancelled" if task and task["state"] == "cancelled" else (
                "exited" if exit_code == 0 else "failed"
            )
            self._update_worker(task_id, process.pid, state)
            if exit_code != 0 and task and task["state"] in {
                "starting",
                "running",
                "waiting",
                "unknown",
            }:
                self._service.transition(
                    task_id,
                    "failed",
                    expected_version=task["version"],
                    summary="worker exited with status %s" % exit_code,
                )
        self.drain_pending()

    def cancel(self, task_id: str, *, grace: float = 1.0) -> dict:
        with self._lock:
            task = self._service.get_task(task_id)
            if task is None:
                raise KeyError(task_id)
            if task["state"] == "cancelled":
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
            self._update_worker(task_id, process.pid, "cancelled")
            current = self._service.get_task(task_id)
            if current["state"] != "cancelled":
                current = self._service.transition(
                    task_id, "cancelled", expected_version=current["version"]
                )
            self.drain_pending()
            return current

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
                except (CapacityExceededError, ResourceBusyError):
                    continue
                argv = self._pending.pop(task_id)
                started.append(self._spawn(starting, argv))
        return started

    def get_worker(self, task_id: str):
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
        """Release supervisor resources without killing detached workers."""


__all__ = ["ProcessSupervisor", "RecursionRefusedError"]
