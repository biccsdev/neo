"""SQLite-backed durable task registry for NEO."""

from __future__ import annotations

import sqlite3
import json
from pathlib import Path
import time
from typing import Callable, Optional, Union
from uuid import uuid4

from state_machine import (
    TERMINAL_STATES,
    Evidence,
    TaskState,
    transition as validate_transition,
)

ConnectionFactory = Callable[[], sqlite3.Connection]


class VersionConflictError(RuntimeError):
    """A caller tried to update a task using a stale version."""

    def __init__(self, task_id: str, expected_version: int, actual_version: int) -> None:
        super().__init__(
            "task %s changed: expected version %s, actual version %s"
            % (task_id, expected_version, actual_version)
        )
        self.task_id = task_id
        self.expected_version = expected_version
        self.actual_version = actual_version


_SCHEMA = (
    """
    CREATE TABLE IF NOT EXISTS tasks (
        id TEXT PRIMARY KEY,
        parent_id TEXT,
        conversation_key TEXT,
        kind TEXT NOT NULL,
        intent TEXT NOT NULL,
        dedupe_key TEXT,
        state TEXT NOT NULL,
        capability_id TEXT,
        executor TEXT,
        executor_session_id TEXT,
        workdir TEXT,
        resource_key TEXT,
        expected_json TEXT NOT NULL DEFAULT '[]',
        artifacts_json TEXT NOT NULL DEFAULT '[]',
        preview_path TEXT,
        preview_rev INTEGER NOT NULL DEFAULT 0,
        approval_json TEXT,
        result_summary TEXT,
        error_code TEXT,
        error_detail TEXT,
        created_at REAL NOT NULL,
        updated_at REAL NOT NULL,
        started_at REAL,
        finished_at REAL,
        version INTEGER NOT NULL DEFAULT 1
    )
    """,
    """
    DROP INDEX IF EXISTS tasks_active_dedupe
    """,
    """
    CREATE UNIQUE INDEX tasks_active_dedupe
    ON tasks(conversation_key, dedupe_key)
    WHERE dedupe_key IS NOT NULL
      AND state IN (
          'queued', 'starting', 'running', 'waiting', 'needs_approval',
          'needs_input', 'cancelling', 'unknown'
      )
    """,
    """
    CREATE UNIQUE INDEX IF NOT EXISTS tasks_resource_lock
    ON tasks(resource_key)
    WHERE resource_key IS NOT NULL
      AND state IN (
          'starting', 'running', 'waiting', 'needs_approval',
          'needs_input', 'cancelling', 'unknown'
      )
    """,
    """
    CREATE TABLE IF NOT EXISTS registry_settings (
        key TEXT PRIMARY KEY,
        value INTEGER NOT NULL
    )
    """,
    """
    CREATE TRIGGER IF NOT EXISTS tasks_max_parallel_insert
    BEFORE INSERT ON tasks
    WHEN NEW.state IN (
        'starting', 'running', 'waiting', 'needs_approval',
        'needs_input', 'cancelling', 'unknown'
    )
    BEGIN
        SELECT RAISE(ABORT, 'max parallel tasks exceeded')
        WHERE (
            SELECT COUNT(*) FROM tasks
            WHERE state IN (
                'starting', 'running', 'waiting', 'needs_approval',
                'needs_input', 'cancelling', 'unknown'
            )
        ) >= (
            SELECT value FROM registry_settings WHERE key = 'max_parallel'
        );
    END
    """,
    """
    CREATE TABLE IF NOT EXISTS task_events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        task_id TEXT NOT NULL,
        at REAL NOT NULL,
        kind TEXT NOT NULL,
        payload_json TEXT NOT NULL DEFAULT '{}',
        delivered INTEGER NOT NULL DEFAULT 0,
        FOREIGN KEY (task_id) REFERENCES tasks(id) ON DELETE CASCADE
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS workers (
        id TEXT PRIMARY KEY,
        task_id TEXT NOT NULL,
        pid INTEGER,
        pgid INTEGER,
        state TEXT NOT NULL,
        started_at REAL NOT NULL,
        heartbeat_at REAL,
        log_path TEXT,
        FOREIGN KEY (task_id) REFERENCES tasks(id) ON DELETE CASCADE
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS capability_consents (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        source TEXT NOT NULL,
        pinned_sha TEXT NOT NULL,
        disclosure_hash TEXT NOT NULL,
        granted_at REAL NOT NULL
    )
    """,
)


def _hermes_connection() -> sqlite3.Connection:
    """Resolve Hermes lazily so standalone tests do not need it installed."""
    from plugins.plugin_storage import plugin_db

    return plugin_db("neo")


class Registry:
    """Own SQLite migrations and task persistence without sharing connections."""

    def __init__(
        self,
        *,
        path: Optional[Union[str, Path]] = None,
        connection_factory: Optional[ConnectionFactory] = None,
        max_parallel: int = 4,
    ) -> None:
        if path is not None and connection_factory is not None:
            raise ValueError("provide either path or connection_factory, not both")
        if max_parallel < 1:
            raise ValueError("max_parallel must be at least one")
        self.max_parallel = max_parallel
        if connection_factory is not None:
            self._connection_factory = connection_factory
        elif path is not None:
            db_path = Path(path)
            db_path.parent.mkdir(parents=True, exist_ok=True)

            def connect() -> sqlite3.Connection:
                return sqlite3.connect(
                    db_path,
                    timeout=5.0,
                    isolation_level=None,
                    check_same_thread=False,
                )

            self._connection_factory = connect
        else:
            self._connection_factory = _hermes_connection
        self.migrate()

    def _connect(self) -> sqlite3.Connection:
        connection = self._connection_factory()
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout = 5000")
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        return connection

    def migrate(self) -> None:
        """Apply the current schema repeatedly without replacing existing data."""
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            for statement in _SCHEMA:
                connection.execute(statement)
            connection.execute(
                "INSERT OR REPLACE INTO registry_settings (key, value) "
                "VALUES ('max_parallel', ?)",
                (self.max_parallel,),
            )
            connection.execute("PRAGMA user_version = 1")
            connection.execute("COMMIT")
        except Exception:
            connection.execute("ROLLBACK")
            raise
        finally:
            connection.close()

    def create_task(
        self,
        *,
        conversation_key: str,
        kind: str,
        intent: str,
        workdir: Optional[str] = None,
        task_id: Optional[str] = None,
        dedupe_key: Optional[str] = None,
        resource_key: Optional[str] = None,
    ) -> dict:
        """Atomically deduplicate and reserve a resource, or queue the task."""
        identifier = task_id or uuid4().hex
        now = time.time()
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            if dedupe_key is not None:
                existing = connection.execute(
                    """
                    SELECT * FROM tasks
                    WHERE conversation_key = ? AND dedupe_key = ?
                      AND state IN (
                          'queued', 'starting', 'running', 'waiting',
                          'needs_approval', 'needs_input', 'cancelling', 'unknown'
                      )
                    """,
                    (conversation_key, dedupe_key),
                ).fetchone()
                if existing is not None:
                    connection.execute("COMMIT")
                    return dict(existing)
            state = "starting"
            executing = connection.execute(
                """
                SELECT COUNT(*) AS count FROM tasks
                WHERE state IN (
                    'starting', 'running', 'waiting', 'needs_approval',
                    'needs_input', 'cancelling', 'unknown'
                )
                """
            ).fetchone()["count"]
            max_parallel = connection.execute(
                "SELECT value FROM registry_settings WHERE key = 'max_parallel'"
            ).fetchone()["value"]
            if executing >= max_parallel:
                state = "queued"
            if resource_key is not None and state == "starting":
                locked = connection.execute(
                    """
                    SELECT 1 FROM tasks
                    WHERE resource_key = ?
                      AND state IN (
                          'starting', 'running', 'waiting', 'needs_approval',
                          'needs_input', 'cancelling', 'unknown'
                      )
                    """,
                    (resource_key,),
                ).fetchone()
                if locked is not None:
                    state = "queued"
            connection.execute(
                """
                INSERT INTO tasks (
                    id, conversation_key, kind, intent, dedupe_key, state, workdir,
                    resource_key, created_at, updated_at, started_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    identifier,
                    conversation_key,
                    kind,
                    intent,
                    dedupe_key,
                    state,
                    workdir,
                    resource_key,
                    now,
                    now,
                    now if state == "starting" else None,
                ),
            )
            row = connection.execute(
                "SELECT * FROM tasks WHERE id = ?", (identifier,)
            ).fetchone()
            connection.execute("COMMIT")
            return dict(row)
        except Exception:
            connection.execute("ROLLBACK")
            raise
        finally:
            connection.close()

    def get_task(self, task_id: str) -> Optional[dict]:
        connection = self._connect()
        try:
            row = connection.execute(
                "SELECT * FROM tasks WHERE id = ?", (task_id,)
            ).fetchone()
            return dict(row) if row is not None else None
        finally:
            connection.close()

    def list_tasks(self) -> list:
        connection = self._connect()
        try:
            rows = connection.execute(
                "SELECT * FROM tasks ORDER BY created_at, id"
            ).fetchall()
            return [dict(row) for row in rows]
        finally:
            connection.close()

    def update_task(
        self, task_id: str, *, expected_version: int, **changes: object
    ) -> dict:
        allowed = {
            "parent_id",
            "conversation_key",
            "kind",
            "intent",
            "capability_id",
            "executor",
            "executor_session_id",
            "workdir",
            "resource_key",
            "expected_json",
            "artifacts_json",
            "preview_path",
            "preview_rev",
            "approval_json",
            "result_summary",
            "error_code",
            "error_detail",
            "started_at",
            "finished_at",
        }
        if not changes or not set(changes).issubset(allowed):
            raise ValueError("update_task requires one or more mutable task fields")
        assignments = ", ".join("%s = ?" % field for field in sorted(changes))
        values = [changes[field] for field in sorted(changes)]
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            cursor = connection.execute(
                "UPDATE tasks SET %s, version = version + 1, updated_at = ? "
                "WHERE id = ? AND version = ?" % assignments,
                [*values, time.time(), task_id, expected_version],
            )
            if cursor.rowcount != 1:
                current = connection.execute(
                    "SELECT version FROM tasks WHERE id = ?", (task_id,)
                ).fetchone()
                if current is None:
                    raise KeyError(task_id)
                raise VersionConflictError(
                    task_id, expected_version, current["version"]
                )
            row = connection.execute(
                "SELECT * FROM tasks WHERE id = ?", (task_id,)
            ).fetchone()
            connection.execute("COMMIT")
            return dict(row)
        except Exception:
            connection.execute("ROLLBACK")
            raise
        finally:
            connection.close()

    def transition_task(
        self,
        task_id: str,
        target: Union[TaskState, str],
        *,
        expected_version: int,
        evidence: Optional[Evidence] = None,
    ) -> dict:
        """Validate and persist one optimistic, state-machine-checked transition."""
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM tasks WHERE id = ?", (task_id,)
            ).fetchone()
            if row is None:
                raise KeyError(task_id)
            if row["version"] != expected_version:
                raise VersionConflictError(task_id, expected_version, row["version"])

            new_state = validate_transition(row["state"], target, evidence=evidence)
            now = time.time()
            started_at = row["started_at"] if row["started_at"] is not None else now
            finished_at = now if new_state in TERMINAL_STATES else row["finished_at"]
            cursor = connection.execute(
                """
                UPDATE tasks
                SET state = ?, started_at = ?, finished_at = ?,
                    version = version + 1, updated_at = ?
                WHERE id = ? AND version = ?
                """,
                (
                    new_state.value,
                    started_at,
                    finished_at,
                    now,
                    task_id,
                    expected_version,
                ),
            )
            if cursor.rowcount != 1:
                current = connection.execute(
                    "SELECT version FROM tasks WHERE id = ?", (task_id,)
                ).fetchone()
                actual = current["version"] if current is not None else expected_version
                raise VersionConflictError(task_id, expected_version, actual)
            updated = connection.execute(
                "SELECT * FROM tasks WHERE id = ?", (task_id,)
            ).fetchone()
            connection.execute("COMMIT")
            return dict(updated)
        except Exception:
            connection.execute("ROLLBACK")
            raise
        finally:
            connection.close()

    def delete_task(self, task_id: str) -> bool:
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            cursor = connection.execute("DELETE FROM tasks WHERE id = ?", (task_id,))
            connection.execute("COMMIT")
            return cursor.rowcount == 1
        except Exception:
            connection.execute("ROLLBACK")
            raise
        finally:
            connection.close()

    def append_event(self, task_id: str, kind: str, payload: dict) -> dict:
        """Durably append one task event in its own transaction."""
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            cursor = connection.execute(
                """
                INSERT INTO task_events (task_id, at, kind, payload_json)
                VALUES (?, ?, ?, ?)
                """,
                (task_id, time.time(), kind, json.dumps(payload)),
            )
            row = connection.execute(
                "SELECT * FROM task_events WHERE id = ?", (cursor.lastrowid,)
            ).fetchone()
            connection.execute("COMMIT")
            return dict(row)
        except Exception:
            connection.execute("ROLLBACK")
            raise
        finally:
            connection.close()

    def events_since(self, since: int = 0, *, task_id: Optional[str] = None) -> list:
        """Return events strictly after the cursor in global insertion order."""
        connection = self._connect()
        try:
            if task_id is None:
                rows = connection.execute(
                    "SELECT * FROM task_events WHERE id > ? ORDER BY id", (since,)
                ).fetchall()
            else:
                rows = connection.execute(
                    "SELECT * FROM task_events "
                    "WHERE id > ? AND task_id = ? ORDER BY id",
                    (since, task_id),
                ).fetchall()
            return [dict(row) for row in rows]
        finally:
            connection.close()
