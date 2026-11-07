"""SQLite-backed durable task registry for NEO."""

from __future__ import annotations

import sqlite3
import json
import hmac
from pathlib import Path
import time
from typing import Callable, Optional, Union
from uuid import uuid4

from state_machine import (
    TERMINAL_STATES,
    CompletionEvidence,
    Evidence,
    TaskState,
    UnknownStateError,
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


class RegistryError(RuntimeError):
    """Base class for a refused registry operation."""


class CapacityExceededError(RegistryError):
    """Starting this task would exceed the configured parallel limit."""


class ResourceBusyError(RegistryError):
    """The requested exclusive resource is already held by an active task."""


class ActiveTaskError(RegistryError):
    """An active task cannot be deleted out from under its worker."""

    def __init__(self, task_id: str, state: str) -> None:
        super().__init__("task %s is %s and cannot be deleted" % (task_id, state))
        self.task_id = task_id
        self.state = state


class InvalidCursorError(ValueError):
    """An event cursor was not a non-negative integer."""


class SchemaVersionError(RegistryError):
    """The database was written by a newer registry schema."""


class DuplicateTaskError(RegistryError):
    """Another active task already owns this conversation dedupe key."""


class TaskNotFoundError(RegistryError):
    """An operation referenced a task that does not exist."""

    def __init__(self, task_id: str) -> None:
        super().__init__("task %s does not exist" % task_id)
        self.task_id = task_id


class InvalidWorkerTokenError(RegistryError):
    """A worker credential is absent, stale, or does not match the task."""


ACTIVE_STATES = (
    "starting",
    "running",
    "waiting",
    "needs_approval",
    "needs_input",
    "cancelling",
    "unknown",
)
_ACTIVE_SQL = ", ".join("'%s'" % state for state in ACTIVE_STATES)
_SCHEMA_VERSION = 5


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
        pid INTEGER,
        pgid INTEGER,
        log_path TEXT,
        heartbeat_at REAL,
        created_at REAL NOT NULL,
        updated_at REAL NOT NULL,
        started_at REAL,
        finished_at REAL,
        version INTEGER NOT NULL DEFAULT 1
    )
    """,
    """
    CREATE UNIQUE INDEX IF NOT EXISTS tasks_active_dedupe_v2
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
    DROP TRIGGER IF EXISTS tasks_max_parallel_insert
    """,
    """
    DROP TRIGGER IF EXISTS tasks_max_parallel_update
    """,
    """
    CREATE TRIGGER tasks_max_parallel_insert
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
        ) >= COALESCE(
            (SELECT value FROM registry_settings WHERE key = 'max_parallel'), 4
        );
    END
    """,
    """
    CREATE TRIGGER tasks_max_parallel_update
    BEFORE UPDATE OF state ON tasks
    WHEN NEW.state IN (
        'starting', 'running', 'waiting', 'needs_approval',
        'needs_input', 'cancelling', 'unknown'
    )
    AND OLD.state NOT IN (
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
        ) >= COALESCE(
            (SELECT value FROM registry_settings WHERE key = 'max_parallel'), 4
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
    CREATE INDEX IF NOT EXISTS task_events_task_cursor
    ON task_events(task_id, id)
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
    """
    CREATE TABLE IF NOT EXISTS task_worker_tokens (
        task_id TEXT PRIMARY KEY,
        token_hash TEXT NOT NULL,
        issued_at REAL NOT NULL,
        FOREIGN KEY (task_id) REFERENCES tasks(id) ON DELETE CASCADE
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS task_presentations (
        task_id TEXT NOT NULL,
        version INTEGER NOT NULL,
        claimed_at REAL NOT NULL,
        presented_at REAL,
        path TEXT NOT NULL,
        PRIMARY KEY (task_id, version),
        FOREIGN KEY (task_id) REFERENCES tasks(id) ON DELETE CASCADE
    )
    """,
)


def _hermes_connection() -> sqlite3.Connection:
    """Resolve Hermes lazily so standalone tests do not need it installed."""
    from plugins.plugin_storage import plugin_db

    return plugin_db("neo")


def _translate_integrity_error(error: sqlite3.IntegrityError) -> RegistryError:
    """Convert SQLite backstops into errors the task service can act on."""
    message = str(error)
    if "max parallel" in message:
        return CapacityExceededError("maximum parallel task count reached")
    if "tasks.resource_key" in message:
        return ResourceBusyError("exclusive task resource is already in use")
    if "tasks.conversation_key, tasks.dedupe_key" in message:
        return DuplicateTaskError("an equivalent task is already active")
    return RegistryError("registry integrity check failed: %s" % message)


def _json_string_list(raw: object, field: str) -> tuple:
    """Decode a persisted JSON string list or fail with a domain error."""
    try:
        value = json.loads(str(raw or "[]"))
    except (TypeError, ValueError) as error:
        raise RegistryError("%s is not valid JSON: %s" % (field, error)) from error
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise RegistryError("%s must be a JSON list of strings" % field)
    return tuple(value)


def _coerce_target(value: Union[TaskState, str]) -> TaskState:
    """Return a TaskState while preserving the state-machine error type."""
    if isinstance(value, TaskState):
        return value
    if isinstance(value, str):
        try:
            return TaskState(value)
        except ValueError:
            raise UnknownStateError("unknown task target state: %r" % value) from None
    raise UnknownStateError(
        "task target state must be a state name, got %s" % type(value).__name__
    )


def _coerce_cursor(since: object) -> int:
    """Return a validated event cursor; ``bool`` is refused as an integer."""
    if isinstance(since, bool):
        raise InvalidCursorError("event cursor must be a non-negative integer")
    if isinstance(since, int):
        cursor_value = since
    elif isinstance(since, str) and since.isascii() and since.isdigit():
        cursor_value = int(since)
    else:
        raise InvalidCursorError("event cursor must be a non-negative integer")
    if cursor_value < 0:
        raise InvalidCursorError("event cursor must be a non-negative integer")
    return cursor_value


def _presentation_key(task_id: object, version: object) -> int:
    """Validate a task/version pair used by the presentation ledger."""
    if not isinstance(task_id, str) or not task_id.strip():
        raise ValueError("presentation task id must be a non-empty string")
    if isinstance(version, bool) or not isinstance(version, int) or version < 0:
        raise ValueError("presentation version must be a non-negative integer")
    return version


def _task_event_payload(
    task: dict, summary: Optional[str], extra: Optional[dict] = None
) -> dict:
    """Build the normalized public payload for one task event."""
    payload = {
        "task_id": task["id"],
        "state": task["state"],
        "version": task["version"],
    }
    if summary is not None:
        payload["summary"] = summary
    if extra:
        if set(extra).intersection(payload):
            raise ValueError("event extra fields cannot replace normalized fields")
        payload.update(extra)
    return payload


def _insert_event(
    connection: sqlite3.Connection,
    *,
    task_id: str,
    kind: str,
    payload: dict,
    at: Optional[float] = None,
) -> dict:
    """Insert one event on an open transaction; the caller owns the commit."""
    if not isinstance(kind, str) or not kind:
        raise ValueError("event kind must be a non-empty string")
    if not isinstance(payload, dict):
        raise ValueError("event payload must be a mapping")
    cursor = connection.execute(
        """
        INSERT INTO task_events (task_id, at, kind, payload_json)
        VALUES (?, ?, ?, ?)
        """,
        (task_id, time.time() if at is None else at, kind, json.dumps(payload)),
    )
    return dict(
        connection.execute(
            "SELECT * FROM task_events WHERE id = ?", (cursor.lastrowid,)
        ).fetchone()
    )


class Registry:
    """Own SQLite migrations and task persistence without sharing connections."""

    def __init__(
        self,
        *,
        path: Optional[Union[str, Path]] = None,
        connection_factory: Optional[ConnectionFactory] = None,
        max_parallel: Optional[int] = None,
    ) -> None:
        if path is not None and connection_factory is not None:
            raise ValueError("provide either path or connection_factory, not both")
        if max_parallel is not None and max_parallel < 1:
            raise ValueError("max_parallel must be at least one")
        self._requested_max_parallel = max_parallel
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
        connection = self._connect()
        try:
            row = connection.execute(
                "SELECT value FROM registry_settings WHERE key = 'max_parallel'"
            ).fetchone()
            self.max_parallel = int(row["value"]) if row is not None else 4
        finally:
            connection.close()

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
        started = False
        try:
            current_version = connection.execute("PRAGMA user_version").fetchone()[0]
            if current_version > _SCHEMA_VERSION:
                raise SchemaVersionError(
                    "registry schema %s is newer than supported version %s"
                    % (current_version, _SCHEMA_VERSION)
                )
            connection.execute("BEGIN IMMEDIATE")
            started = True
            if current_version < 2:
                connection.execute("DROP INDEX IF EXISTS tasks_active_dedupe")
            for statement in _SCHEMA:
                connection.execute(statement)
            task_columns = {
                row["name"]
                for row in connection.execute("PRAGMA table_info(tasks)").fetchall()
            }
            for column, declaration in (
                ("pid", "INTEGER"),
                ("pgid", "INTEGER"),
                ("log_path", "TEXT"),
                ("heartbeat_at", "REAL"),
            ):
                if column not in task_columns:
                    connection.execute(
                        "ALTER TABLE tasks ADD COLUMN %s %s" % (column, declaration)
                    )
            connection.execute(
                "INSERT OR IGNORE INTO registry_settings (key, value) "
                "VALUES ('max_parallel', 4)"
            )
            if self._requested_max_parallel is not None:
                connection.execute(
                    "UPDATE registry_settings SET value = ? "
                    "WHERE key = 'max_parallel'",
                    (self._requested_max_parallel,),
                )
            connection.execute("PRAGMA user_version = %d" % _SCHEMA_VERSION)
            connection.execute("COMMIT")
        except Exception:
            if started:
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
        event_kind: Optional[str] = None,
        event_summary: Optional[str] = None,
    ) -> dict:
        """Atomically deduplicate and reserve a resource, or queue the task.

        When ``event_kind`` is given, the event row is inserted in the same
        transaction, so a task can never exist without its creation event and a
        refused create leaves no event behind. A dedupe hit returns the existing
        task and creates nothing, so it emits no event.
        """
        if not isinstance(conversation_key, str) or not conversation_key.strip():
            raise ValueError("conversation_key must be a non-empty string")
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
            limit_row = connection.execute(
                "SELECT value FROM registry_settings WHERE key = 'max_parallel'"
            ).fetchone()
            max_parallel = int(limit_row["value"]) if limit_row is not None else 4
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
            if event_kind is not None:
                _insert_event(
                    connection,
                    task_id=row["id"],
                    kind=event_kind,
                    payload=_task_event_payload(dict(row), event_summary),
                    at=now,
                )
            connection.execute("COMMIT")
            return dict(row)
        except sqlite3.IntegrityError as error:
            connection.execute("ROLLBACK")
            raise _translate_integrity_error(error) from error
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
            "pid",
            "pgid",
            "log_path",
            "heartbeat_at",
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
        except sqlite3.IntegrityError as error:
            connection.execute("ROLLBACK")
            raise _translate_integrity_error(error) from error
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
        event_kind: Optional[str] = None,
        event_summary: Optional[str] = None,
        event_extra: Optional[dict] = None,
        worker_token_digest: Optional[str] = None,
    ) -> dict:
        """Validate and persist one optimistic, state-machine-checked transition.

        When ``event_kind`` is given, the event is inserted in the same
        transaction as the state change. A refused transition therefore emits
        nothing, and a failed event insert rolls the state change back.
        """
        connection = self._connect()
        if (
            not isinstance(expected_version, int)
            or isinstance(expected_version, bool)
            or expected_version < 0
        ):
            connection.close()
            raise ValueError("expected_version must be a non-negative integer")
        try:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM tasks WHERE id = ?", (task_id,)
            ).fetchone()
            if row is None:
                if worker_token_digest is not None:
                    raise InvalidWorkerTokenError(
                        "worker token rejected for task %s" % task_id
                    )
                raise KeyError(task_id)
            if worker_token_digest is not None:
                token_row = connection.execute(
                    "SELECT token_hash FROM task_worker_tokens WHERE task_id = ?",
                    (task_id,),
                ).fetchone()
                if token_row is None or not hmac.compare_digest(
                    token_row["token_hash"], worker_token_digest
                ):
                    raise InvalidWorkerTokenError(
                        "worker token rejected for task %s" % task_id
                    )
            if row["version"] != expected_version:
                raise VersionConflictError(task_id, expected_version, row["version"])

            target_state = _coerce_target(target)
            if row["state"] == "queued" and target_state == TaskState.STARTING:
                active = connection.execute(
                    "SELECT COUNT(*) AS count FROM tasks WHERE state IN (%s)"
                    % _ACTIVE_SQL
                ).fetchone()["count"]
                limit_row = connection.execute(
                    "SELECT value FROM registry_settings WHERE key = 'max_parallel'"
                ).fetchone()
                limit = int(limit_row["value"]) if limit_row is not None else 4
                if active >= limit:
                    raise CapacityExceededError(
                        "cannot start task: %s active tasks, limit %s"
                        % (active, limit)
                    )
                if row["resource_key"] is not None:
                    busy = connection.execute(
                        "SELECT 1 FROM tasks WHERE resource_key = ? "
                        "AND state IN (%s) LIMIT 1" % _ACTIVE_SQL,
                        (row["resource_key"],),
                    ).fetchone()
                    if busy is not None:
                        raise ResourceBusyError(
                            "resource %s is already in use" % row["resource_key"]
                        )

            expected_json = row["expected_json"]
            artifacts_json = row["artifacts_json"]
            checked_evidence = evidence
            if target_state == TaskState.COMPLETED and isinstance(
                evidence, CompletionEvidence
            ):
                stored_expected = _json_string_list(expected_json, "expected_json")
                combined_expected = tuple(
                    dict.fromkeys(
                        (*stored_expected, *tuple(evidence.expected_artifacts))
                    )
                )
                checked_evidence = CompletionEvidence(
                    executor_confirmed=evidence.executor_confirmed,
                    expected_artifacts=combined_expected,
                    verified_artifacts=evidence.verified_artifacts,
                )
                expected_json = json.dumps(list(combined_expected))
                artifacts_json = json.dumps(list(evidence.verified_artifacts))

            new_state = validate_transition(
                row["state"], target_state, evidence=checked_evidence
            )
            now = time.time()
            if new_state == TaskState.STARTING and row["started_at"] is None:
                started_at = now
            else:
                started_at = row["started_at"]
            finished_at = now if new_state in TERMINAL_STATES else row["finished_at"]
            cursor = connection.execute(
                """
                UPDATE tasks
                SET state = ?, started_at = ?, finished_at = ?,
                    expected_json = ?, artifacts_json = ?,
                    version = version + 1, updated_at = ?
                WHERE id = ? AND version = ?
                """,
                (
                    new_state.value,
                    started_at,
                    finished_at,
                    expected_json,
                    artifacts_json,
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
            if event_kind is not None:
                _insert_event(
                    connection,
                    task_id=task_id,
                    kind=event_kind,
                    payload=_task_event_payload(
                        dict(updated), event_summary, event_extra
                    ),
                    at=now,
                )
            connection.execute("COMMIT")
            return dict(updated)
        except sqlite3.IntegrityError as error:
            connection.execute("ROLLBACK")
            raise _translate_integrity_error(error) from error
        except Exception:
            connection.execute("ROLLBACK")
            raise
        finally:
            connection.close()

    def delete_task(self, task_id: str) -> bool:
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT state FROM tasks WHERE id = ?", (task_id,)
            ).fetchone()
            if row is None:
                connection.execute("COMMIT")
                return False
            if row["state"] in ACTIVE_STATES:
                raise ActiveTaskError(task_id, row["state"])
            connection.execute("DELETE FROM tasks WHERE id = ?", (task_id,))
            connection.execute("COMMIT")
            return True
        except Exception:
            connection.execute("ROLLBACK")
            raise
        finally:
            connection.close()

    def claim_presentation(self, task_id: str, version: int, path: str) -> bool:
        """Claim one ``(task_id, version)`` presentation, or report it taken.

        The claim is taken inside ``BEGIN IMMEDIATE`` so two processes cannot
        both win, and it is only taken after the caller has verified the
        artifact. A claimed row that never finishes can be released by the
        caller, which makes a retry possible without ever opening twice.
        """
        version = _presentation_key(task_id, version)
        if not isinstance(path, str) or not path:
            raise ValueError("presentation path must be a non-empty string")
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            exists = connection.execute(
                "SELECT 1 FROM tasks WHERE id = ?", (task_id,)
            ).fetchone()
            if exists is None:
                raise TaskNotFoundError(task_id)
            claimed = connection.execute(
                "SELECT 1 FROM task_presentations WHERE task_id = ? AND version = ?",
                (task_id, version),
            ).fetchone()
            if claimed is not None:
                connection.execute("COMMIT")
                return False
            connection.execute(
                """
                INSERT INTO task_presentations (task_id, version, claimed_at, path)
                VALUES (?, ?, ?, ?)
                """,
                (task_id, version, time.time(), path),
            )
            connection.execute("COMMIT")
            return True
        except TaskNotFoundError:
            connection.execute("ROLLBACK")
            raise
        except sqlite3.IntegrityError as error:
            connection.execute("ROLLBACK")
            if "FOREIGN KEY constraint failed" in str(error):
                raise TaskNotFoundError(task_id) from error
            raise RegistryError(
                "presentation claim failed integrity check: %s" % error
            ) from error
        except Exception:
            connection.execute("ROLLBACK")
            raise
        finally:
            connection.close()

    def finish_presentation(self, task_id: str, version: int) -> bool:
        """Mark a held claim as presented; an unclaimed key is left alone."""
        version = _presentation_key(task_id, version)
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            cursor = connection.execute(
                """
                UPDATE task_presentations SET presented_at = ?
                WHERE task_id = ? AND version = ? AND presented_at IS NULL
                """,
                (time.time(), task_id, version),
            )
            connection.execute("COMMIT")
            return cursor.rowcount == 1
        except sqlite3.IntegrityError as error:
            connection.execute("ROLLBACK")
            raise RegistryError(
                "presentation finish failed integrity check: %s" % error
            ) from error
        except Exception:
            connection.execute("ROLLBACK")
            raise
        finally:
            connection.close()

    def release_presentation(self, task_id: str, version: int) -> bool:
        """Release an unfinished claim so a failed open can be retried.

        A finished presentation is never released: that is what makes a
        successful open idempotent per task version.
        """
        version = _presentation_key(task_id, version)
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            cursor = connection.execute(
                """
                DELETE FROM task_presentations
                WHERE task_id = ? AND version = ? AND presented_at IS NULL
                """,
                (task_id, version),
            )
            connection.execute("COMMIT")
            return cursor.rowcount == 1
        except sqlite3.IntegrityError as error:
            connection.execute("ROLLBACK")
            raise RegistryError(
                "presentation release failed integrity check: %s" % error
            ) from error
        except Exception:
            connection.execute("ROLLBACK")
            raise
        finally:
            connection.close()

    def claim(self, task_id: str, version: int, path: str) -> bool:
        """Ledger port alias for :meth:`claim_presentation`."""
        return self.claim_presentation(task_id, version, path)

    def finish(self, task_id: str, version: int) -> bool:
        """Ledger port alias for :meth:`finish_presentation`."""
        return self.finish_presentation(task_id, version)

    def release(self, task_id: str, version: int) -> bool:
        """Ledger port alias for :meth:`release_presentation`."""
        return self.release_presentation(task_id, version)

    def append_event(self, task_id: str, kind: str, payload: dict) -> dict:
        """Durably append one task event in its own transaction."""
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            row = _insert_event(
                connection, task_id=task_id, kind=kind, payload=payload
            )
            connection.execute("COMMIT")
            return row
        except sqlite3.IntegrityError as error:
            connection.execute("ROLLBACK")
            if "FOREIGN KEY constraint failed" in str(error):
                raise TaskNotFoundError(task_id) from error
            raise RegistryError(
                "event insert failed integrity check: %s" % error
            ) from error
        except Exception:
            connection.execute("ROLLBACK")
            raise
        finally:
            connection.close()

    def events_since(self, since: int = 0, *, task_id: Optional[str] = None) -> list:
        """Return events strictly after the cursor in global insertion order."""
        cursor_value = _coerce_cursor(since)
        connection = self._connect()
        try:
            if task_id is None:
                rows = connection.execute(
                    "SELECT * FROM task_events WHERE id > ? ORDER BY id",
                    (cursor_value,),
                ).fetchall()
            else:
                rows = connection.execute(
                    "SELECT * FROM task_events "
                    "WHERE id > ? AND task_id = ? ORDER BY id",
                    (cursor_value, task_id),
                ).fetchall()
            return [dict(row) for row in rows]
        finally:
            connection.close()

    def events_for_conversation(
        self,
        conversation_key: str,
        since: int = 0,
        *,
        task_id: Optional[str] = None,
    ) -> list:
        """Return only events whose task belongs to ``conversation_key``."""
        if not isinstance(conversation_key, str) or not conversation_key:
            raise ValueError("conversation_key must be a non-empty string")
        cursor_value = _coerce_cursor(since)
        statement = (
            "SELECT task_events.* FROM task_events "
            "JOIN tasks ON tasks.id = task_events.task_id "
            "WHERE tasks.conversation_key = ? AND task_events.id > ?"
        )
        parameters: list = [conversation_key, cursor_value]
        if task_id is not None:
            statement += " AND task_events.task_id = ?"
            parameters.append(task_id)
        statement += " ORDER BY task_events.id"
        connection = self._connect()
        try:
            rows = connection.execute(statement, parameters).fetchall()
            return [dict(row) for row in rows]
        finally:
            connection.close()

    def store_worker_token(self, task_id: str, token_hash: str) -> None:
        """Store (or rotate to) the digest of a task's worker token."""
        if not isinstance(token_hash, str) or not token_hash:
            raise ValueError("token hash must be a non-empty string")
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            exists = connection.execute(
                "SELECT 1 FROM tasks WHERE id = ?", (task_id,)
            ).fetchone()
            if exists is None:
                raise TaskNotFoundError(task_id)
            connection.execute(
                """
                INSERT INTO task_worker_tokens (task_id, token_hash, issued_at)
                VALUES (?, ?, ?)
                ON CONFLICT(task_id) DO UPDATE SET
                    token_hash = excluded.token_hash,
                    issued_at = excluded.issued_at
                """,
                (task_id, token_hash, time.time()),
            )
            connection.execute("COMMIT")
        except Exception:
            connection.execute("ROLLBACK")
            raise
        finally:
            connection.close()

    def worker_token_hash(self, task_id: str) -> Optional[str]:
        """Return the stored digest for a task, or ``None`` when none is issued."""
        connection = self._connect()
        try:
            row = connection.execute(
                "SELECT token_hash FROM task_worker_tokens WHERE task_id = ?",
                (task_id,),
            ).fetchone()
            return row["token_hash"] if row is not None else None
        finally:
            connection.close()
