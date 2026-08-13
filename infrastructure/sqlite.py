"""SQLite schema, legacy JSON migration, and repository adapters.

Large model artifacts stay on disk. SQLite stores durable metadata and paths.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from pathlib import Path
from typing import Any, Callable, TypeVar

from domain.entities import Experiment
from domain.training import TrainingJob
from domain.validation import ValidationJob
from domain.ticket import Ticket, TicketType
from infrastructure.repository import JsonExperimentRepository
from infrastructure.training_repository import JsonTrainingJobRepository
from infrastructure.validation_repository import JsonValidationRepository

T = TypeVar("T")
SCHEMA_VERSION = 2
logger = logging.getLogger(__name__)


def initialize_database(data_dir: Path) -> Path:
    """Create the database, atomically importing legacy JSON on first use.

    Legacy files are deliberately left untouched as migration backups.
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    database = data_dir / "studio.db"
    if database.is_file():
        with _connect(database) as connection:
            _create_schema(connection)
            _verify(connection)
        return database

    temporary = data_dir / "studio.db.migrating"
    if temporary.exists():
        temporary.unlink()
    try:
        connection = sqlite3.connect(temporary, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = DELETE")
        connection.execute("PRAGMA busy_timeout = 5000")
        _create_schema(connection)
        _import_legacy_json(connection, data_dir)
        _verify(connection)
        connection.close()
        temporary.replace(database)
    except Exception:
        try:
            connection.close()
        except UnboundLocalError:
            pass
        if temporary.exists():
            temporary.unlink()
        raise
    return database


def _connect(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path, timeout=30)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA journal_mode = WAL")
    connection.execute("PRAGMA busy_timeout = 5000")
    return connection


def checkpoint_database(path: Path) -> bool:
    """Best-effort non-blocking WAL flush for external database viewers."""
    try:
        with _connect(path) as connection:
            busy, _, _ = connection.execute("PRAGMA wal_checkpoint(PASSIVE)").fetchone()
        return not bool(busy)
    except sqlite3.OperationalError as exc:
        logger.warning("Could not checkpoint SQLite WAL for %s: %s", path, exc)
        return False


def _create_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS schema_info (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS experiments (
            id TEXT PRIMARY KEY,
            config_json TEXT NOT NULL,
            status TEXT NOT NULL,
            best_trial INTEGER,
            error TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS experiment_trials (
            experiment_id TEXT NOT NULL,
            number INTEGER NOT NULL,
            payload_json TEXT NOT NULL,
            PRIMARY KEY (experiment_id, number),
            FOREIGN KEY (experiment_id) REFERENCES experiments(id) ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS idx_experiments_status ON experiments(status);
        CREATE INDEX IF NOT EXISTS idx_experiments_created ON experiments(created_at DESC);

        CREATE TABLE IF NOT EXISTS training_jobs (
            id TEXT PRIMARY KEY,
            status TEXT NOT NULL,
            name TEXT NOT NULL,
            experiment_id TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            payload_json TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_training_status ON training_jobs(status);
        CREATE INDEX IF NOT EXISTS idx_training_created ON training_jobs(created_at DESC);

        CREATE TABLE IF NOT EXISTS validation_jobs (
            id TEXT PRIMARY KEY,
            status TEXT NOT NULL,
            name TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            payload_json TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS validation_models (
            validation_job_id TEXT NOT NULL,
            position INTEGER NOT NULL,
            label TEXT NOT NULL,
            model_path TEXT NOT NULL,
            status TEXT NOT NULL,
            payload_json TEXT NOT NULL,
            PRIMARY KEY (validation_job_id, position),
            FOREIGN KEY (validation_job_id) REFERENCES validation_jobs(id) ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS idx_validation_status ON validation_jobs(status);
        CREATE INDEX IF NOT EXISTS idx_validation_created ON validation_jobs(created_at DESC);

        CREATE TABLE IF NOT EXISTS tickets (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            type TEXT NOT NULL CHECK (type IN ('feature', 'bug', 'misc')),
            message TEXT NOT NULL,
            page TEXT,
            status TEXT NOT NULL DEFAULT 'new',
            created_at TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_tickets_status ON tickets(status);
        CREATE INDEX IF NOT EXISTS idx_tickets_created ON tickets(created_at DESC);
        """
    )
    connection.execute(
        "INSERT OR REPLACE INTO schema_info(key, value) VALUES ('schema_version', ?)",
        (str(SCHEMA_VERSION),),
    )
    connection.commit()


def _read_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot migrate invalid JSON file {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"cannot migrate {path}: root value must be an object")
    return value


def _import_legacy_json(connection: sqlite3.Connection, data_dir: Path) -> None:
    sources = {
        "experiments": data_dir / "experiments.json",
        "training_jobs": data_dir / "training_jobs.json",
        "validation_jobs": data_dir / "validation_jobs.json",
    }
    experiment_values = _read_json_object(sources["experiments"])
    training_values = _read_json_object(sources["training_jobs"])
    validation_values = _read_json_object(sources["validation_jobs"])
    with connection:
        for raw in experiment_values.values():
            _save_experiment_raw(connection, raw)
        for raw in training_values.values():
            _save_training_raw(connection, raw)
        for raw in validation_values.values():
            _save_validation_raw(connection, raw)
        connection.execute(
            "INSERT OR REPLACE INTO schema_info(key, value) VALUES ('legacy_migration', ?)",
            (
                json.dumps(
                    {
                        "experiments": len(experiment_values),
                        "training_jobs": len(training_values),
                        "validation_jobs": len(validation_values),
                        "source_files": {name: str(path) for name, path in sources.items()},
                    },
                    sort_keys=True,
                ),
            ),
        )


def _verify(connection: sqlite3.Connection) -> None:
    integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
    if integrity != "ok":
        raise RuntimeError(f"SQLite integrity check failed: {integrity}")
    foreign_keys = connection.execute("PRAGMA foreign_key_check").fetchall()
    if foreign_keys:
        raise RuntimeError(f"SQLite foreign-key check failed: {foreign_keys}")


def _save_experiment_raw(connection: sqlite3.Connection, raw: dict[str, Any]) -> None:
    connection.execute(
        """INSERT INTO experiments(id, config_json, status, best_trial, error, created_at, updated_at)
           VALUES (?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(id) DO UPDATE SET config_json=excluded.config_json,
             status=excluded.status, best_trial=excluded.best_trial, error=excluded.error,
             created_at=excluded.created_at, updated_at=excluded.updated_at""",
        (
            raw["id"], json.dumps(raw["config"]), raw["status"], raw.get("best_trial"), raw.get("error"),
            raw["created_at"], raw["updated_at"],
        ),
    )
    numbers: list[int] = []
    for trial in raw.get("trials", []):
        number = int(trial["number"])
        numbers.append(number)
        connection.execute(
            """INSERT INTO experiment_trials(experiment_id, number, payload_json) VALUES (?, ?, ?)
               ON CONFLICT(experiment_id, number) DO UPDATE SET payload_json=excluded.payload_json""",
            (raw["id"], number, json.dumps(trial)),
        )
    if numbers:
        placeholders = ",".join("?" for _ in numbers)
        connection.execute(
            f"DELETE FROM experiment_trials WHERE experiment_id=? AND number NOT IN ({placeholders})",
            (raw["id"], *numbers),
        )
    else:
        connection.execute("DELETE FROM experiment_trials WHERE experiment_id=?", (raw["id"],))


def _save_training_raw(connection: sqlite3.Connection, raw: dict[str, Any]) -> None:
    connection.execute(
        """INSERT INTO training_jobs(id, status, name, experiment_id, created_at, updated_at, payload_json)
           VALUES (?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(id) DO UPDATE SET status=excluded.status, name=excluded.name,
             experiment_id=excluded.experiment_id, created_at=excluded.created_at,
             updated_at=excluded.updated_at, payload_json=excluded.payload_json""",
        (
            raw["id"], raw["status"], raw["name"], raw["experiment_id"], raw["created_at"], raw["updated_at"],
            json.dumps(raw),
        ),
    )


def _save_validation_raw(connection: sqlite3.Connection, raw: dict[str, Any]) -> None:
    parent = dict(raw)
    models = parent.pop("models", [])
    connection.execute(
        """INSERT INTO validation_jobs(id, status, name, created_at, updated_at, payload_json)
           VALUES (?, ?, ?, ?, ?, ?)
           ON CONFLICT(id) DO UPDATE SET status=excluded.status, name=excluded.name,
             created_at=excluded.created_at, updated_at=excluded.updated_at, payload_json=excluded.payload_json""",
        (raw["id"], raw["status"], raw["name"], raw["created_at"], raw["updated_at"], json.dumps(parent)),
    )
    for position, model in enumerate(models):
        connection.execute(
            """INSERT INTO validation_models(validation_job_id, position, label, model_path, status, payload_json)
               VALUES (?, ?, ?, ?, ?, ?)
               ON CONFLICT(validation_job_id, position) DO UPDATE SET label=excluded.label,
                 model_path=excluded.model_path, status=excluded.status, payload_json=excluded.payload_json""",
            (raw["id"], position, model["label"], model["model_path"], model["status"], json.dumps(model)),
        )
    connection.execute("DELETE FROM validation_models WHERE validation_job_id=? AND position>=?", (raw["id"], len(models)))


class _SqliteRepository:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.RLock()

    def _write(self, callback: Callable[[sqlite3.Connection], None]) -> None:
        with self._lock, _connect(self.path) as connection:
            callback(connection)
            connection.commit()


class SqliteExperimentRepository(_SqliteRepository):
    def save(self, experiment: Experiment) -> None:
        self._write(lambda connection: _save_experiment_raw(connection, experiment.to_dict()))

    def get(self, experiment_id: str) -> Experiment | None:
        with _connect(self.path) as connection:
            row = connection.execute("SELECT * FROM experiments WHERE id=?", (experiment_id,)).fetchone()
            if not row:
                return None
            raw = self._raw(connection, row)
        return JsonExperimentRepository._hydrate(raw)

    def list(self) -> list[Experiment]:
        with _connect(self.path) as connection:
            rows = connection.execute("SELECT * FROM experiments ORDER BY created_at DESC").fetchall()
            values = [JsonExperimentRepository._hydrate(self._raw(connection, row)) for row in rows]
        return values

    @staticmethod
    def _raw(connection: sqlite3.Connection, row: sqlite3.Row) -> dict[str, Any]:
        trials = connection.execute(
            "SELECT payload_json FROM experiment_trials WHERE experiment_id=? ORDER BY number", (row["id"],)
        ).fetchall()
        return {
            "id": row["id"], "config": json.loads(row["config_json"]), "status": row["status"],
            "trials": [json.loads(item["payload_json"]) for item in trials], "best_trial": row["best_trial"],
            "error": row["error"], "created_at": row["created_at"], "updated_at": row["updated_at"],
        }


class SqliteTrainingJobRepository(_SqliteRepository):
    def save(self, job: TrainingJob) -> None:
        self._write(lambda connection: _save_training_raw(connection, job.to_dict()))

    def get(self, job_id: str) -> TrainingJob | None:
        raw = self._get_raw(job_id)
        return JsonTrainingJobRepository._hydrate(raw) if raw else None

    def list(self) -> list[TrainingJob]:
        with _connect(self.path) as connection:
            rows = connection.execute("SELECT payload_json FROM training_jobs ORDER BY created_at DESC").fetchall()
        return [JsonTrainingJobRepository._hydrate(json.loads(row["payload_json"])) for row in rows]

    def _get_raw(self, job_id: str) -> dict[str, Any] | None:
        with _connect(self.path) as connection:
            row = connection.execute("SELECT payload_json FROM training_jobs WHERE id=?", (job_id,)).fetchone()
        return json.loads(row["payload_json"]) if row else None


class SqliteValidationRepository(_SqliteRepository):
    def save(self, job: ValidationJob) -> None:
        self._write(lambda connection: _save_validation_raw(connection, job.to_dict()))

    def get(self, job_id: str) -> ValidationJob | None:
        with _connect(self.path) as connection:
            row = connection.execute("SELECT payload_json FROM validation_jobs WHERE id=?", (job_id,)).fetchone()
            if not row:
                return None
            raw = json.loads(row["payload_json"])
            raw["models"] = self._models(connection, job_id)
        return JsonValidationRepository._hydrate(raw)

    def list(self) -> list[ValidationJob]:
        with _connect(self.path) as connection:
            rows = connection.execute("SELECT id, payload_json FROM validation_jobs ORDER BY created_at DESC").fetchall()
            values = []
            for row in rows:
                raw = json.loads(row["payload_json"])
                raw["models"] = self._models(connection, row["id"])
                values.append(JsonValidationRepository._hydrate(raw))
        return values

    @staticmethod
    def _models(connection: sqlite3.Connection, job_id: str) -> list[dict[str, Any]]:
        rows = connection.execute(
            "SELECT payload_json FROM validation_models WHERE validation_job_id=? ORDER BY position", (job_id,)
        ).fetchall()
        return [json.loads(row["payload_json"]) for row in rows]


class SqliteTicketRepository(_SqliteRepository):
    def save(self, ticket: Ticket) -> None:
        self._write(
            lambda connection: connection.execute(
                """INSERT INTO tickets(id, title, type, message, page, status, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    ticket.id,
                    ticket.title,
                    ticket.type.value,
                    ticket.message,
                    ticket.page,
                    ticket.status,
                    ticket.created_at,
                ),
            )
        )

    def get(self, ticket_id: str) -> Ticket | None:
        with _connect(self.path) as connection:
            row = connection.execute("SELECT * FROM tickets WHERE id=?", (ticket_id,)).fetchone()
        return self._hydrate(row) if row else None

    def list(self) -> list[Ticket]:
        with _connect(self.path) as connection:
            rows = connection.execute("SELECT * FROM tickets ORDER BY created_at DESC").fetchall()
        return [self._hydrate(row) for row in rows]

    @staticmethod
    def _hydrate(row: sqlite3.Row) -> Ticket:
        return Ticket(
            id=row["id"],
            title=row["title"],
            type=TicketType(row["type"]),
            message=row["message"],
            page=row["page"],
            status=row["status"],
            created_at=row["created_at"],
        )


def database_summary(path: Path) -> dict[str, Any]:
    """Return record counts and integrity information for documentation/support."""
    with _connect(path) as connection:
        return {
            "experiments": connection.execute("SELECT COUNT(*) FROM experiments").fetchone()[0],
            "experiment_trials": connection.execute("SELECT COUNT(*) FROM experiment_trials").fetchone()[0],
            "training_jobs": connection.execute("SELECT COUNT(*) FROM training_jobs").fetchone()[0],
            "validation_jobs": connection.execute("SELECT COUNT(*) FROM validation_jobs").fetchone()[0],
            "validation_models": connection.execute("SELECT COUNT(*) FROM validation_models").fetchone()[0],
            "tickets": connection.execute("SELECT COUNT(*) FROM tickets").fetchone()[0],
            "integrity_check": connection.execute("PRAGMA integrity_check").fetchone()[0],
            "foreign_key_violations": len(connection.execute("PRAGMA foreign_key_check").fetchall()),
        }
