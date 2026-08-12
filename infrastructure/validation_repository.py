"""JSON persistence for model-comparison validation jobs."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

from domain.entities import ExperimentStatus
from domain.validation import ModelValidationResult, ValidationJob


class JsonValidationRepository:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    def _read(self) -> dict[str, Any]:
        if not self.path.is_file():
            return {}
        try:
            value = json.loads(self.path.read_text(encoding="utf-8"))
            return value if isinstance(value, dict) else {}
        except (OSError, json.JSONDecodeError):
            return {}

    def save(self, job: ValidationJob) -> None:
        with self._lock:
            values = self._read()
            values[job.id] = job.to_dict()
            temporary = self.path.with_suffix(".tmp")
            temporary.write_text(json.dumps(values, indent=2), encoding="utf-8")
            temporary.replace(self.path)

    def get(self, job_id: str) -> ValidationJob | None:
        with self._lock:
            raw = self._read().get(job_id)
        return self._hydrate(raw) if raw else None

    def list(self) -> list[ValidationJob]:
        with self._lock:
            jobs = [self._hydrate(raw) for raw in self._read().values()]
        return sorted(jobs, key=lambda item: item.created_at, reverse=True)

    @staticmethod
    def _hydrate(raw: dict[str, Any]) -> ValidationJob:
        values = dict(raw)
        values["status"] = ExperimentStatus(values["status"])
        values["models"] = [ModelValidationResult(**model) for model in values.get("models", [])]
        return ValidationJob(**values)
