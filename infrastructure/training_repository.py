"""JSON persistence for final model training jobs."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

from domain.entities import ExperimentStatus
from domain.training import TrainingJob, TrainingMode


class JsonTrainingJobRepository:
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

    def save(self, job: TrainingJob) -> None:
        with self._lock:
            data = self._read()
            data[job.id] = job.to_dict()
            temporary = self.path.with_suffix(".tmp")
            temporary.write_text(json.dumps(data, indent=2), encoding="utf-8")
            temporary.replace(self.path)

    def get(self, job_id: str) -> TrainingJob | None:
        with self._lock:
            raw = self._read().get(job_id)
        return self._hydrate(raw) if raw else None

    def list(self) -> list[TrainingJob]:
        with self._lock:
            jobs = [self._hydrate(raw) for raw in self._read().values()]
        return sorted(jobs, key=lambda job: job.created_at, reverse=True)

    @staticmethod
    def _hydrate(raw: dict[str, Any]) -> TrainingJob:
        values = dict(raw)
        values["mode"] = TrainingMode(values["mode"])
        values["status"] = ExperimentStatus(values["status"])
        return TrainingJob(**values)
