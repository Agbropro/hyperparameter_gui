"""Small, thread-safe JSON experiment repository."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

from domain.entities import (
    Experiment,
    ExperimentConfig,
    ExperimentStatus,
    Range,
    SearchSpace,
    TaskType,
    TrialResult,
)


class JsonExperimentRepository:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    def _read(self) -> dict[str, Any]:
        if not self.path.exists():
            return {}
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}

    def save(self, experiment: Experiment) -> None:
        with self._lock:
            data = self._read()
            data[experiment.id] = experiment.to_dict()
            temporary = self.path.with_suffix(".tmp")
            temporary.write_text(json.dumps(data, indent=2), encoding="utf-8")
            temporary.replace(self.path)

    def get(self, experiment_id: str) -> Experiment | None:
        with self._lock:
            raw = self._read().get(experiment_id)
        return self._hydrate(raw) if raw else None

    def list(self) -> list[Experiment]:
        with self._lock:
            values = self._read().values()
            experiments = [self._hydrate(item) for item in values]
        return sorted(experiments, key=lambda item: item.created_at, reverse=True)

    @staticmethod
    def _hydrate(raw: dict[str, Any]) -> Experiment:
        config_raw = raw["config"]
        space_raw = config_raw.get("search_space", {})
        space = SearchSpace(
            ranges={name: Range(**value) for name, value in space_raw.get("ranges", {}).items()},
            choices=space_raw.get("choices", {}),
        )
        config = ExperimentConfig(
            name=config_raw["name"],
            task=TaskType(config_raw["task"]),
            model=config_raw["model"],
            dataset=config_raw["dataset"],
            trials=config_raw["trials"],
            epochs=config_raw["epochs"],
            metrics=config_raw["metrics"],
            search_space=space,
            device=config_raw.get("device"),
            image_size=config_raw.get("image_size", 640),
            seed=config_raw.get("seed", 42),
        )
        return Experiment(
            id=raw["id"],
            config=config,
            status=ExperimentStatus(raw["status"]),
            trials=[TrialResult(**trial) for trial in raw.get("trials", [])],
            best_trial=raw.get("best_trial"),
            error=raw.get("error"),
            created_at=raw["created_at"],
            updated_at=raw["updated_at"],
        )
