"""Domain objects for full-budget model training jobs."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any
from uuid import uuid4

from domain.entities import ExperimentStatus, utc_now


class TrainingMode(str, Enum):
    NEW = "new"
    CONTINUE = "continue"


@dataclass
class TrainingJob:
    name: str
    mode: TrainingMode
    experiment_path: str
    experiment_id: str
    best_trial: int
    task: str
    model: str
    dataset: str
    epochs: int
    batch: int
    image_size: int
    device: str | int | None
    hyperparameters: dict[str, Any]
    source_weights: str | None = None
    id: str = field(default_factory=lambda: uuid4().hex)
    status: ExperimentStatus = ExperimentStatus.QUEUED
    metrics: dict[str, float] = field(default_factory=dict)
    run_directory: str | None = None
    error: str | None = None
    resumed: bool = False
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
