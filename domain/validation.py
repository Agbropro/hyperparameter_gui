"""Domain objects for comparing model checkpoints on a held-out split."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any
from uuid import uuid4

from domain.entities import ExperimentStatus, utc_now


@dataclass
class ModelValidationResult:
    label: str
    model_path: str
    status: str = "queued"
    metrics: dict[str, float] = field(default_factory=dict)
    per_class: list[dict[str, Any]] = field(default_factory=list)
    run_directory: str | None = None
    duration_seconds: float | None = None
    error: str | None = None


@dataclass
class ValidationJob:
    name: str
    dataset: str
    split: str
    confidence: float
    iou: float
    image_size: int
    batch: int
    device: str | int | None
    models: list[ModelValidationResult]
    id: str = field(default_factory=lambda: uuid4().hex)
    status: ExperimentStatus = ExperimentStatus.QUEUED
    error: str | None = None
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
