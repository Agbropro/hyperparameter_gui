"""Domain entities for hyperparameter experiments.

This module deliberately has no framework or Ultralytics imports.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class TaskType(str, Enum):
    DETECT = "detect"
    SEGMENT = "segment"
    CLASSIFY = "classify"


class ExperimentStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


SUPPORTED_METRICS: dict[TaskType, tuple[str, ...]] = {
    TaskType.DETECT: ("precision", "recall", "f1", "map50", "map50_95", "fitness"),
    TaskType.SEGMENT: ("precision", "recall", "f1", "map50", "map50_95", "fitness"),
    TaskType.CLASSIFY: ("accuracy_top1", "accuracy_top5", "fitness"),
}


@dataclass(frozen=True)
class Range:
    """A numeric search range. Integers produce integer samples."""

    low: float
    high: float
    integer: bool = False
    log: bool = False

    def __post_init__(self) -> None:
        if self.low > self.high:
            raise ValueError("range low must not exceed high")
        if self.log and self.low <= 0:
            raise ValueError("logarithmic ranges must be positive")


@dataclass
class SearchSpace:
    ranges: dict[str, Range] = field(default_factory=dict)
    choices: dict[str, list[Any]] = field(default_factory=dict)

    @classmethod
    def yolo_defaults(cls) -> "SearchSpace":
        return cls(
            ranges={
                "lr0": Range(1e-5, 1e-1, log=True),
                "lrf": Range(0.01, 1.0, log=True),
                "momentum": Range(0.6, 0.98),
                "weight_decay": Range(0.0, 0.001),
                "warmup_epochs": Range(0.0, 5.0),
                "warmup_momentum": Range(0.0, 0.95),
                "box": Range(1.0, 12.0),
                "cls": Range(0.2, 4.0),
                "dfl": Range(0.5, 3.0),
                "hsv_h": Range(0.0, 0.1),
                "hsv_s": Range(0.0, 0.9),
                "hsv_v": Range(0.0, 0.9),
                "degrees": Range(0.0, 45.0),
                "translate": Range(0.0, 0.5),
                "scale": Range(0.0, 0.9),
                "shear": Range(0.0, 10.0),
                "perspective": Range(0.0, 0.001),
                "flipud": Range(0.0, 1.0),
                "fliplr": Range(0.0, 1.0),
                "mosaic": Range(0.0, 1.0),
                "mixup": Range(0.0, 1.0),
                "copy_paste": Range(0.0, 1.0),
            },
            choices={"batch": [4, 8, 16, 32], "optimizer": ["SGD", "Adam", "AdamW"]},
        )


@dataclass
class ExperimentConfig:
    name: str
    task: TaskType
    model: str
    dataset: str
    trials: int
    epochs: int
    metrics: list[str]
    search_space: SearchSpace = field(default_factory=SearchSpace.yolo_defaults)
    device: str | int | None = None
    image_size: int = 640
    seed: int = 42

    def validate(self) -> None:
        if not self.name.strip():
            raise ValueError("experiment name is required")
        if not 1 <= self.trials <= 1000:
            raise ValueError("trials must be between 1 and 1000")
        if not 1 <= self.epochs <= 10000:
            raise ValueError("epochs must be between 1 and 10000")
        if self.image_size < 32:
            raise ValueError("image_size must be at least 32")
        valid = SUPPORTED_METRICS[self.task]
        invalid = set(self.metrics) - set(valid)
        if not self.metrics or invalid:
            raise ValueError(f"metrics must be selected from: {', '.join(valid)}")


@dataclass
class TrialResult:
    number: int
    hyperparameters: dict[str, Any]
    metrics: dict[str, float] = field(default_factory=dict)
    score: float | None = None
    duration_seconds: float | None = None
    status: str = "queued"
    error: str | None = None
    run_directory: str | None = None
    run_name: str | None = None


@dataclass
class Experiment:
    config: ExperimentConfig
    id: str = field(default_factory=lambda: uuid4().hex)
    status: ExperimentStatus = ExperimentStatus.QUEUED
    trials: list[TrialResult] = field(default_factory=list)
    best_trial: int | None = None
    error: str | None = None
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
