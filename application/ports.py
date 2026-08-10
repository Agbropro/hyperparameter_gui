"""Ports implemented by infrastructure adapters."""

from __future__ import annotations

from typing import Any, Protocol

from domain.entities import Experiment


class ModelTrainer(Protocol):
    def train(
        self,
        *,
        task: str,
        model: str,
        dataset: str,
        epochs: int,
        image_size: int,
        device: str | int | None,
        hyperparameters: dict[str, Any],
        run_name: str,
    ) -> tuple[dict[str, float], str | None]: ...


class ExperimentRepository(Protocol):
    def save(self, experiment: Experiment) -> None: ...

    def get(self, experiment_id: str) -> Experiment | None: ...

    def list(self) -> list[Experiment]: ...
