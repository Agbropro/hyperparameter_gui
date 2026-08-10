"""Full-budget training use case with checkpoint recovery."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from domain.entities import ExperimentStatus, utc_now
from domain.training import TrainingJob


class FinalTrainer(Protocol):
    def train(self, job: TrainingJob) -> tuple[dict[str, float], str, bool]: ...


class TrainingJobRepository(Protocol):
    def save(self, job: TrainingJob) -> None: ...


class FinalTrainingService:
    def __init__(self, trainer: FinalTrainer, repository: TrainingJobRepository) -> None:
        self.trainer = trainer
        self.repository = repository

    def run(self, job: TrainingJob) -> None:
        job.status = ExperimentStatus.RUNNING
        job.error = None
        job.updated_at = utc_now()
        self.repository.save(job)
        try:
            metrics, run_directory, resumed = self.trainer.train(job)
            job.metrics = {name: float(value) for name, value in metrics.items()}
            job.run_directory = run_directory
            job.resumed = resumed
            job.status = ExperimentStatus.COMPLETED
        except Exception as exc:
            job.status = ExperimentStatus.FAILED
            job.error = str(exc)
        finally:
            job.updated_at = utc_now()
            self.repository.save(job)
