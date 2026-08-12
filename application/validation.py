"""Sequential multi-model validation use case."""

from __future__ import annotations

import time
from typing import Any, Protocol

from domain.entities import ExperimentStatus, utc_now
from domain.validation import ValidationJob


class ModelValidator(Protocol):
    def validate(self, job: ValidationJob, model_index: int) -> tuple[dict[str, float], list[dict[str, Any]], str]: ...


class ValidationRepository(Protocol):
    def save(self, job: ValidationJob) -> None: ...


class ValidationService:
    def __init__(self, validator: ModelValidator, repository: ValidationRepository) -> None:
        self.validator = validator
        self.repository = repository

    def run(self, job: ValidationJob) -> None:
        job.status = ExperimentStatus.RUNNING
        job.error = None
        job.updated_at = utc_now()
        self.repository.save(job)
        try:
            for index, model in enumerate(job.models):
                if model.status == "completed":
                    continue
                model.status = "running"
                model.error = None
                self.repository.save(job)
                started = time.monotonic()
                try:
                    metrics, per_class, run_dir = self.validator.validate(job, index)
                    model.metrics = {name: float(value) for name, value in metrics.items()}
                    model.per_class = per_class
                    model.run_directory = run_dir
                    model.status = "completed"
                except Exception as exc:
                    model.status = "failed"
                    model.error = str(exc)
                model.duration_seconds = round(time.monotonic() - started, 3)
                job.updated_at = utc_now()
                self.repository.save(job)

            completed = [model for model in job.models if model.status == "completed"]
            if not completed:
                raise RuntimeError("all model validations failed")
            job.status = ExperimentStatus.COMPLETED
            if any(model.status == "failed" for model in job.models):
                job.error = "one or more models failed validation"
        except Exception as exc:
            job.status = ExperimentStatus.FAILED
            job.error = str(exc)
        finally:
            job.updated_at = utc_now()
            self.repository.save(job)
