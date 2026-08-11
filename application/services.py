"""Hyperparameter search use case."""

from __future__ import annotations

import math
import random
import time
from typing import Any, Callable

from application.ports import ExperimentRepository, ModelTrainer
from domain.entities import Experiment, ExperimentStatus, SearchSpace, TrialResult, utc_now
from domain.naming import optimizer_run_name


class HyperparameterOptimizer:
    def __init__(self, trainer: ModelTrainer, repository: ExperimentRepository) -> None:
        self.trainer = trainer
        self.repository = repository

    @staticmethod
    def sample(space: SearchSpace, rng: random.Random) -> dict[str, Any]:
        values: dict[str, Any] = {}
        for name, bounds in space.ranges.items():
            if bounds.log:
                value = math.exp(rng.uniform(math.log(bounds.low), math.log(bounds.high)))
            else:
                value = rng.uniform(bounds.low, bounds.high)
            values[name] = int(round(value)) if bounds.integer else round(value, 8)
        for name, choices in space.choices.items():
            if not choices:
                raise ValueError(f"search choices for {name} cannot be empty")
            values[name] = rng.choice(choices)
        return values

    @staticmethod
    def score(metrics: dict[str, float], selected: list[str]) -> float:
        normalized = dict(metrics)
        if "f1" not in normalized:
            precision, recall = normalized.get("precision"), normalized.get("recall")
            if precision is not None and recall is not None and precision + recall:
                normalized["f1"] = 2 * precision * recall / (precision + recall)
        missing = [name for name in selected if name not in normalized]
        if missing:
            raise ValueError(f"trainer did not return selected metric(s): {', '.join(missing)}")
        return sum(float(normalized[name]) for name in selected) / len(selected)

    def run(self, experiment: Experiment, should_cancel: Callable[[], bool] | None = None) -> None:
        config = experiment.config
        rng = random.Random(config.seed)
        experiment.status = ExperimentStatus.RUNNING
        experiment.updated_at = utc_now()
        self.repository.save(experiment)

        try:
            for number in range(1, config.trials + 1):
                # Sampling is performed even for an existing trial so the seeded
                # random-number stream remains aligned for later trial numbers.
                sampled_params = self.sample(config.search_space, rng)
                if should_cancel and should_cancel():
                    experiment.status = ExperimentStatus.CANCELLED
                    break

                existing = next((item for item in experiment.trials if item.number == number), None)
                if existing and existing.status in ("completed", "failed"):
                    continue

                if existing:
                    # A running trial persisted before a process interruption.
                    # Reuse its exact configuration; the trainer adapter will use
                    # last.pt when one is available.
                    trial = existing
                    params = dict(trial.hyperparameters)
                    trial.error = None
                    trial.status = "running"
                    # Persisted trials created before readable names were added use
                    # their original folder name so checkpoint recovery still works.
                    run_name = trial.run_name or f"{experiment.id}-trial-{number}"
                else:
                    params = sampled_params
                    # Keep training randomness constant across trials so
                    # hyperparameters are the main changing variable.
                    params["seed"] = config.seed
                    run_name = optimizer_run_name(config.name, experiment.id, number)
                    trial = TrialResult(
                        number=number,
                        hyperparameters=params,
                        status="running",
                        run_name=run_name,
                    )
                    experiment.trials.append(trial)
                self.repository.save(experiment)
                started = time.monotonic()
                try:
                    metrics, run_dir = self.trainer.train(
                        task=config.task.value,
                        model=config.model,
                        dataset=config.dataset,
                        epochs=config.epochs,
                        image_size=config.image_size,
                        device=config.device,
                        hyperparameters=params,
                        run_name=run_name,
                    )
                    trial.metrics = {key: float(value) for key, value in metrics.items()}
                    trial.score = self.score(trial.metrics, config.metrics)
                    trial.status = "completed"
                    trial.run_directory = run_dir
                except Exception as exc:  # preserve the remaining search after a bad trial
                    trial.status = "failed"
                    trial.error = str(exc)
                trial.duration_seconds = round(time.monotonic() - started, 3)
                completed = [item for item in experiment.trials if item.score is not None]
                if completed:
                    experiment.best_trial = max(completed, key=lambda item: item.score or 0).number
                experiment.updated_at = utc_now()
                self.repository.save(experiment)

            if experiment.status != ExperimentStatus.CANCELLED:
                if not any(item.status == "completed" for item in experiment.trials):
                    raise RuntimeError("all training trials failed")
                experiment.status = ExperimentStatus.COMPLETED
        except Exception as exc:
            experiment.status = ExperimentStatus.FAILED
            experiment.error = str(exc)
        finally:
            experiment.updated_at = utc_now()
            self.repository.save(experiment)
