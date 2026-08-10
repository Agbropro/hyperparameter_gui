from pathlib import Path

import pytest

from application.services import HyperparameterOptimizer
from domain.entities import Experiment, ExperimentConfig, ExperimentStatus, Range, SearchSpace, TaskType, TrialResult
from infrastructure.repository import JsonExperimentRepository


class FakeTrainer:
    def __init__(self) -> None:
        self.calls = 0
        self.received = []

    def train(self, **kwargs):
        self.calls += 1
        self.received.append(kwargs)
        precision = 0.5 + self.calls / 10
        recall = 0.4 + self.calls / 10
        return {"precision": precision, "recall": recall}, f"/tmp/run-{self.calls}"


def config(trials: int = 3) -> ExperimentConfig:
    return ExperimentConfig(
        name="test",
        task=TaskType.DETECT,
        model="local.pt",
        dataset="data.yaml",
        trials=trials,
        epochs=1,
        metrics=["f1"],
        search_space=SearchSpace(ranges={"lr0": Range(0.001, 0.01, log=True)}, choices={"batch": [4]}),
    )


def test_optimizer_completes_and_picks_best_trial(tmp_path: Path):
    repository = JsonExperimentRepository(tmp_path / "experiments.json")
    experiment = Experiment(config=config())

    HyperparameterOptimizer(FakeTrainer(), repository).run(experiment)

    assert experiment.status is ExperimentStatus.COMPLETED
    assert len(experiment.trials) == 3
    assert experiment.best_trial == 3
    assert all(trial.status == "completed" for trial in experiment.trials)
    assert all(trial.hyperparameters["seed"] == 42 for trial in experiment.trials)
    assert repository.get(experiment.id).best_trial == 3


def test_sampling_is_reproducible():
    import random

    space = config().search_space
    left = HyperparameterOptimizer.sample(space, random.Random(42))
    right = HyperparameterOptimizer.sample(space, random.Random(42))
    assert left == right
    assert 0.001 <= left["lr0"] <= 0.01
    assert left["batch"] == 4


def test_score_computes_f1():
    score = HyperparameterOptimizer.score({"precision": 0.8, "recall": 0.5}, ["f1"])
    assert score == pytest.approx(2 * 0.8 * 0.5 / 1.3)


def test_cancellation_stops_before_first_trial(tmp_path: Path):
    repository = JsonExperimentRepository(tmp_path / "experiments.json")
    experiment = Experiment(config=config())
    HyperparameterOptimizer(FakeTrainer(), repository).run(experiment, lambda: True)
    assert experiment.status is ExperimentStatus.CANCELLED
    assert experiment.trials == []


def test_config_rejects_wrong_task_metric():
    value = config()
    value.metrics = ["accuracy_top1"]
    with pytest.raises(ValueError, match="metrics must be selected"):
        value.validate()


def test_recovery_skips_completed_and_reuses_interrupted_trial(tmp_path: Path):
    repository = JsonExperimentRepository(tmp_path / "experiments.json")
    experiment = Experiment(config=config())
    experiment.trials = [
        # Trial 1 must not run again.
        TrialResult(
            number=1,
            hyperparameters={"lr0": 0.002, "batch": 4, "seed": 42},
            metrics={"precision": 0.7, "recall": 0.6},
            score=0.64,
            status="completed",
        ),
        # Trial 2 must keep these exact parameters when resumed.
        TrialResult(
            number=2,
            hyperparameters={"lr0": 0.009, "batch": 4, "seed": 42},
            status="running",
        ),
    ]
    trainer = FakeTrainer()

    HyperparameterOptimizer(trainer, repository).run(experiment)

    assert trainer.calls == 2
    assert trainer.received[0]["hyperparameters"] == {"lr0": 0.009, "batch": 4, "seed": 42}
    assert [trial.number for trial in experiment.trials] == [1, 2, 3]
    assert experiment.status is ExperimentStatus.COMPLETED
