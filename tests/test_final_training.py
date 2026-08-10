import json
import sys
from pathlib import Path
from types import SimpleNamespace

from application.final_training import FinalTrainingService
from domain.entities import ExperimentStatus
from domain.training import TrainingJob, TrainingMode
from infrastructure.experiment_importer import read_experiment_file
from infrastructure.final_trainer import UltralyticsFinalTrainer
from infrastructure.training_repository import JsonTrainingJobRepository
from interfaces.api import _ensure_train_and_val_only
from domain.entities import TaskType


def make_job(**overrides) -> TrainingJob:
    values = {
        "name": "Final model",
        "mode": TrainingMode.NEW,
        "experiment_path": "/tmp/experiments.json",
        "experiment_id": "experiment-1",
        "best_trial": 2,
        "task": "detect",
        "model": "yolov8n.pt",
        "dataset": "/tmp/data.yaml",
        "epochs": 100,
        "batch": 8,
        "image_size": 640,
        "device": "cpu",
        "hyperparameters": {"lr0": 0.001, "optimizer": "Adam"},
    }
    values.update(overrides)
    return TrainingJob(**values)


def test_importer_finds_best_trial_and_available_latest_weights(tmp_path: Path):
    dataset = tmp_path / "data.yaml"
    dataset.write_text("train: images/train\nval: images/val\ntest: images/test\n", encoding="utf-8")
    run = tmp_path / "run-2"
    (run / "weights").mkdir(parents=True)
    (run / "weights" / "last.pt").write_bytes(b"checkpoint")
    source = tmp_path / "experiments.json"
    source.write_text(
        json.dumps(
            {
                "experiment-1": {
                    "id": "experiment-1",
                    "status": "completed",
                    "best_trial": 2,
                    "config": {"name": "Search", "task": "detect", "model": "yolov8n.pt", "dataset": str(dataset)},
                    "trials": [
                        {"number": 1, "score": 0.4},
                        {"number": 2, "score": 0.8, "metrics": {"f1": 0.8}, "hyperparameters": {"batch": 8}, "run_directory": str(run)},
                    ],
                }
            }
        ),
        encoding="utf-8",
    )

    imported = read_experiment_file(str(source))[0]
    assert imported["best_trial"] == 2
    assert imported["last_weights"] == str(run / "weights" / "last.pt")
    assert imported["dataset_splits"] == {"train": "images/train", "val": "images/val", "test": "images/test"}


def test_final_trainer_uses_train_validation_and_recovers_checkpoint(tmp_path: Path, monkeypatch):
    calls = []

    class FakeModel:
        def __init__(self, source, task=None):
            calls.append(("model", source, task))

        def train(self, **kwargs):
            calls.append(("train", kwargs))
            return SimpleNamespace(results_dict={"fitness": 0.7}, save_dir=tmp_path / "saved")

    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=FakeModel))
    job = make_job(id="fixed-id")
    trainer = UltralyticsFinalTrainer(tmp_path)

    trainer.train(job)
    first_args = calls[-1][1]
    assert first_args["val"] is True
    assert first_args["data"] == job.dataset
    assert "split" not in first_args

    checkpoint = tmp_path / "fixed-id-Final-model" / "weights" / "last.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    _, _, resumed = trainer.train(job)
    assert resumed is True
    assert calls[-1] == ("train", {"resume": True, "val": True})


def test_failed_job_is_persisted_and_can_be_run_again(tmp_path: Path):
    class FlakyTrainer:
        def __init__(self):
            self.calls = 0

        def train(self, job):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("interrupted")
            return {"fitness": 0.9}, "/tmp/final", True

    repository = JsonTrainingJobRepository(tmp_path / "jobs.json")
    trainer = FlakyTrainer()
    service = FinalTrainingService(trainer, repository)
    job = make_job()
    service.run(job)
    assert job.status is ExperimentStatus.FAILED
    service.run(job)
    assert job.status is ExperimentStatus.COMPLETED
    assert repository.get(job.id).resumed is True


def test_classification_final_training_never_falls_back_to_test(tmp_path: Path):
    (tmp_path / "train").mkdir()
    (tmp_path / "test").mkdir()
    try:
        _ensure_train_and_val_only(str(tmp_path), TaskType.CLASSIFY)
        assert False, "a test-only evaluation split must not be accepted as validation"
    except ValueError as exc:
        assert "requires a val folder" in str(exc)
    (tmp_path / "val").mkdir()
    _ensure_train_and_val_only(str(tmp_path), TaskType.CLASSIFY)
