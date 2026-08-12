import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from application.validation import ValidationService
from domain.entities import ExperimentStatus
from domain.validation import ModelValidationResult, ValidationJob
from infrastructure.validation_repository import JsonValidationRepository
from infrastructure.yolo_validator import UltralyticsModelValidator
from interfaces.api import _ensure_test_split


def make_job(tmp_path: Path) -> ValidationJob:
    return ValidationJob(
        name="Compare finals",
        dataset=str(tmp_path / "data.yaml"),
        split="test",
        confidence=0.25,
        iou=0.6,
        image_size=640,
        batch=8,
        device="cpu",
        models=[
            ModelValidationResult(label="Model A", model_path=str(tmp_path / "a.pt")),
            ModelValidationResult(label="Model B", model_path=str(tmp_path / "b.pt")),
        ],
        id="validation-id",
    )


def test_validator_calls_model_val_on_test_split(tmp_path: Path, monkeypatch):
    job = make_job(tmp_path)
    Path(job.models[0].model_path).write_bytes(b"weights")
    calls = []

    class FakeMetric:
        mp = 0.8
        mr = 0.7
        map50 = 0.85
        map75 = 0.6
        map = 0.5

    class FakeResult:
        results_dict = {"fitness": 0.5}
        box = FakeMetric()
        speed = {"inference": 2.3}
        save_dir = tmp_path / "validation-output"

        def summary(self, **kwargs):
            return [{"Class": "all", "Precision": 0.8}]

    class FakeYOLO:
        def __init__(self, path):
            calls.append(("load", path))

        def val(self, **kwargs):
            calls.append(("val", kwargs))
            return FakeResult()

    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=FakeYOLO))
    metrics, per_class, _ = UltralyticsModelValidator(tmp_path / "outputs").validate(job, 0)

    arguments = calls[-1][1]
    assert arguments["split"] == "test"
    assert arguments["conf"] == 0.25
    assert arguments["iou"] == 0.6
    assert metrics["f1"] == pytest.approx(2 * 0.8 * 0.7 / 1.5)
    assert metrics["map50_95"] == 0.5
    assert metrics["inference_ms"] == 2.3
    assert per_class == [{"Class": "all", "Precision": 0.8}]


def test_validation_service_keeps_success_and_recovers_unfinished_models(tmp_path: Path):
    repository = JsonValidationRepository(tmp_path / "validation_jobs.json")
    job = make_job(tmp_path)
    job.models[0].status = "completed"
    job.models[0].metrics = {"map50": 0.8}

    class Validator:
        calls = []

        def validate(self, current_job, index):
            self.calls.append(index)
            return {"map50": 0.9}, [], "/tmp/validation"

    validator = Validator()
    ValidationService(validator, repository).run(job)
    assert validator.calls == [1]
    assert job.status is ExperimentStatus.COMPLETED
    assert repository.get(job.id).models[0].metrics == {"map50": 0.8}


def test_validation_repository_round_trip(tmp_path: Path):
    repository = JsonValidationRepository(tmp_path / "jobs.json")
    job = make_job(tmp_path)
    repository.save(job)
    loaded = repository.get(job.id)
    assert loaded.id == job.id
    assert len(loaded.models) == 2


def test_dataset_yaml_requires_held_out_test_split(tmp_path: Path):
    dataset = tmp_path / "data.yaml"
    dataset.write_text("train: images/train\nval: images/val\n", encoding="utf-8")
    with pytest.raises(ValueError, match="must define a test split"):
        _ensure_test_split(dataset)
    dataset.write_text("train: images/train\nval: images/val\ntest: images/test\n", encoding="utf-8")
    _ensure_test_split(dataset)
