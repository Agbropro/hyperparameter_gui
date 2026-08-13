import json
from pathlib import Path

import pytest

from domain.entities import Experiment, ExperimentConfig, TaskType, TrialResult
from domain.training import TrainingJob, TrainingMode
from domain.ticket import Ticket, TicketType
from domain.validation import ModelValidationResult, ValidationJob
from infrastructure.repository import JsonExperimentRepository
from infrastructure.sqlite import (
    SqliteExperimentRepository,
    SqliteTicketRepository,
    SqliteTrainingJobRepository,
    SqliteValidationRepository,
    checkpoint_database,
    database_summary,
    initialize_database,
)
from infrastructure.training_repository import JsonTrainingJobRepository
from infrastructure.validation_repository import JsonValidationRepository


def test_migrates_all_json_histories_without_changing_sources(tmp_path: Path):
    experiment = Experiment(
        config=ExperimentConfig(
            name="Search", task=TaskType.DETECT, model="yolo11n.pt", dataset="/data/data.yaml",
            trials=1, epochs=2, metrics=["f1"],
        ),
        trials=[TrialResult(number=1, hyperparameters={"batch": 8}, metrics={"f1": 0.7}, score=0.7)],
        best_trial=1,
    )
    training = TrainingJob(
        name="Final", mode=TrainingMode.NEW, experiment_path=str(tmp_path / "experiments.json"),
        experiment_id=experiment.id, best_trial=1, task="detect", model="yolo11n.pt",
        dataset="/data/data.yaml", epochs=10, batch=8, image_size=640, device="cpu",
        hyperparameters={"lr0": 0.01},
    )
    validation = ValidationJob(
        name="Compare", dataset="/data/data.yaml", split="test", confidence=0.001, iou=0.7,
        image_size=640, batch=8, device="cpu",
        models=[ModelValidationResult(label="Final", model_path="/models/best.pt", metrics={"map50": 0.8})],
    )
    JsonExperimentRepository(tmp_path / "experiments.json").save(experiment)
    JsonTrainingJobRepository(tmp_path / "training_jobs.json").save(training)
    JsonValidationRepository(tmp_path / "validation_jobs.json").save(validation)
    originals = {path.name: path.read_bytes() for path in tmp_path.glob("*.json")}

    database = initialize_database(tmp_path)

    assert SqliteExperimentRepository(database).get(experiment.id).best_trial == 1
    assert SqliteTrainingJobRepository(database).get(training.id).hyperparameters == {"lr0": 0.01}
    assert SqliteValidationRepository(database).get(validation.id).models[0].metrics == {"map50": 0.8}
    assert {path.name: path.read_bytes() for path in tmp_path.glob("*.json")} == originals
    assert database_summary(database) == {
        "experiments": 1,
        "experiment_trials": 1,
        "training_jobs": 1,
        "validation_jobs": 1,
        "validation_models": 1,
        "tickets": 0,
        "integrity_check": "ok",
        "foreign_key_violations": 0,
    }


def test_sqlite_repositories_update_individual_records(tmp_path: Path):
    database = initialize_database(tmp_path)
    repository = SqliteExperimentRepository(database)
    experiment = Experiment(
        config=ExperimentConfig(
            name="Search", task=TaskType.DETECT, model="yolo11n.pt", dataset="data.yaml",
            trials=1, epochs=1, metrics=["f1"],
        )
    )
    repository.save(experiment)
    experiment.trials.append(TrialResult(number=1, hyperparameters={"batch": 4}, status="completed"))
    experiment.best_trial = 1
    repository.save(experiment)
    loaded = repository.get(experiment.id)
    assert loaded.best_trial == 1
    assert loaded.trials[0].hyperparameters == {"batch": 4}


def test_ticket_repository_persists_developer_visible_reports(tmp_path: Path):
    database = initialize_database(tmp_path)
    repository = SqliteTicketRepository(database)
    ticket = Ticket(
        title="Add dark mode",
        type=TicketType.FEATURE,
        message="A dark theme would make overnight training easier to monitor.",
        page="/training",
    )

    repository.save(ticket)

    loaded = repository.get(ticket.id)
    assert loaded is not None
    assert loaded.title == "Add dark mode"
    assert loaded.type is TicketType.FEATURE
    assert loaded.status == "new"
    assert repository.list()[0].page == "/training"
    assert database_summary(database)["tickets"] == 1
    assert checkpoint_database(database) is True


def test_invalid_legacy_json_aborts_without_replacing_database(tmp_path: Path):
    (tmp_path / "experiments.json").write_text("{not valid", encoding="utf-8")
    with pytest.raises(RuntimeError, match="cannot migrate invalid JSON"):
        initialize_database(tmp_path)
    assert not (tmp_path / "studio.db").exists()
    assert not (tmp_path / "studio.db.migrating").exists()
