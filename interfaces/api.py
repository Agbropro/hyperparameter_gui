"""FastAPI composition root and HTTP API."""

from __future__ import annotations

import os
import threading
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from application.services import HyperparameterOptimizer
from application.final_training import FinalTrainingService
from application.validation import ValidationService
from domain.entities import Experiment, ExperimentConfig, Range, SearchSpace, SUPPORTED_METRICS, TaskType
from domain.training import TrainingJob, TrainingMode
from domain.validation import ModelValidationResult, ValidationJob
from domain.ticket import Ticket, TicketType
from domain.naming import final_run_name
from infrastructure.datasets import inspect_dataset
from infrastructure.experiment_importer import get_imported_experiment, read_dataset_splits, read_experiment_file
from infrastructure.final_trainer import UltralyticsFinalTrainer
from infrastructure.yolo_trainer import UltralyticsTrainer
from infrastructure.yolo_validator import UltralyticsModelValidator
from infrastructure.sqlite import (
    SqliteExperimentRepository,
    SqliteTrainingJobRepository,
    SqliteValidationRepository,
    SqliteTicketRepository,
    initialize_database,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.getenv("HYPER_GUI_DATA", ROOT / "data"))
DATABASE_PATH = initialize_database(DATA_DIR)
repository = SqliteExperimentRepository(DATABASE_PATH)
trainer = UltralyticsTrainer(DATA_DIR / "runs")
optimizer = HyperparameterOptimizer(trainer, repository)
training_repository = SqliteTrainingJobRepository(DATABASE_PATH)
final_trainer = UltralyticsFinalTrainer(DATA_DIR / "final_runs")
final_training = FinalTrainingService(final_trainer, training_repository)
validation_repository = SqliteValidationRepository(DATABASE_PATH)
model_validator = UltralyticsModelValidator(DATA_DIR / "validation_runs")
validation_service = ValidationService(model_validator, validation_repository)
ticket_repository = SqliteTicketRepository(DATABASE_PATH)
executor = ThreadPoolExecutor(max_workers=int(os.getenv("HYPER_GUI_WORKERS", "1")))
cancel_events: dict[str, threading.Event] = {}
active_training_jobs: set[str] = set()
active_training_lock = threading.Lock()
active_validation_jobs: set[str] = set()
active_validation_lock = threading.Lock()

def enqueue_experiment(experiment: Experiment) -> None:
    """Queue an experiment once, including recovery after an app restart."""
    event = cancel_events.setdefault(experiment.id, threading.Event())
    executor.submit(optimizer.run, experiment, event.is_set)


def enqueue_training_job(job: TrainingJob) -> None:
    with active_training_lock:
        if job.id in active_training_jobs:
            return
        active_training_jobs.add(job.id)

    def run() -> None:
        try:
            final_training.run(job)
        finally:
            with active_training_lock:
                active_training_jobs.discard(job.id)

    executor.submit(run)


def enqueue_validation_job(job: ValidationJob) -> None:
    with active_validation_lock:
        if job.id in active_validation_jobs:
            return
        active_validation_jobs.add(job.id)

    def run() -> None:
        try:
            validation_service.run(job)
        finally:
            with active_validation_lock:
                active_validation_jobs.discard(job.id)

    executor.submit(run)


@asynccontextmanager
async def lifespan(_: FastAPI):
    for experiment in repository.list():
        if experiment.status.value in ("queued", "running"):
            enqueue_experiment(experiment)
    for job in training_repository.list():
        if job.status.value in ("queued", "running"):
            enqueue_training_job(job)
    for job in validation_repository.list():
        if job.status.value in ("queued", "running"):
            enqueue_validation_job(job)
    yield


app = FastAPI(title="YOLO Hyperparameter Studio", version="1.0.0", lifespan=lifespan)
app.mount("/assets", StaticFiles(directory=ROOT / "frontend"), name="assets")


class DatasetRequest(BaseModel):
    path: str
    task: TaskType


class RangeRequest(BaseModel):
    low: float
    high: float
    integer: bool = False
    log: bool = False


class ExperimentRequest(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    task: TaskType
    model: str = Field(min_length=1)
    dataset: str = Field(min_length=1)
    trials: int = Field(default=10, ge=1, le=1000)
    epochs: int = Field(default=30, ge=1, le=10000)
    metrics: list[str]
    image_size: int = Field(default=640, ge=32)
    device: str | int | None = None
    seed: int = 42
    ranges: dict[str, RangeRequest] | None = None
    choices: dict[str, list[Any]] | None = None


class ExperimentFileRequest(BaseModel):
    path: str = Field(min_length=1)


class TrainingJobRequest(BaseModel):
    experiment_path: str | None = None
    experiment_id: str = Field(min_length=1)
    name: str = Field(min_length=1, max_length=100)
    mode: TrainingMode
    epochs: int = Field(ge=1, le=10000)
    dataset: str = Field(min_length=1)
    task: TaskType
    version: str
    size: str
    batch: int = Field(ge=1)
    image_size: int = Field(ge=32)
    device: str | int | None = None


class ValidationModelRequest(BaseModel):
    label: str = Field(min_length=1, max_length=100)
    path: str = Field(min_length=1)


class ValidationJobRequest(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    dataset: str = Field(min_length=1)
    models: list[ValidationModelRequest] = Field(min_length=1, max_length=20)
    confidence: float = Field(default=0.001, ge=0.0, le=1.0)
    iou: float = Field(default=0.7, ge=0.0, le=1.0)
    image_size: int = Field(default=640, ge=32)
    batch: int = Field(default=16, ge=1)
    device: str | int | None = None


class TicketRequest(BaseModel):
    title: str = Field(min_length=3, max_length=120)
    type: TicketType
    message: str = Field(min_length=5, max_length=5000)
    page: str | None = Field(default=None, max_length=500)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(ROOT / "frontend" / "index.html")


@app.get("/training")
def training_page() -> FileResponse:
    return FileResponse(ROOT / "frontend" / "training.html")


@app.get("/validation")
def validation_page() -> FileResponse:
    return FileResponse(ROOT / "frontend" / "validation.html")


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/options")
def options() -> dict[str, Any]:
    defaults = SearchSpace.yolo_defaults()
    return {
        "metrics": {task.value: values for task, values in SUPPORTED_METRICS.items()},
        "versions": {
            "yolo26": {"label": "YOLO26", "note": "Latest · recommended"},
            "yolo11": {"label": "YOLO11", "note": "Stable production"},
            "yolov8": {"label": "YOLOv8", "note": "Mature baseline"},
        },
        "model_sizes": ["n", "s", "m", "l", "x"],
        "search_space": defaults,
    }


@app.post("/api/datasets/inspect")
def inspect(request: DatasetRequest) -> dict[str, Any]:
    try:
        return inspect_dataset(request.path, request.task)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/experiments")
def list_experiments() -> list[dict[str, Any]]:
    return [item.to_dict() for item in repository.list()]


@app.get("/api/experiments/{experiment_id}")
def get_experiment(experiment_id: str) -> dict[str, Any]:
    experiment = repository.get(experiment_id)
    if not experiment:
        raise HTTPException(status_code=404, detail="experiment not found")
    return experiment.to_dict()


@app.post("/api/experiments", status_code=202)
def create_experiment(request: ExperimentRequest) -> dict[str, Any]:
    try:
        dataset = inspect_dataset(request.dataset, request.task)["dataset"]
        defaults = SearchSpace.yolo_defaults()
        space = SearchSpace(
            ranges={name: Range(**bounds.model_dump()) for name, bounds in request.ranges.items()}
            if request.ranges is not None
            else defaults.ranges,
            choices=request.choices if request.choices is not None else defaults.choices,
        )
        config = ExperimentConfig(
            name=request.name,
            task=request.task,
            model=request.model,
            dataset=dataset,
            trials=request.trials,
            epochs=request.epochs,
            metrics=request.metrics,
            search_space=space,
            image_size=request.image_size,
            device=request.device,
            seed=request.seed,
        )
        config.validate()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    experiment = Experiment(config=config)
    repository.save(experiment)
    enqueue_experiment(experiment)
    return experiment.to_dict()


@app.post("/api/experiments/{experiment_id}/cancel", status_code=202)
def cancel_experiment(experiment_id: str) -> dict[str, str]:
    experiment = repository.get(experiment_id)
    if not experiment:
        raise HTTPException(status_code=404, detail="experiment not found")
    cancel_events.setdefault(experiment_id, threading.Event()).set()
    return {"status": "cancellation_requested"}


@app.post("/api/training/experiments/inspect")
def inspect_experiment_file(request: ExperimentFileRequest) -> dict[str, Any]:
    try:
        experiments = read_experiment_file(request.path)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"path": experiments[0]["source_path"], "experiments": experiments}


@app.get("/api/training/experiments")
def list_training_experiments() -> dict[str, Any]:
    """Return optimizer winners directly from the active SQLite database."""
    experiments = []
    for experiment in repository.list():
        source = _experiment_source(experiment)
        if source:
            experiments.append(source)
    return {"source": "sqlite", "path": str(DATABASE_PATH), "experiments": experiments}


@app.get("/api/training/jobs")
def list_training_jobs() -> list[dict[str, Any]]:
    return [job.to_dict() for job in training_repository.list()]


@app.get("/api/training/jobs/{job_id}")
def get_training_job(job_id: str) -> dict[str, Any]:
    job = training_repository.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="training job not found")
    return job.to_dict()


@app.post("/api/training/jobs", status_code=202)
def create_training_job(request: TrainingJobRequest) -> dict[str, Any]:
    try:
        if request.experiment_path:
            source = get_imported_experiment(request.experiment_path, request.experiment_id)
        else:
            experiment = repository.get(request.experiment_id)
            source = _experiment_source(experiment) if experiment else None
            if not source:
                raise ValueError("selected optimizer experiment has no completed best trial")
        selected_task = TaskType(source["task"]) if request.mode is TrainingMode.CONTINUE else request.task
        selected_dataset = source["dataset"] if request.mode is TrainingMode.CONTINUE else request.dataset
        inspected = inspect_dataset(selected_dataset, selected_task)
        dataset = inspected["dataset"]
        _ensure_train_and_val_only(dataset, selected_task)
        if request.version not in {"yolo26", "yolo11", "yolov8"}:
            raise ValueError("unsupported YOLO version")
        if request.size not in {"n", "s", "m", "l", "x"}:
            raise ValueError("model size must be n, s, m, l, or x")

        hyperparameters = dict(source["hyperparameters"])
        if request.mode is TrainingMode.CONTINUE:
            if not source["last_weights"]:
                raise ValueError("the winning trial's weights/last.pt is not available on this server")
            task = source["task"]
            model = source["model"]
            batch = int(hyperparameters.get("batch", request.batch))
            image_size = int(source["image_size"])
            device = source.get("device")
            source_weights = source["last_weights"]
        else:
            task = request.task.value
            model = _model_name(request.version, request.size, task)
            batch = request.batch
            image_size = request.image_size
            device = request.device
            source_weights = None

        # Explicit final-training controls take precedence over the tuned values.
        hyperparameters.pop("batch", None)
        job = TrainingJob(
            name=request.name,
            mode=request.mode,
            experiment_path=source["source_path"],
            experiment_id=source["id"],
            best_trial=source["best_trial"],
            task=task,
            model=model,
            dataset=dataset,
            epochs=request.epochs,
            batch=batch,
            image_size=image_size,
            device=device,
            hyperparameters=hyperparameters,
            source_weights=source_weights,
        )
        job.run_name = final_run_name(job.name, job.id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    training_repository.save(job)
    enqueue_training_job(job)
    return job.to_dict()


@app.post("/api/training/jobs/{job_id}/resume", status_code=202)
def resume_training_job(job_id: str) -> dict[str, Any]:
    job = training_repository.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="training job not found")
    if job.status.value not in ("failed", "queued"):
        raise HTTPException(status_code=409, detail="only failed or queued jobs can be resumed")
    job.status = job.status.__class__.QUEUED
    job.error = None
    training_repository.save(job)
    enqueue_training_job(job)
    return job.to_dict()


def _model_name(version: str, size: str, task: str) -> str:
    suffix = {"detect": "", "segment": "-seg", "classify": "-cls"}[task]
    return f"{version}{size}{suffix}.pt"


def _experiment_source(experiment: Experiment | None) -> dict[str, Any] | None:
    """Translate a persisted optimizer winner into the final-training preview."""
    if not experiment or experiment.best_trial is None:
        return None
    best = next(
        (
            trial
            for trial in experiment.trials
            if trial.number == experiment.best_trial and trial.status == "completed"
        ),
        None,
    )
    if not best:
        return None
    run_dir = Path(best.run_directory) if best.run_directory else None
    last_weights = run_dir / "weights" / "last.pt" if run_dir else None
    best_weights = run_dir / "weights" / "best.pt" if run_dir else None
    config = experiment.config
    return {
        "id": experiment.id,
        "name": config.name,
        "status": experiment.status.value,
        "task": config.task.value,
        "model": config.model,
        "dataset": config.dataset,
        "image_size": config.image_size,
        "device": config.device,
        "best_trial": best.number,
        "metrics": best.metrics,
        "score": best.score,
        "hyperparameters": best.hyperparameters,
        "run_directory": best.run_directory,
        "last_weights": str(last_weights) if last_weights and last_weights.is_file() else None,
        "best_weights": str(best_weights) if best_weights and best_weights.is_file() else None,
        "dataset_splits": read_dataset_splits(config.dataset, config.task.value),
        "source_path": str(DATABASE_PATH),
        "source": "sqlite",
    }


def _ensure_train_and_val_only(dataset: str, task: TaskType) -> None:
    if task is TaskType.CLASSIFY:
        root = Path(dataset)
        if not any((root / name).is_dir() for name in ("val", "valid", "validation")):
            raise ValueError(
                "classification final training requires a val folder; the test folder is reserved"
            )
        return
    import yaml

    try:
        document = yaml.safe_load(Path(dataset).read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"could not read dataset YAML: {exc}") from exc
    missing = [split for split in ("train", "val") if not document.get(split)]
    if missing:
        raise ValueError(f"dataset YAML must define {', '.join(missing)}")


@app.get("/api/validation/jobs")
def list_validation_jobs() -> list[dict[str, Any]]:
    return [job.to_dict() for job in validation_repository.list()]


@app.get("/api/validation/jobs/{job_id}")
def get_validation_job(job_id: str) -> dict[str, Any]:
    job = validation_repository.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="validation job not found")
    return job.to_dict()


@app.post("/api/validation/jobs", status_code=202)
def create_validation_job(request: ValidationJobRequest) -> dict[str, Any]:
    try:
        dataset = Path(request.dataset).expanduser().resolve()
        if not dataset.is_file():
            raise ValueError(f"dataset YAML does not exist: {dataset}")
        _ensure_test_split(dataset)
        models: list[ModelValidationResult] = []
        seen: set[str] = set()
        for item in request.models:
            path = Path(item.path).expanduser().resolve()
            if not path.is_file():
                raise ValueError(f"model weights do not exist: {path}")
            if path.suffix.lower() != ".pt":
                raise ValueError(f"model must be a .pt checkpoint: {path}")
            if str(path) in seen:
                raise ValueError(f"duplicate model path: {path}")
            seen.add(str(path))
            models.append(ModelValidationResult(label=item.label.strip(), model_path=str(path)))
        job = ValidationJob(
            name=request.name.strip(),
            dataset=str(dataset),
            split="test",
            confidence=request.confidence,
            iou=request.iou,
            image_size=request.image_size,
            batch=request.batch,
            device=request.device,
            models=models,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    validation_repository.save(job)
    enqueue_validation_job(job)
    return job.to_dict()


@app.post("/api/validation/jobs/{job_id}/retry", status_code=202)
def retry_validation_job(job_id: str) -> dict[str, Any]:
    job = validation_repository.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="validation job not found")
    failed = [model for model in job.models if model.status == "failed"]
    if not failed:
        raise HTTPException(status_code=409, detail="this job has no failed model validations")
    for model in failed:
        model.status = "queued"
        model.error = None
    job.status = job.status.__class__.QUEUED
    job.error = None
    validation_repository.save(job)
    enqueue_validation_job(job)
    return job.to_dict()


def _ensure_test_split(dataset: Path) -> None:
    import yaml

    try:
        document = yaml.safe_load(dataset.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"could not read dataset YAML: {exc}") from exc
    if not isinstance(document, dict) or not document.get("test"):
        raise ValueError("dataset YAML must define a test split for held-out validation")


@app.post("/api/tickets", status_code=201)
def create_ticket(request: TicketRequest) -> dict[str, str]:
    """Accept a ticket; reading remains database-only for now."""
    title = request.title.strip()
    message = request.message.strip()
    if len(title) < 3 or len(message) < 5:
        raise HTTPException(status_code=422, detail="title and description cannot be blank")
    ticket = Ticket(
        title=title,
        type=request.type,
        message=message,
        page=request.page.strip() if request.page else None,
    )
    ticket_repository.save(ticket)
    return {"id": ticket.id, "status": ticket.status}
