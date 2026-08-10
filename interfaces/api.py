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
from domain.entities import Experiment, ExperimentConfig, Range, SearchSpace, SUPPORTED_METRICS, TaskType
from domain.training import TrainingJob, TrainingMode
from infrastructure.datasets import inspect_dataset
from infrastructure.experiment_importer import get_imported_experiment, read_experiment_file
from infrastructure.final_trainer import UltralyticsFinalTrainer
from infrastructure.repository import JsonExperimentRepository
from infrastructure.training_repository import JsonTrainingJobRepository
from infrastructure.yolo_trainer import UltralyticsTrainer

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.getenv("HYPER_GUI_DATA", ROOT / "data"))
repository = JsonExperimentRepository(DATA_DIR / "experiments.json")
trainer = UltralyticsTrainer(DATA_DIR / "runs")
optimizer = HyperparameterOptimizer(trainer, repository)
training_repository = JsonTrainingJobRepository(DATA_DIR / "training_jobs.json")
final_trainer = UltralyticsFinalTrainer(DATA_DIR / "final_runs")
final_training = FinalTrainingService(final_trainer, training_repository)
executor = ThreadPoolExecutor(max_workers=int(os.getenv("HYPER_GUI_WORKERS", "1")))
cancel_events: dict[str, threading.Event] = {}
active_training_jobs: set[str] = set()
active_training_lock = threading.Lock()

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


@asynccontextmanager
async def lifespan(_: FastAPI):
    for experiment in repository.list():
        if experiment.status.value in ("queued", "running"):
            enqueue_experiment(experiment)
    for job in training_repository.list():
        if job.status.value in ("queued", "running"):
            enqueue_training_job(job)
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
    experiment_path: str = Field(min_length=1)
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


@app.get("/")
def index() -> FileResponse:
    return FileResponse(ROOT / "frontend" / "index.html")


@app.get("/training")
def training_page() -> FileResponse:
    return FileResponse(ROOT / "frontend" / "training.html")


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
        source = get_imported_experiment(request.experiment_path, request.experiment_id)
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
