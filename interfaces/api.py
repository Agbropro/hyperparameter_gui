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
from domain.entities import Experiment, ExperimentConfig, Range, SearchSpace, SUPPORTED_METRICS, TaskType
from infrastructure.datasets import inspect_dataset
from infrastructure.repository import JsonExperimentRepository
from infrastructure.yolo_trainer import UltralyticsTrainer

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.getenv("HYPER_GUI_DATA", ROOT / "data"))
repository = JsonExperimentRepository(DATA_DIR / "experiments.json")
trainer = UltralyticsTrainer(DATA_DIR / "runs")
optimizer = HyperparameterOptimizer(trainer, repository)
executor = ThreadPoolExecutor(max_workers=int(os.getenv("HYPER_GUI_WORKERS", "1")))
cancel_events: dict[str, threading.Event] = {}

def enqueue_experiment(experiment: Experiment) -> None:
    """Queue an experiment once, including recovery after an app restart."""
    event = cancel_events.setdefault(experiment.id, threading.Event())
    executor.submit(optimizer.run, experiment, event.is_set)


@asynccontextmanager
async def lifespan(_: FastAPI):
    for experiment in repository.list():
        if experiment.status.value in ("queued", "running"):
            enqueue_experiment(experiment)
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


@app.get("/")
def index() -> FileResponse:
    return FileResponse(ROOT / "frontend" / "index.html")


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
