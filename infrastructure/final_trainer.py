"""Ultralytics adapter for final, full-budget training."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from domain.training import TrainingJob
from domain.naming import final_run_name
from infrastructure.yolo_trainer import UltralyticsTrainer


class UltralyticsFinalTrainer:
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir.resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def train(self, job: TrainingJob) -> tuple[dict[str, float], str, bool]:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("Ultralytics is not installed. Run: pip install -r requirements.txt") from exc

        run_name = job.run_name or final_run_name(job.name, job.id)
        job.run_name = run_name
        run_dir = self.output_dir / run_name
        recovery_checkpoint = run_dir / "weights" / "last.pt"
        if recovery_checkpoint.is_file():
            model = YOLO(str(recovery_checkpoint), task=job.task)
            result = model.train(resume=True, val=True)
            resumed = True
        else:
            if job.mode.value == "continue" and not job.source_weights:
                raise RuntimeError("the selected optimizer trial has no available weights/last.pt")
            source = job.source_weights if job.mode.value == "continue" else job.model
            if not source or (job.mode.value == "continue" and not Path(source).is_file()):
                raise RuntimeError(f"source weights do not exist: {source}")
            arguments: dict[str, Any] = {
                **job.hyperparameters,
                "data": job.dataset,
                "epochs": job.epochs,
                "imgsz": job.image_size,
                "batch": job.batch,
                "device": job.device,
                "project": str(self.output_dir),
                "name": run_name,
                "exist_ok": True,
                "val": True,
                "save": True,
                "plots": True,
            }
            if job.device in (None, "", "auto"):
                arguments.pop("device")
            model = YOLO(source, task=job.task)
            result = model.train(**arguments)
            resumed = False

        raw = getattr(result, "results_dict", {}) or {}
        metrics = UltralyticsTrainer._normalize_metrics(raw)
        save_dir = str(getattr(result, "save_dir", run_dir))
        return metrics, save_dir, resumed
