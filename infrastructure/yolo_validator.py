"""Ultralytics model.val adapter for held-out model comparison."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from domain.naming import safe_name
from domain.validation import ValidationJob
from infrastructure.yolo_trainer import UltralyticsTrainer


class UltralyticsModelValidator:
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir.resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def validate(self, job: ValidationJob, model_index: int) -> tuple[dict[str, float], list[dict[str, Any]], str]:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("Ultralytics is not installed. Run: pip install -r requirements.txt") from exc

        selected = job.models[model_index]
        model_path = Path(selected.model_path).expanduser().resolve()
        if not model_path.is_file():
            raise RuntimeError(f"model weights do not exist: {model_path}")
        run_name = f"{safe_name(job.name, 'validation')}-{job.id[:8]}-{model_index + 1:02d}-{safe_name(selected.label, 'model')}"
        model = YOLO(str(model_path))
        arguments: dict[str, Any] = {
            "data": job.dataset,
            "split": job.split,
            "conf": job.confidence,
            "iou": job.iou,
            "imgsz": job.image_size,
            "batch": job.batch,
            "project": str(self.output_dir),
            "name": run_name,
            "exist_ok": True,
            "plots": True,
            "save_json": False,
        }
        if job.device not in (None, "", "auto"):
            arguments["device"] = job.device
        result = model.val(**arguments)
        metrics = self._metrics(result)
        per_class = self._summary(result)
        save_dir = str(getattr(result, "save_dir", self.output_dir / run_name))
        return metrics, per_class, save_dir

    @staticmethod
    def _metrics(result: Any) -> dict[str, float]:
        raw = getattr(result, "results_dict", {}) or {}
        normalized = UltralyticsTrainer._normalize_metrics(raw)
        box = getattr(result, "box", None)
        if box is not None:
            for name, attr in (("precision", "mp"), ("recall", "mr"), ("map50", "map50"), ("map75", "map75"), ("map50_95", "map")):
                if hasattr(box, attr):
                    normalized[name] = float(getattr(box, attr))
        segment = getattr(result, "seg", None)
        if segment is not None:
            for name, attr in (("mask_precision", "mp"), ("mask_recall", "mr"), ("mask_map50", "map50"), ("mask_map75", "map75"), ("mask_map50_95", "map")):
                if hasattr(segment, attr):
                    normalized[name] = float(getattr(segment, attr))
            mask_precision, mask_recall = normalized.get("mask_precision"), normalized.get("mask_recall")
            if mask_precision is not None and mask_recall is not None and mask_precision + mask_recall:
                normalized["mask_f1"] = 2 * mask_precision * mask_recall / (mask_precision + mask_recall)
        top1 = getattr(result, "top1", None)
        top5 = getattr(result, "top5", None)
        if top1 is not None:
            normalized["accuracy_top1"] = float(top1)
        if top5 is not None:
            normalized["accuracy_top5"] = float(top5)
        precision, recall = normalized.get("precision"), normalized.get("recall")
        if precision is not None and recall is not None and precision + recall:
            normalized["f1"] = 2 * precision * recall / (precision + recall)
        speed = getattr(result, "speed", {}) or {}
        if "inference" in speed:
            normalized["inference_ms"] = float(speed["inference"])
        return normalized

    @staticmethod
    def _summary(result: Any) -> list[dict[str, Any]]:
        summary = getattr(result, "summary", None)
        if not callable(summary):
            return []
        try:
            rows = summary(normalize=True, decimals=5)
        except TypeError:
            rows = summary()
        return [_json_safe(row) for row in rows] if isinstance(rows, list) else []


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return value
