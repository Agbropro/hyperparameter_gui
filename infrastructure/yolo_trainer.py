"""Ultralytics implementation of the model-training port."""

from __future__ import annotations

from pathlib import Path
from typing import Any


class UltralyticsTrainer:
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir.resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def train(
        self,
        *,
        task: str,
        model: str,
        dataset: str,
        epochs: int,
        image_size: int,
        device: str | int | None,
        hyperparameters: dict[str, Any],
        run_name: str,
    ) -> tuple[dict[str, float], str | None]:
        try:
            import yaml
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "Ultralytics is not installed. Run: pip install -r requirements.txt"
            ) from exc

        trial_dir = self.output_dir / run_name
        trial_dir.mkdir(parents=True, exist_ok=True)
        (trial_dir / "hyperparameters.yaml").write_text(
            yaml.safe_dump(hyperparameters, sort_keys=True), encoding="utf-8"
        )

        training_args: dict[str, Any] = {
            "data": dataset,
            "epochs": epochs,
            "imgsz": image_size,
            "project": str(self.output_dir),
            "name": run_name,
            "exist_ok": True,
            **hyperparameters,
        }
        if device not in (None, "", "auto"):
            training_args["device"] = device

        checkpoint = trial_dir / "weights" / "last.pt"
        if checkpoint.is_file():
            # Ultralytics restores the epoch, optimizer, scheduler, and original
            # training arguments from last.pt. Passing only resume avoids
            # accidentally changing the interrupted run's configuration.
            yolo = YOLO(str(checkpoint), task=task)
            result = yolo.train(resume=True)
        else:
            yolo = YOLO(model, task=task)
            result = yolo.train(**training_args)
        raw = getattr(result, "results_dict", {}) or {}
        metrics = self._normalize_metrics(raw)
        save_dir = getattr(result, "save_dir", trial_dir)
        return metrics, str(save_dir)

    @staticmethod
    def _normalize_metrics(raw: dict[str, Any]) -> dict[str, float]:
        aliases = {
            "metrics/precision(B)": "precision",
            "metrics/precision(M)": "precision",
            "metrics/recall(B)": "recall",
            "metrics/recall(M)": "recall",
            "metrics/mAP50(B)": "map50",
            "metrics/mAP50(M)": "map50",
            "metrics/mAP50-95(B)": "map50_95",
            "metrics/mAP50-95(M)": "map50_95",
            "metrics/accuracy_top1": "accuracy_top1",
            "metrics/accuracy_top5": "accuracy_top5",
            "fitness": "fitness",
        }
        normalized: dict[str, float] = {}
        for source, destination in aliases.items():
            if source in raw:
                value = raw[source]
                normalized[destination] = float(value.item() if hasattr(value, "item") else value)
        precision, recall = normalized.get("precision"), normalized.get("recall")
        if precision is not None and recall is not None and precision + recall:
            normalized["f1"] = 2 * precision * recall / (precision + recall)
        return normalized
