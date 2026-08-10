"""Read completed optimizer experiments from a user-selected JSON file."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


def read_experiment_file(value: str) -> list[dict[str, Any]]:
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"experiments JSON does not exist: {path}")
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read experiments JSON: {exc}") from exc
    if not isinstance(document, dict):
        raise ValueError("experiments JSON must contain an object keyed by experiment ID")

    imported: list[dict[str, Any]] = []
    for fallback_id, raw in document.items():
        if not isinstance(raw, dict) or not isinstance(raw.get("config"), dict):
            continue
        trials = raw.get("trials", [])
        best_number = raw.get("best_trial")
        best = next((trial for trial in trials if trial.get("number") == best_number), None)
        if not best:
            continue
        config = raw["config"]
        raw_run_directory = best.get("run_directory")
        run_dir = Path(raw_run_directory) if raw_run_directory else None
        last_weights = run_dir / "weights" / "last.pt" if run_dir else None
        best_weights = run_dir / "weights" / "best.pt" if run_dir else None
        splits = read_dataset_splits(config.get("dataset", ""), config.get("task", "detect"))
        imported.append(
            {
                "id": raw.get("id", fallback_id),
                "name": config.get("name", fallback_id),
                "status": raw.get("status"),
                "task": config.get("task"),
                "model": config.get("model"),
                "dataset": config.get("dataset"),
                "image_size": config.get("image_size", 640),
                "device": config.get("device"),
                "best_trial": best_number,
                "metrics": best.get("metrics", {}),
                "score": best.get("score"),
                "hyperparameters": best.get("hyperparameters", {}),
                "run_directory": raw_run_directory,
                "last_weights": str(last_weights) if last_weights and last_weights.is_file() else None,
                "best_weights": str(best_weights) if best_weights and best_weights.is_file() else None,
                "dataset_splits": splits,
                "source_path": str(path),
            }
        )
    if not imported:
        raise ValueError("no experiment with a completed best trial was found")
    return imported


def read_dataset_splits(dataset: str, task: str) -> dict[str, Any]:
    path = Path(dataset).expanduser()
    if task == "classify" or not path.is_file():
        return {"train": "train", "val": "val", "test": "test (reserved)"} if task == "classify" else {}
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return {}
    return {key: document.get(key) for key in ("train", "val", "test") if key in document}


def get_imported_experiment(path: str, experiment_id: str) -> dict[str, Any]:
    experiment = next((item for item in read_experiment_file(path) if item["id"] == experiment_id), None)
    if not experiment:
        raise ValueError("selected experiment was not found in that JSON file")
    return experiment
