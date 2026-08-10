"""Dataset inspection and YOLO data-file discovery."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from domain.entities import TaskType


def inspect_dataset(value: str, task: TaskType) -> dict[str, Any]:
    path = Path(value).expanduser().resolve()
    if not path.exists():
        raise ValueError(f"dataset does not exist: {path}")

    if task in (TaskType.DETECT, TaskType.SEGMENT):
        yaml_path = path if path.is_file() else _find_yaml(path)
        if not yaml_path:
            raise ValueError("detection/segmentation datasets need a data.yaml file")
        root = yaml_path.parent
        folders = _folder_candidates(root)
        return {"dataset": str(yaml_path), "root": str(root), "folders": folders, "valid": True}

    if not path.is_dir():
        raise ValueError("classification dataset must be a directory")
    folders = _folder_candidates(path)
    if "train" not in folders:
        raise ValueError("classification dataset needs a train folder")
    if "val" not in folders and "test" not in folders:
        raise ValueError("classification dataset needs a val or test folder")
    return {"dataset": str(path), "root": str(path), "folders": folders, "valid": True}


def _find_yaml(root: Path) -> Path | None:
    preferred = [root / "data.yaml", root / "dataset.yaml", root / "data.yml"]
    for candidate in preferred:
        if candidate.is_file():
            return candidate
    return next(iter(sorted(root.glob("*.yaml"))), None)


def _folder_candidates(root: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    aliases = {"train": ("train",), "val": ("val", "valid", "validation"), "test": ("test",)}
    for name, candidates in aliases.items():
        for candidate in candidates:
            direct = root / candidate
            images = root / "images" / candidate
            if direct.is_dir():
                result[name] = str(direct)
                break
            if images.is_dir():
                result[name] = str(images)
                break
    return result
