from pathlib import Path

import pytest

from domain.entities import TaskType
from infrastructure.datasets import inspect_dataset


def test_detect_dataset_discovers_yaml_and_folders(tmp_path: Path):
    (tmp_path / "data.yaml").write_text("path: .\n", encoding="utf-8")
    (tmp_path / "images" / "train").mkdir(parents=True)
    (tmp_path / "images" / "val").mkdir(parents=True)
    result = inspect_dataset(str(tmp_path), TaskType.DETECT)
    assert result["dataset"].endswith("data.yaml")
    assert set(result["folders"]) == {"train", "val"}


def test_classification_requires_train_and_evaluation_folder(tmp_path: Path):
    (tmp_path / "train").mkdir()
    with pytest.raises(ValueError, match="val or test"):
        inspect_dataset(str(tmp_path), TaskType.CLASSIFY)
    (tmp_path / "test").mkdir()
    assert inspect_dataset(str(tmp_path), TaskType.CLASSIFY)["valid"] is True
