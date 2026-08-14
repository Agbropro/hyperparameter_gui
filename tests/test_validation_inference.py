import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from domain.entities import ExperimentStatus
from domain.validation import ModelValidationResult, ValidationJob
from infrastructure.validation_inference import ValidationInferenceBrowser, read_test_dataset


def make_dataset(tmp_path: Path) -> tuple[Path, list[Path]]:
    image_dir = tmp_path / "images" / "test"
    label_dir = tmp_path / "labels" / "test"
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    images = [image_dir / "a.jpg", image_dir / "b.jpg"]
    for image in images:
        cv2.imwrite(str(image), np.zeros((100, 140, 3), dtype=np.uint8))
    (label_dir / "a.txt").write_text("0 0.5 0.5 0.4 0.4\n", encoding="utf-8")
    (label_dir / "b.txt").write_text("0 0.2 0.2 0.8 0.2 0.8 0.8 0.2 0.8\n", encoding="utf-8")
    dataset = tmp_path / "data.yaml"
    dataset.write_text(f"path: {tmp_path}\ntest: images/test\nnames: [person]\n", encoding="utf-8")
    return dataset, images


def make_completed_job(tmp_path: Path, dataset: Path) -> ValidationJob:
    weights = tmp_path / "model-seg.pt"
    weights.write_bytes(b"weights")
    return ValidationJob(
        name="Visual compare",
        dataset=str(dataset),
        split="test",
        confidence=0.25,
        iou=0.7,
        image_size=640,
        batch=8,
        device="cpu",
        models=[ModelValidationResult(label="Segment model", model_path=str(weights), status="completed")],
        id="visual-job",
        status=ExperimentStatus.COMPLETED,
    )


def test_inference_browser_pages_ground_truth_and_cached_predictions(tmp_path: Path, monkeypatch):
    dataset, images = make_dataset(tmp_path)
    job = make_completed_job(tmp_path, dataset)
    calls = []

    class FakeResult:
        def __init__(self, image_path):
            self.image_path = image_path

        def plot(self, **kwargs):
            image = cv2.imread(self.image_path)
            cv2.circle(image, (15, 15), 8, (0, 0, 255), -1)
            return image

    class FakeYOLO:
        task = "segment"

        def __init__(self, model_path):
            calls.append(("load", model_path))

        def predict(self, **kwargs):
            calls.append(("predict", kwargs))
            return (FakeResult(path) for path in kwargs["source"])

    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=FakeYOLO))
    browser = ValidationInferenceBrowser(tmp_path / "inference")

    first = browser.browse(job, page=1, page_size=1)
    assert first["task"] == "segment"
    assert first["total_images"] == 2
    assert first["total_pages"] == 2
    assert first["items"][0]["filename"] == images[0].name
    assert len(first["items"][0]["predictions"]) == 1
    assert browser.resolve_asset(job.id, first["items"][0]["ground_truth_url"].split("/")[-3], "ground-truth/0000001.jpg").is_file()

    browser.browse(job, page=1, page_size=1)
    assert [call[0] for call in calls].count("predict") == 1

    second = browser.browse(job, page=2, page_size=1)
    assert second["page"] == 2
    ground_truth_url = second["items"][0]["ground_truth_url"]
    cache_key = ground_truth_url.split("/")[-3]
    mask_image = cv2.imread(str(browser.resolve_asset(job.id, cache_key, "ground-truth/0000002.jpg")))
    assert mask_image[50, 70].any()


def test_reads_test_images_and_names_from_dataset_yaml(tmp_path: Path):
    dataset, images = make_dataset(tmp_path)
    loaded, names = read_test_dataset(dataset)
    assert loaded == images
    assert names == {0: "person"}
