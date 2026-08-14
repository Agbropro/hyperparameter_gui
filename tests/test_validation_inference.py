import os
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from domain.entities import ExperimentStatus
from domain.validation import ModelValidationResult, ValidationJob
from infrastructure.validation_inference import ValidationInferenceBrowser, _plot_prediction, read_test_dataset


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
    browser = ValidationInferenceBrowser(
        tmp_path / "inference",
        mask_opacity=0.25,
        cache_version=1,
        cache_retention_days=30,
        cache_max_size_gb=10,
    )

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


def test_prediction_masks_use_light_overlay_before_box_plotting():
    class FakeTensor:
        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return np.array([0])

    class FakeResult:
        orig_img = np.zeros((100, 100, 3), dtype=np.uint8)
        masks = SimpleNamespace(xy=[np.array([[20, 20], [80, 20], [80, 80], [20, 80]])])
        boxes = SimpleNamespace(cls=FakeTensor())

        def plot(self, **kwargs):
            assert kwargs["masks"] is False
            return kwargs["img"]

    mask_opacity = 0.25
    plotted = _plot_prediction(FakeResult(), mask_opacity)
    assert plotted[50, 50].max() > 0
    assert plotted[50, 50].max() <= round(255 * mask_opacity) + 1


def test_cache_cleanup_removes_expired_directories(tmp_path: Path):
    root = tmp_path / "inference"
    browser = ValidationInferenceBrowser(
        root,
        mask_opacity=0.25,
        cache_version=1,
        cache_retention_days=1,
        cache_max_size_gb=0,
    )
    old_cache = root / "job-old" / "cache-old"
    recent_cache = root / "job-new" / "cache-new"
    for cache in (old_cache, recent_cache):
        cache.mkdir(parents=True)
        (cache / "image.jpg").write_bytes(b"jpeg")
    now = 2_000_000.0
    os.utime(old_cache, (now - 172800, now - 172800))
    os.utime(recent_cache, (now - 60, now - 60))

    result = browser.cleanup(now=now)

    assert result == {"removed_directories": 1, "removed_bytes": 4}
    assert not old_cache.exists()
    assert recent_cache.exists()


def test_cache_cleanup_enforces_size_oldest_first_and_keeps_newest(tmp_path: Path):
    root = tmp_path / "inference"
    browser = ValidationInferenceBrowser(
        root,
        mask_opacity=0.25,
        cache_version=1,
        cache_retention_days=0,
        cache_max_size_gb=10 / 1024**3,
    )
    caches = [root / f"job-{index}" / "cache" for index in range(3)]
    for index, cache in enumerate(caches):
        cache.mkdir(parents=True)
        (cache / "image.jpg").write_bytes(b"12345678")
        os.utime(cache, (100 + index, 100 + index))

    result = browser.cleanup(now=1000)

    assert result["removed_directories"] == 2
    assert not caches[0].exists()
    assert not caches[1].exists()
    assert caches[2].exists()
