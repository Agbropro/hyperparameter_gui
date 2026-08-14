"""On-demand, cached visual comparison for held-out validation images."""

from __future__ import annotations

import hashlib
import json
import math
import threading
from pathlib import Path
from typing import Any

import yaml

from domain.validation import ValidationJob


IMAGE_SUFFIXES = {".bmp", ".dng", ".jpeg", ".jpg", ".mpo", ".png", ".tif", ".tiff", ".webp"}
MASK_ALPHA = 0.16
RENDERER_VERSION = 2


class ValidationInferenceBrowser:
    """Render ground truth and model predictions only for requested pages."""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir.resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def browse(self, job: ValidationJob, page: int, page_size: int) -> dict[str, Any]:
        images, names = read_test_dataset(Path(job.dataset))
        if not images:
            raise ValueError("the dataset test split contains no supported image files")
        total_pages = math.ceil(len(images) / page_size)
        if page > total_pages:
            raise ValueError(f"page must be between 1 and {total_pages}")
        start = (page - 1) * page_size
        selected_images = images[start : start + page_size]
        completed_models = [(index, model) for index, model in enumerate(job.models) if model.status == "completed"]
        if not completed_models:
            raise ValueError("this validation has no completed models to visualize")

        cache_key = self._cache_key(job, images)
        cache_root = self.output_dir / job.id / cache_key
        task = "detect"
        with self._lock:
            ground_truth, has_segments = self._render_ground_truth(
                cache_root, selected_images, names, start
            )
            if has_segments:
                task = "segment"
            predictions: dict[int, list[Path]] = {}
            for model_index, model_result in completed_models:
                rendered, model_task = self._render_predictions(
                    cache_root,
                    selected_images,
                    start,
                    job,
                    model_index,
                    model_result.model_path,
                )
                predictions[model_index] = rendered
                if model_task == "segment":
                    task = "segment"

        items = []
        for offset, image_path in enumerate(selected_images):
            items.append(
                {
                    "index": start + offset + 1,
                    "filename": image_path.name,
                    "ground_truth_url": self._asset_url(job.id, cache_key, ground_truth[offset], cache_root),
                    "predictions": [
                        {
                            "label": model_result.label,
                            "url": self._asset_url(
                                job.id, cache_key, predictions[model_index][offset], cache_root
                            ),
                        }
                        for model_index, model_result in completed_models
                    ],
                }
            )
        return {
            "job_id": job.id,
            "task": task,
            "page": page,
            "page_size": page_size,
            "total_images": len(images),
            "total_pages": total_pages,
            "items": items,
        }

    def resolve_asset(self, job_id: str, cache_key: str, asset_path: str) -> Path:
        root = (self.output_dir / job_id / cache_key).resolve()
        candidate = (root / asset_path).resolve()
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise ValueError("invalid inference asset path") from exc
        if not candidate.is_file() or candidate.suffix.lower() != ".jpg":
            raise FileNotFoundError("inference image was not found")
        return candidate

    def _render_ground_truth(
        self,
        cache_root: Path,
        images: list[Path],
        names: dict[int, str],
        start: int,
    ) -> tuple[list[Path], bool]:
        paths = [cache_root / "ground-truth" / f"{start + offset + 1:07d}.jpg" for offset in range(len(images))]
        has_segments = False
        for image_path, output_path in zip(images, paths, strict=True):
            if output_path.is_file():
                has_segments = has_segments or _label_contains_segments(label_path_for(image_path))
                continue
            segmented = _draw_ground_truth(image_path, label_path_for(image_path), names, output_path)
            has_segments = has_segments or segmented
        return paths, has_segments

    def _render_predictions(
        self,
        cache_root: Path,
        images: list[Path],
        start: int,
        job: ValidationJob,
        model_index: int,
        model_path: str,
    ) -> tuple[list[Path], str]:
        directory = cache_root / f"model-{model_index + 1:02d}"
        task_file = directory / "task.txt"
        paths = [directory / f"{start + offset + 1:07d}.jpg" for offset in range(len(images))]
        missing = [(image, output) for image, output in zip(images, paths, strict=True) if not output.is_file()]
        if not missing:
            cached_task = task_file.read_text(encoding="utf-8").strip() if task_file.is_file() else "detect"
            return paths, cached_task

        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("Ultralytics is not installed. Run: pip install -r requirements.txt") from exc
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("OpenCV is not installed. Run: pip install -r requirements.txt") from exc

        model = YOLO(model_path)
        model_task = str(getattr(model, "task", "detect"))
        if model_task not in ("detect", "segment"):
            raise RuntimeError(f"visual inference supports detection and segmentation models, not {model_task}")
        arguments: dict[str, Any] = {
            "source": [str(image) for image, _ in missing],
            "conf": job.confidence,
            "iou": job.iou,
            "imgsz": job.image_size,
            "batch": min(job.batch, len(missing)),
            "stream": True,
            "verbose": False,
        }
        if job.device not in (None, "", "auto"):
            arguments["device"] = job.device
        results = model.predict(**arguments)
        count = 0
        for result, (_, output_path) in zip(results, missing, strict=True):
            output_path.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(output_path), _plot_prediction(result)):
                raise RuntimeError(f"could not write inference image: {output_path}")
            count += 1
        if count != len(missing):
            raise RuntimeError(f"model returned {count} predictions for {len(missing)} images")
        task_file.write_text(model_task, encoding="utf-8")
        return paths, model_task

    @staticmethod
    def _cache_key(job: ValidationJob, images: list[Path]) -> str:
        model_files = []
        for model in job.models:
            path = Path(model.model_path)
            model_files.append((str(path), path.stat().st_mtime_ns if path.is_file() else None))
        dataset = Path(job.dataset)
        payload = {
            "dataset": str(dataset),
            "dataset_mtime": dataset.stat().st_mtime_ns if dataset.is_file() else None,
            "models": model_files,
            "confidence": job.confidence,
            "iou": job.iou,
            "image_size": job.image_size,
            "renderer_version": RENDERER_VERSION,
        }
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode())
        for image in images:
            label = label_path_for(image)
            for path in (image, label):
                stat = path.stat() if path.is_file() else None
                digest.update(str(path).encode())
                digest.update(str((stat.st_mtime_ns, stat.st_size) if stat else None).encode())
        return digest.hexdigest()[:16]

    @staticmethod
    def _asset_url(job_id: str, cache_key: str, path: Path, root: Path) -> str:
        relative = path.relative_to(root).as_posix()
        return f"/api/validation/jobs/{job_id}/inference/assets/{cache_key}/{relative}"


def read_test_dataset(dataset_yaml: Path) -> tuple[list[Path], dict[int, str]]:
    try:
        document = yaml.safe_load(dataset_yaml.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"could not read dataset YAML: {exc}") from exc
    if not isinstance(document, dict) or not document.get("test"):
        raise ValueError("dataset YAML must define a test split")
    configured_root = Path(str(document.get("path", dataset_yaml.parent))).expanduser()
    root = configured_root if configured_root.is_absolute() else dataset_yaml.parent / configured_root
    root = root.resolve()
    entries = document["test"] if isinstance(document["test"], list) else [document["test"]]
    images: list[Path] = []
    for entry in entries:
        images.extend(_expand_image_entry(root, str(entry)))
    unique = sorted(dict.fromkeys(path.resolve() for path in images), key=lambda path: str(path))
    raw_names = document.get("names", {})
    if isinstance(raw_names, list):
        names = {index: str(name) for index, name in enumerate(raw_names)}
    elif isinstance(raw_names, dict):
        names = {int(index): str(name) for index, name in raw_names.items()}
    else:
        names = {}
    return unique, names


def _expand_image_entry(root: Path, entry: str) -> list[Path]:
    candidate = Path(entry).expanduser()
    candidate = candidate if candidate.is_absolute() else root / candidate
    if candidate.is_dir():
        return [path for path in candidate.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES]
    if candidate.is_file() and candidate.suffix.lower() in IMAGE_SUFFIXES:
        return [candidate]
    if candidate.is_file() and candidate.suffix.lower() == ".txt":
        return _images_from_list(candidate, root)
    if any(character in entry for character in "*?["):
        return [path for path in root.glob(entry) if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES]
    raise ValueError(f"test split entry does not exist or is unsupported: {candidate}")


def _images_from_list(list_path: Path, dataset_root: Path) -> list[Path]:
    images = []
    for raw in list_path.read_text(encoding="utf-8").splitlines():
        value = raw.strip()
        if not value:
            continue
        path = Path(value).expanduser()
        if not path.is_absolute():
            root_candidate = dataset_root / path
            path = root_candidate if root_candidate.exists() else list_path.parent / path
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            images.append(path)
    return images


def label_path_for(image_path: Path) -> Path:
    parts = list(image_path.parts)
    image_indices = [index for index, part in enumerate(parts) if part.lower() == "images"]
    if image_indices:
        parts[image_indices[-1]] = "labels"
        return Path(*parts).with_suffix(".txt")
    return image_path.parent.parent / "labels" / image_path.parent.name / f"{image_path.stem}.txt"


def _label_contains_segments(label_path: Path) -> bool:
    if not label_path.is_file():
        return False
    return any(len(line.split()) >= 7 for line in label_path.read_text(encoding="utf-8").splitlines())


def _draw_ground_truth(image_path: Path, label_path: Path, names: dict[int, str], output_path: Path) -> bool:
    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("OpenCV and NumPy are required to render validation images") from exc
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"could not read test image: {image_path}")
    height, width = image.shape[:2]
    segmented = False
    annotations: list[tuple[int, tuple[int, int, int, int]]] = []
    if label_path.is_file():
        for raw_line in label_path.read_text(encoding="utf-8").splitlines():
            fields = raw_line.split()
            if not fields:
                continue
            try:
                class_id = int(float(fields[0]))
                values = [float(value) for value in fields[1:]]
            except ValueError:
                continue
            color = _class_color(class_id)
            if len(values) == 4:
                center_x, center_y, box_width, box_height = values
                x1 = round((center_x - box_width / 2) * width)
                y1 = round((center_y - box_height / 2) * height)
                x2 = round((center_x + box_width / 2) * width)
                y2 = round((center_y + box_height / 2) * height)
                annotations.append((class_id, (x1, y1, x2, y2)))
            elif len(values) >= 6 and len(values) % 2 == 0:
                segmented = True
                points = np.array(
                    [[round(values[index] * width), round(values[index + 1] * height)] for index in range(0, len(values), 2)],
                    dtype=np.int32,
                )
                overlay = image.copy()
                cv2.fillPoly(overlay, [points], color)
                cv2.addWeighted(overlay, MASK_ALPHA, image, 1 - MASK_ALPHA, 0, image)
                x, y, box_width, box_height = cv2.boundingRect(points)
                annotations.append((class_id, (x, y, x + box_width, y + box_height)))
            else:
                continue
    image = _draw_box_labels(image, annotations, names)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), image):
        raise RuntimeError(f"could not write ground-truth image: {output_path}")
    return segmented


def _draw_box_labels(
    image: Any,
    annotations: list[tuple[int, tuple[int, int, int, int]]],
    names: dict[int, str],
) -> Any:
    try:
        from ultralytics.utils.plotting import Annotator

        annotator = Annotator(image, line_width=None, example=str(names))
        for class_id, box in annotations:
            annotator.box_label(box, names.get(class_id, str(class_id)), color=_class_color(class_id))
        return annotator.result()
    except (ImportError, AttributeError):
        pass

    import cv2

    thickness = max(2, round(image.shape[1] / 500))
    for class_id, (x1, y1, x2, y2) in annotations:
        color = _class_color(class_id)
        label = names.get(class_id, str(class_id))
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
        (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        top = max(0, y1 - text_height - baseline - 7)
        cv2.rectangle(image, (x1, top), (x1 + text_width + 8, top + text_height + baseline + 7), color, -1)
        cv2.putText(image, label, (x1 + 4, top + text_height + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return image


def _plot_prediction(result: Any) -> Any:
    masks = getattr(result, "masks", None)
    original = getattr(result, "orig_img", None)
    boxes = getattr(result, "boxes", None)
    if masks is None or original is None:
        return result.plot(boxes=True, masks=True, labels=True, conf=True)

    import cv2
    import numpy as np

    image = original.copy()
    overlay = image.copy()
    polygons = getattr(masks, "xy", []) or []
    class_ids = boxes.cls.detach().cpu().numpy().astype(int) if boxes is not None else np.zeros(len(polygons), dtype=int)
    for polygon, class_id in zip(polygons, class_ids):
        points = np.asarray(polygon, dtype=np.int32)
        if len(points) >= 3:
            cv2.fillPoly(overlay, [points], _class_color(int(class_id)))
    cv2.addWeighted(overlay, MASK_ALPHA, image, 1 - MASK_ALPHA, 0, image)
    return result.plot(img=image, boxes=True, masks=False, labels=True, conf=True)


def _class_color(class_id: int) -> tuple[int, int, int]:
    try:
        from ultralytics.utils.plotting import colors

        return tuple(int(value) for value in colors(class_id, bgr=True))
    except (ImportError, AttributeError):
        pass
    palette = ((255, 92, 121), (67, 255, 217), (75, 113, 255), (184, 255, 91), (255, 168, 74), (216, 77, 190))
    return palette[class_id % len(palette)]
