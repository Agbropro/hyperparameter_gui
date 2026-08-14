"""Open a video and display YOLO bounding-box predictions with OpenCV.

Edit the CONFIGURATION values below, then run:

    python tools/yolo_video_viewer.py

Keyboard controls:
    q or Esc  Quit
    Space     Pause/resume
    r         Restart a video file from the beginning

This deliberately has no argparse or command-line options.
"""

from __future__ import annotations

from pathlib import Path
from time import perf_counter

import cv2
from ultralytics import YOLO


# ---------------------------------------------------------------------------
# CONFIGURATION — edit these values before running the script.
# ---------------------------------------------------------------------------

MODEL_PATH = "/opt/model/hush_0.pt"

# Use a video path such as "/absolute/path/to/video.mp4".
# Use 0 for the default webcam, 1 for the second camera, and so on.
VIDEO_SOURCE: str | int = "/mnt/hush/swarm_1.mp4"

CONFIDENCE = 0.25
IOU = 0.70
IMAGE_SIZE = 640
DEVICE: str | int | None = None  # None = automatic, "cpu", 0, "0", etc.

WINDOW_TITLE = "YOLO Video Viewer"
DISPLAY_MAX_WIDTH: int | None = 1280
LOOP_VIDEO = False
BOX_THICKNESS = 2
SHOW_CONFIDENCE = True
SHOW_FPS = True


def validate_configuration() -> None:
    model_path = Path(MODEL_PATH).expanduser()
    if not model_path.is_file():
        raise FileNotFoundError(f"YOLO model was not found: {model_path}")
    if isinstance(VIDEO_SOURCE, str):
        video_path = Path(VIDEO_SOURCE).expanduser()
        if not video_path.is_file():
            raise FileNotFoundError(f"Video was not found: {video_path}")
    if not 0 <= CONFIDENCE <= 1:
        raise ValueError("CONFIDENCE must be between 0 and 1")
    if not 0 <= IOU <= 1:
        raise ValueError("IOU must be between 0 and 1")


def class_color(class_id: int) -> tuple[int, int, int]:
    """Return a stable, bright BGR color for a class number."""
    palette = (
        (255, 92, 121),
        (67, 255, 217),
        (75, 113, 255),
        (184, 255, 91),
        (255, 168, 74),
        (216, 77, 190),
        (39, 187, 226),
        (167, 117, 255),
    )
    return palette[class_id % len(palette)]


def class_name(names: dict | list, class_id: int) -> str:
    if isinstance(names, dict):
        return str(names.get(class_id, class_id))
    return str(names[class_id]) if class_id < len(names) else str(class_id)


def draw_detections(frame, result) -> int:
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return 0

    coordinates = boxes.xyxy.detach().cpu().numpy().astype(int)
    confidences = boxes.conf.detach().cpu().numpy()
    classes = boxes.cls.detach().cpu().numpy().astype(int)

    for (x1, y1, x2, y2), confidence, class_id in zip(
        coordinates, confidences, classes, strict=True
    ):
        color = class_color(int(class_id))
        label = class_name(result.names, int(class_id))
        if SHOW_CONFIDENCE:
            label = f"{label} {confidence:.2f}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, BOX_THICKNESS)
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1
        )
        label_top = max(0, y1 - text_height - baseline - 8)
        cv2.rectangle(
            frame,
            (x1, label_top),
            (x1 + text_width + 8, label_top + text_height + baseline + 8),
            color,
            -1,
        )
        cv2.putText(
            frame,
            label,
            (x1 + 4, label_top + text_height + 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (18, 19, 15),
            1,
            cv2.LINE_AA,
        )
    return len(boxes)


def fit_for_display(frame):
    if not DISPLAY_MAX_WIDTH or frame.shape[1] <= DISPLAY_MAX_WIDTH:
        return frame
    scale = DISPLAY_MAX_WIDTH / frame.shape[1]
    size = (DISPLAY_MAX_WIDTH, round(frame.shape[0] * scale))
    return cv2.resize(frame, size, interpolation=cv2.INTER_AREA)


def draw_status(frame, fps: float, detections: int, paused: bool) -> None:
    parts = [f"Objects: {detections}"]
    if SHOW_FPS:
        parts.append(f"Inference: {fps:.1f} FPS")
    if paused:
        parts.append("PAUSED")
    text = "  |  ".join(parts)
    cv2.putText(
        frame,
        text,
        (14, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        text,
        (14, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (18, 19, 15),
        1,
        cv2.LINE_AA,
    )


def main() -> None:
    validate_configuration()
    model = YOLO(str(Path(MODEL_PATH).expanduser()))
    source = str(Path(VIDEO_SOURCE).expanduser()) if isinstance(VIDEO_SOURCE, str) else VIDEO_SOURCE
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV could not open video source: {source}")

    cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)
    paused = False
    displayed_frame = None

    try:
        while True:
            if not paused:
                available, frame = capture.read()
                if not available:
                    if LOOP_VIDEO and isinstance(VIDEO_SOURCE, str):
                        capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    break

                started = perf_counter()
                predict_options = {
                    "source": frame,
                    "conf": CONFIDENCE,
                    "iou": IOU,
                    "imgsz": IMAGE_SIZE,
                    "verbose": False,
                }
                if DEVICE is not None:
                    predict_options["device"] = DEVICE
                result = model.predict(**predict_options)[0]
                elapsed = perf_counter() - started
                detections = draw_detections(frame, result)
                draw_status(frame, 1 / elapsed if elapsed else 0, detections, paused=False)
                displayed_frame = fit_for_display(frame)

            if displayed_frame is not None:
                cv2.imshow(WINDOW_TITLE, displayed_frame)

            key = cv2.waitKey(30 if paused else 1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord(" "):
                paused = not paused
            if key == ord("r") and isinstance(VIDEO_SOURCE, str):
                capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                paused = False
            if cv2.getWindowProperty(WINDOW_TITLE, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
