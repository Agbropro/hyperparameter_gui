from interfaces.api import options


def test_model_versions_cover_every_supported_task_and_size():
    values = options()
    assert set(values["versions"]) == {"yolo26", "yolo11", "yolov8"}
    assert values["model_sizes"] == ["n", "s", "m", "l", "x"]
    assert set(values["metrics"]) == {"detect", "segment", "classify"}
