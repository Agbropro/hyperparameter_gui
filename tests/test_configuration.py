from pathlib import Path

import pytest

from infrastructure.configuration import (
    DatabaseSettings,
    ServerSettings,
    ValidationInferenceSettings,
    load_application_settings,
    load_server_settings,
)


def test_load_server_settings(tmp_path: Path):
    path = tmp_path / "config.yaml"
    path.write_text("server:\n  host: 0.0.0.0\n  port: 9010\n  reload: false\n", encoding="utf-8")
    assert load_server_settings(path) == ServerSettings(host="0.0.0.0", port=9010, reload=False)


def test_loads_database_and_validation_inference_settings(tmp_path: Path):
    path = tmp_path / "config.yaml"
    path.write_text(
        """server:
  host: 127.0.0.1
  port: 8000
database:
  wal_checkpoint_seconds: 15
validation_inference:
  mask_opacity: 0.35
  default_page_size: 12
  max_page_size: 40
  cache_version: 4
  asset_cache_seconds: 120
  cache_retention_days: 14
  cache_max_size_gb: 5
  cache_cleanup_seconds: 600
""",
        encoding="utf-8",
    )
    settings = load_application_settings(path)
    assert settings.database == DatabaseSettings(wal_checkpoint_seconds=15.0)
    assert settings.validation_inference == ValidationInferenceSettings(
        mask_opacity=0.35,
        default_page_size=12,
        max_page_size=40,
        cache_version=4,
        asset_cache_seconds=120,
        cache_retention_days=14.0,
        cache_max_size_gb=5.0,
        cache_cleanup_seconds=600.0,
    )


@pytest.mark.parametrize(
    ("section", "message"),
    [
        ("mask_opacity: 1.5", "mask_opacity"),
        ("default_page_size: 20\n  max_page_size: 10", "cannot exceed"),
        ("cache_version: 0", "cache_version"),
    ],
)
def test_rejects_invalid_validation_inference_settings(tmp_path: Path, section: str, message: str):
    path = tmp_path / "config.yaml"
    path.write_text(
        f"server:\n  host: 127.0.0.1\n  port: 8000\nvalidation_inference:\n  {section}\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match=message):
        load_application_settings(path)


@pytest.mark.parametrize("port", [0, 65536, "eight-thousand", True])
def test_rejects_invalid_port(tmp_path: Path, port):
    path = tmp_path / "config.yaml"
    path.write_text(f"server:\n  host: 127.0.0.1\n  port: {str(port).lower()}\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="server.port"):
        load_server_settings(path)
