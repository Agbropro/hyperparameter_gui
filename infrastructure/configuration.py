"""Application configuration loaded from the project YAML file."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ServerSettings:
    host: str = "127.0.0.1"
    port: int = 8000
    reload: bool = True


@dataclass(frozen=True)
class DatabaseSettings:
    wal_checkpoint_seconds: float = 10.0


@dataclass(frozen=True)
class ValidationInferenceSettings:
    mask_opacity: float = 0.25
    default_page_size: int = 10
    max_page_size: int = 50
    cache_version: int = 1
    asset_cache_seconds: int = 86400
    cache_retention_days: float = 30.0
    cache_max_size_gb: float = 10.0
    cache_cleanup_seconds: float = 3600.0


@dataclass(frozen=True)
class ApplicationSettings:
    server: ServerSettings
    database: DatabaseSettings
    validation_inference: ValidationInferenceSettings


def load_server_settings(path: Path) -> ServerSettings:
    """Load and validate the ``server`` section of config.yaml."""
    return load_application_settings(path).server


def load_application_settings(path: Path) -> ApplicationSettings:
    """Load and validate all application settings from config.yaml."""
    if not path.is_file():
        raise RuntimeError(f"configuration file not found: {path}")

    try:
        document: Any = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise RuntimeError(f"could not read configuration file {path}: {exc}") from exc

    if not isinstance(document, dict) or not isinstance(document.get("server"), dict):
        raise RuntimeError("config.yaml must contain a 'server' mapping")

    values = document["server"]
    host = values.get("host", "127.0.0.1")
    port = values.get("port", 8000)
    reload_enabled = values.get("reload", True)

    if not isinstance(host, str) or not host.strip():
        raise RuntimeError("server.host must be a non-empty string")
    if isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535:
        raise RuntimeError("server.port must be an integer between 1 and 65535")
    if not isinstance(reload_enabled, bool):
        raise RuntimeError("server.reload must be true or false")

    database_values = document.get("database", {})
    if not isinstance(database_values, dict):
        raise RuntimeError("database must be a mapping")
    checkpoint_seconds = database_values.get("wal_checkpoint_seconds", 10.0)
    if isinstance(checkpoint_seconds, bool) or not isinstance(checkpoint_seconds, (int, float)) or checkpoint_seconds <= 0:
        raise RuntimeError("database.wal_checkpoint_seconds must be a number greater than 0")

    inference_values = document.get("validation_inference", {})
    if not isinstance(inference_values, dict):
        raise RuntimeError("validation_inference must be a mapping")
    mask_opacity = inference_values.get("mask_opacity", 0.25)
    default_page_size = inference_values.get("default_page_size", 10)
    max_page_size = inference_values.get("max_page_size", 50)
    cache_version = inference_values.get("cache_version", 1)
    asset_cache_seconds = inference_values.get("asset_cache_seconds", 86400)
    cache_retention_days = inference_values.get("cache_retention_days", 30.0)
    cache_max_size_gb = inference_values.get("cache_max_size_gb", 10.0)
    cache_cleanup_seconds = inference_values.get("cache_cleanup_seconds", 3600.0)
    if isinstance(mask_opacity, bool) or not isinstance(mask_opacity, (int, float)) or not 0 <= mask_opacity <= 1:
        raise RuntimeError("validation_inference.mask_opacity must be a number between 0 and 1")
    for name, value in (("default_page_size", default_page_size), ("max_page_size", max_page_size)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise RuntimeError(f"validation_inference.{name} must be an integer greater than 0")
    if default_page_size > max_page_size:
        raise RuntimeError("validation_inference.default_page_size cannot exceed max_page_size")
    if isinstance(cache_version, bool) or not isinstance(cache_version, int) or cache_version < 1:
        raise RuntimeError("validation_inference.cache_version must be an integer greater than 0")
    if isinstance(asset_cache_seconds, bool) or not isinstance(asset_cache_seconds, int) or asset_cache_seconds < 0:
        raise RuntimeError("validation_inference.asset_cache_seconds must be an integer of 0 or greater")
    for name, value in (("cache_retention_days", cache_retention_days), ("cache_max_size_gb", cache_max_size_gb)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
            raise RuntimeError(f"validation_inference.{name} must be a number of 0 or greater")
    if isinstance(cache_cleanup_seconds, bool) or not isinstance(cache_cleanup_seconds, (int, float)) or cache_cleanup_seconds <= 0:
        raise RuntimeError("validation_inference.cache_cleanup_seconds must be a number greater than 0")

    return ApplicationSettings(
        server=ServerSettings(host=host.strip(), port=port, reload=reload_enabled),
        database=DatabaseSettings(wal_checkpoint_seconds=float(checkpoint_seconds)),
        validation_inference=ValidationInferenceSettings(
            mask_opacity=float(mask_opacity),
            default_page_size=default_page_size,
            max_page_size=max_page_size,
            cache_version=cache_version,
            asset_cache_seconds=asset_cache_seconds,
            cache_retention_days=float(cache_retention_days),
            cache_max_size_gb=float(cache_max_size_gb),
            cache_cleanup_seconds=float(cache_cleanup_seconds),
        ),
    )
