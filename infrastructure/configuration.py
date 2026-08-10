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


def load_server_settings(path: Path) -> ServerSettings:
    """Load and validate the ``server`` section of config.yaml."""
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

    return ServerSettings(host=host.strip(), port=port, reload=reload_enabled)
