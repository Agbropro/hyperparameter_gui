"""ASGI entry point and config.yaml-driven development server."""

from pathlib import Path

from interfaces.api import app

__all__ = ["app"]


if __name__ == "__main__":
    import uvicorn

    from infrastructure.configuration import load_server_settings

    settings = load_server_settings(Path(__file__).with_name("config.yaml"))
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.reload,
    )
