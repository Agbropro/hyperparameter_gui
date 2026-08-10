"""ASGI entry point. Start with: uvicorn main:app --reload"""

from interfaces.api import app

__all__ = ["app"]
