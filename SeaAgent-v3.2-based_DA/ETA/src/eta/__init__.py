"""ETA/DPD research package."""

from .config import ETA_ROOT, load_config

__all__ = ["ETA_ROOT", "build_model", "load_config"]


def __getattr__(name: str):
    if name == "build_model":
        from .model import build_model

        return build_model
    raise AttributeError(name)
