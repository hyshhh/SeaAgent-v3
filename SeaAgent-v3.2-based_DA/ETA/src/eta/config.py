"""Configuration loading and path resolution for the isolated ETA project."""

from __future__ import annotations

import copy
import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ETA_ROOT = Path(__file__).resolve().parents[2]


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = yaml.safe_load(stream) or {}
    if not isinstance(value, dict):
        raise TypeError(f"Configuration must be a mapping: {path}")
    return value


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load one configuration file on top of configs/default.yaml."""
    base_path = ETA_ROOT / "configs" / "default.yaml"
    config = _read_yaml(base_path)
    if path is not None:
        override_path = Path(path)
        if not override_path.is_absolute():
            override_path = ETA_ROOT / override_path
        if override_path.resolve() != base_path.resolve():
            config = _deep_merge(config, _read_yaml(override_path))
    config["paths"] = dict(config["paths"])
    config["paths"]["eta_root"] = str(ETA_ROOT)
    source_root = Path(str(config["paths"]["source_data_root"]))
    if not source_root.is_absolute():
        source_root = (ETA_ROOT / source_root).resolve()
    config["paths"]["source_data_root"] = str(source_root)
    for key in (
        "processed_dir",
        "embedding_cache_dir",
        "artifact_dir",
        "report_dir",
    ):
        item = Path(str(config["paths"][key]))
        if not item.is_absolute():
            item = ETA_ROOT / item
        config["paths"][key] = str(item.resolve())
    return config


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    except ImportError:
        pass


def resolve_source_path(config: dict[str, Any], relative: str | Path) -> Path:
    path = Path(relative)
    return path if path.is_absolute() else Path(config["paths"]["source_data_root"]) / path
