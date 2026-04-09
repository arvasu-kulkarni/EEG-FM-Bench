from __future__ import annotations

from copy import deepcopy
from typing import Any

from pydantic import BaseModel


_RUNTIME_SYNTHETIC_DATASETS: dict[str, dict[str, Any]] = {}


def register_runtime_synthetic_datasets(configs: dict[str, Any] | None) -> None:
    global _RUNTIME_SYNTHETIC_DATASETS

    if not configs:
        _RUNTIME_SYNTHETIC_DATASETS = {}
        return

    normalized: dict[str, dict[str, Any]] = {}
    for name, cfg in configs.items():
        if isinstance(cfg, BaseModel):
            normalized[name] = cfg.model_dump()
        elif isinstance(cfg, dict):
            normalized[name] = deepcopy(cfg)
        else:
            raise TypeError(
                f"Unsupported runtime synthetic dataset config for {name!r}: "
                f"{type(cfg).__name__}"
            )

    _RUNTIME_SYNTHETIC_DATASETS = normalized


def get_runtime_synthetic_dataset(name: str) -> dict[str, Any] | None:
    cfg = _RUNTIME_SYNTHETIC_DATASETS.get(name)
    if cfg is None:
        return None
    return deepcopy(cfg)


def is_runtime_synthetic_dataset(name: str) -> bool:
    return name in _RUNTIME_SYNTHETIC_DATASETS
