"""Model registry — register backends by name and instantiate from config."""

from __future__ import annotations

from typing import Callable, Type

from .base import BaseAudioQAModel


MODEL_REGISTRY: dict[str, Type[BaseAudioQAModel]] = {}


def register_model(name: str) -> Callable[[Type[BaseAudioQAModel]], Type[BaseAudioQAModel]]:
    """Class decorator that registers a backend under `name`."""

    def _wrap(cls: Type[BaseAudioQAModel]) -> Type[BaseAudioQAModel]:
        if name in MODEL_REGISTRY:
            raise ValueError(f"model name already registered: {name}")
        cls.name = name
        MODEL_REGISTRY[name] = cls
        return cls

    return _wrap


def get_model(name: str, **kwargs) -> BaseAudioQAModel:
    """Instantiate a backend by registered name. Lazily imports built-in backends."""
    # Trigger built-in registrations the first time we look up a model.
    if not MODEL_REGISTRY:
        _bootstrap_builtin_models()
    if name not in MODEL_REGISTRY:
        # Try bootstrap once more in case the user passed an alias.
        _bootstrap_builtin_models()
    if name not in MODEL_REGISTRY:
        raise KeyError(
            f"unknown model '{name}'. Available: {sorted(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[name](**kwargs)


def _bootstrap_builtin_models() -> None:
    """Import built-in backends so their @register_model decorators run.
    Imports are wrapped in try/except so a missing optional dep does not block
    the rest of the registry."""
    try:
        from . import qwen_omni  # noqa: F401
    except Exception:  # noqa: BLE001
        pass
    try:
        from . import dummy  # noqa: F401
    except Exception:  # noqa: BLE001
        pass
    try:
        from . import api_placeholder  # noqa: F401
    except Exception:  # noqa: BLE001
        pass
