"""Experiment registry with @register decorator.

The registry is a process-global dict mapping experiment name to
ExperimentConfig. Use ``register()`` to add, ``get()`` to retrieve.
"""

from __future__ import annotations

import logging
from typing import Callable

from .base import ExperimentConfig

log = logging.getLogger(__name__)


_REGISTRY: dict[str, "ExperimentConfig"] = {}


def register(config_or_fn: "ExperimentConfig | Callable[[], ExperimentConfig]") -> "ExperimentConfig":
    """Register an experiment config, or use as a decorator.

    Usage as a function (after constructing the config):

        cfg = ExperimentConfig(...)
        register(cfg)

    Usage as a decorator (in user-facing experiment files):

        @register
        def my_experiment() -> ExperimentConfig:
            return ExperimentConfig(...)

    Parameters
    ----------
    config_or_fn : ExperimentConfig or callable
        If an ``ExperimentConfig`` instance, register it directly.
        If callable (e.g., a function), call it to get the config
        (decorator pattern).

    Returns
    -------
    ExperimentConfig
        The registered config (for chaining/decorator use).

    Raises
    ------
    TypeError
        If ``config_or_fn`` is neither an ``ExperimentConfig`` nor callable.
    """
    if isinstance(config_or_fn, ExperimentConfig):
        config = config_or_fn
    elif callable(config_or_fn):
        # Decorator usage: call the function to get the config
        config = config_or_fn()
    else:
        raise TypeError(
            f"register() expects ExperimentConfig or callable, "
            f"got {type(config_or_fn).__name__}"
        )

    name = config.name
    if name in _REGISTRY:
        log.warning(
            f"Experiment '{name}' already registered. Overwriting. "
            f"Previous hypothesis: {_REGISTRY[name].hypothesis.mechanism[:60]}..."
        )
    _REGISTRY[name] = config
    log.debug(f"Registered experiment: {name}")
    return config


def get(name: str) -> "ExperimentConfig":
    """Get an experiment by name.

    Raises
    ------
    KeyError
        If the experiment is not found. Error message lists available
        experiments for easy debugging.
    """
    if name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY)) or "(none registered)"
        raise KeyError(f"Experiment '{name}' not found. Available: {available}")
    return _REGISTRY[name]


def all_experiments() -> list["ExperimentConfig"]:
    """List all registered experiments, sorted by name."""
    return [_REGISTRY[name] for name in sorted(_REGISTRY)]


def exists(name: str) -> bool:
    """Check if an experiment is registered."""
    return name in _REGISTRY


def clear() -> None:
    """Clear the registry. Used by tests for isolation."""
    _REGISTRY.clear()


def count() -> int:
    """Number of registered experiments."""
    return len(_REGISTRY)
