"""Auto-discover experiments from the `quant_lib/experiments/` package.

Each Python file in this package (except framework modules) is imported,
and is expected to register an ``ExperimentConfig`` via the ``@register``
decorator or by calling ``register(config)`` at module level.

User-facing experiment files live alongside the framework code in this
package, so a fresh ``pip install -e .`` makes them available immediately
without any sys.modules injection.

Failures are logged but don't abort (one bad experiment shouldn't
break the others).

Note
----
``discover_experiments()`` is idempotent. It runs once per process
by default (tracked by ``_DISCOVERED``). Tests can call ``reset()``
to force re-discovery.
"""
from __future__ import annotations

import importlib
import logging
import sys
from pathlib import Path


log = logging.getLogger(__name__)


# Framework modules inside ``quant_lib/experiments/`` that are not
# user experiment files. Discovery skips these.
_FRAMEWORK_MODULES: frozenset[str] = frozenset({
    "base",
    "registry",
    "built_in",
    "__init__",
})


_DISCOVERED: bool = False


def _experiments_dir() -> Path:
    """Return the ``quant_lib/experiments/`` directory containing this file."""
    return Path(__file__).resolve().parent


def _check_legacy_root_experiments() -> None:
    """If a legacy top-level ``experiments/`` directory exists, emit a
    one-time deprecation warning. The 0.3.0 refactor moved user
    experiment files into ``quant_lib/experiments/`` for standard
    Python packaging.
    """
    # Walk up two levels: built_in.py -> experiments/ -> quant_lib/ -> root
    project_root = Path(__file__).resolve().parent.parent.parent
    legacy = project_root / "experiments"
    if legacy.exists() and legacy.is_dir():
        log.warning(
            "Found legacy top-level directory %s. As of 0.3.0, user "
            "experiments live in quant_lib/experiments/. Move your "
            "custom *.py experiment files there, or they will not be "
            "auto-discovered.",
            legacy,
        )


def discover_experiments() -> None:
    """Import all .py files in the ``quant_lib/experiments/`` package.

    Idempotent: subsequent calls are no-ops unless ``reset()`` was called.

    Each module is expected to call ``register(config)`` (or use the
    ``@register`` decorator) at import time. Import failures (syntax
    error, missing dependency, etc.) are logged and other experiments
    continue to load.
    """
    global _DISCOVERED
    if _DISCOVERED:
        return

    _check_legacy_root_experiments()

    exp_dir = _experiments_dir()

    # Import each .py file in this package (except framework modules).
    for path in sorted(exp_dir.glob("*.py")):
        stem = path.stem
        if stem in _FRAMEWORK_MODULES:
            continue
        module_name = f"quant_lib.experiments.{stem}"
        try:
            if module_name in sys.modules:
                # Already imported: reload to re-execute module-level
                # @register calls (important for tests that reset and
                # re-discover).
                importlib.reload(sys.modules[module_name])
            else:
                importlib.import_module(module_name)
        except Exception as e:
            log.warning(
                f"Failed to import experiment '{path.name}': "
                f"{type(e).__name__}: {e}"
            )

    _DISCOVERED = True


def reset() -> None:
    """Reset discovery state (for tests). Next call to discover_experiments()
    will re-walk the experiments/ directory and re-import.
    """
    global _DISCOVERED
    _DISCOVERED = False
