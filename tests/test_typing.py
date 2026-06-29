"""Verify the ``py.typed`` marker and inspect public API type hints.

The framework advertises type information via the ``py.typed`` PEP 561
marker.  These tests confirm:

1. The marker file exists at the package root (length 0 — that's
   correct: any content is fine, but a zero-length file is the
   convention).
2. The public API exports are importable.
3. The public functions have type annotations on their parameters
   and return values (the framework advertises types; the
   annotations should exist).

This is a "smoke test" for the type system.  Full type checking
(``mypy --strict`` / ``pyright``) is out of scope for this test
suite — that would be a separate CI workflow.
"""
from __future__ import annotations

import importlib
import inspect
from pathlib import Path


import quant_lib


# ═══════════════════════════════════════════════════════════════════════
# py.typed marker
# ═══════════════════════════════════════════════════════════════════════


class TestPyTypedMarker:
    """The ``py.typed`` PEP 561 marker must exist at the package root."""

    def test_py_typed_file_exists(self):
        """``quant_lib/py.typed`` must be present."""
        marker = Path(quant_lib.__file__).parent / "py.typed"
        assert marker.exists(), (
            f"py.typed marker not found at {marker}. "
            "This is required for downstream type checkers to use "
            "the framework's type annotations (PEP 561)."
        )

    def test_py_typed_file_is_empty_or_whitespace(self):
        """The marker is conventionally an empty file."""
        marker = Path(quant_lib.__file__).parent / "py.typed"
        if marker.exists():
            content = marker.read_text()
            # Either empty or whitespace-only is acceptable
            assert content.strip() == "", (
                f"py.typed should be empty, got: {content!r}"
            )


# ═══════════════════════════════════════════════════════════════════════
# Public API importability
# ═══════════════════════════════════════════════════════════════════════


class TestPublicAPIImportable:
    """All names in ``__all__`` are importable."""

    def test_all_entries_importable(self):
        """Each name in ``__all__`` is either importable as a module
        or accessible as a top-level attribute.
        """
        for name in quant_lib.__all__:
            # Submodules are importable; functions are attributes
            try:
                importlib.import_module(f"quant_lib.{name}")
            except ModuleNotFoundError:
                # Not a submodule — must be a top-level attribute
                assert hasattr(quant_lib, name), (
                    f"{name} is neither a submodule nor an attribute"
                )

    def test_top_level_exports(self):
        from quant_lib import run_commit, run_explore
        assert callable(run_explore)
        assert callable(run_commit)


# ═══════════════════════════════════════════════════════════════════════
# Type annotations on public API
# ═══════════════════════════════════════════════════════════════════════


class TestPublicAPITypeAnnotations:
    """Public functions should declare parameter and return types."""

    def test_run_explore_has_annotations(self):
        """``run_explore`` has type annotations."""
        sig = inspect.signature(quant_lib.run_explore)
        # At least the return type is annotated
        assert sig.return_annotation is not inspect.Signature.empty, (
            "run_explore missing return annotation"
        )
        # Most parameters should be annotated
        annotated = sum(
            1 for p in sig.parameters.values()
            if p.annotation is not inspect.Parameter.empty
        )
        assert annotated >= 2, (
            f"run_explore has only {annotated} annotated parameters; "
            "expected at least 2 (experiment_name, cache_dir)"
        )

    def test_run_commit_has_annotations(self):
        """``run_commit`` has type annotations."""
        sig = inspect.signature(quant_lib.run_commit)
        assert sig.return_annotation is not inspect.Signature.empty, (
            "run_commit missing return annotation"
        )

    def test_version_annotation(self):
        """``__version__`` is annotated as ``str``."""
        # ``__version__`` is a module attribute, not a function
        # so we use getattr + annotation check
        annotations = quant_lib.__annotations__
        assert "__version__" in annotations
        assert annotations["__version__"] is str


# ═══════════════════════════════════════════════════════════════════════
# Public function signatures are well-formed
# ═══════════════════════════════════════════════════════════════════════


class TestPublicAPISignatures:
    """Public functions have stable, documented signatures."""

    def test_run_explore_has_three_params(self):
        sig = inspect.signature(quant_lib.run_explore)
        params = list(sig.parameters.keys())
        # experiment_name (required), cache_dir, n_spa_iters
        assert "experiment_name" in params
        assert "cache_dir" in params
        assert "n_spa_iters" in params

    def test_run_commit_has_two_params(self):
        sig = inspect.signature(quant_lib.run_commit)
        params = list(sig.parameters.keys())
        assert "experiment_name" in params
        assert "cache_dir" in params

    def test_run_explore_experiment_name_is_required(self):
        """``experiment_name`` has no default (required positional)."""
        sig = inspect.signature(quant_lib.run_explore)
        p = sig.parameters["experiment_name"]
        assert p.default is inspect.Parameter.empty
        assert p.kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.KEYWORD_ONLY,
        )
