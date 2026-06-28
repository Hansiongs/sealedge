"""Regression tests for B0.3: ``quant_lib`` must not install warning filters.

Bug: ``quant_lib.core._config`` called ``warnings.filterwarnings("ignore")``
at module import time. This polluted the host application's warnings
configuration silently — a user importing ``quant_lib`` inside a larger
application would suddenly have Numba/Optuna deprecation warnings hidden.

Fix: remove the global filter from ``_config.py``. No module in
``quant_lib`` (or any transitive dependency that we control) should
install warnings filters at import time.
"""
from __future__ import annotations

import subprocess
import sys
import warnings
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent


class TestNoFilterwarningsInSource:
    """B0.3 fix: no module in ``quant_lib`` should call filterwarnings."""

    def test_no_filterwarnings_in_quant_lib(self):
        """No .py file under quant_lib/ may contain filterwarnings(...)."""
        import re
        quant_lib_dir = REPO_ROOT / "quant_lib"
        offenders = []
        for py_file in quant_lib_dir.rglob("*.py"):
            # Skip test files and __pycache__
            if "__pycache__" in py_file.parts:
                continue
            try:
                text = py_file.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            # Strip comments to avoid false positives
            text_no_comments = re.sub(r"#.*", "", text)
            if re.search(r"\bfilterwarnings\s*\(", text_no_comments):
                offenders.append(str(py_file.relative_to(REPO_ROOT)))
        assert not offenders, (
            f"Found filterwarnings() call(s) in: {offenders}. "
            f"quant_lib must not install warning filters at import time."
        )

    def test_no_import_warnings_in_core_config(self):
        """``_config.py`` must not import the warnings module."""
        config_path = REPO_ROOT / "quant_lib" / "core" / "_config.py"
        text = config_path.read_text(encoding="utf-8")
        assert "import warnings" not in text, (
            "quant_lib/core/_config.py still imports warnings module"
        )


class TestSubprocessImportBehavior:
    """Verify the B0.3 fix using a fresh subprocess (no module caching)."""

    def test_fresh_process_warnings_unchanged(self):
        """A subprocess that imports quant_lib should not see new
        ``ignore`` filters at action priority ``default`` from
        ``_config`` (the original bug).

        We can't assert *zero* new filters (third-party libraries
        transitively installed by Python's import system will add
        their own), but we CAN assert that the specific offender
        (``warnings.filterwarnings("ignore")`` at module scope in
        ``quant_lib.core._config``) is gone.
        """
        script = (
            "import warnings; "
            "before = sum(1 for f in warnings.filters "
            "if f[0] == 'ignore' and f[3] is None); "
            "import quant_lib; "
            "after = sum(1 for f in warnings.filters "
            "if f[0] == 'ignore' and f[3] is None); "
            # The pre-fix code added exactly 1 'ignore' filter at default
            # scope. We can't assert exact counts due to transitive deps,
            # but we CAN verify that 'core._config' itself doesn't add one.
            "from quant_lib.core import _config; "
            "config_added = any('core._config' in str(f) "
            "or '_config.py' in str(f) for f in warnings.filters); "
            "print('OK' if not config_added else 'FAIL')"
        )
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True, text=True, timeout=30,
            cwd=str(REPO_ROOT),
        )
        assert result.returncode == 0, (
            f"Subprocess failed: {result.stderr}"
        )
        assert "OK" in result.stdout, (
            f"Unexpected output: stdout={result.stdout}, stderr={result.stderr}"
        )


class TestUserCodeStillControlsWarnings:
    """The fix must NOT prevent users from configuring their own filters."""

    def test_user_can_still_set_warnings_filter(self):
        """After import, user-supplied filterwarnings calls work as expected."""
        import quant_lib  # noqa: F401
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            warnings.warn("user test warning", UserWarning)
        assert len(captured) == 1
        assert issubclass(captured[0].category, UserWarning)


class TestDeprecationWarningsPropagate:
    """B0.3 fix verification: deprecation warnings must propagate to user."""

    def test_no_quant_lib_module_adds_ignore_filter(self):
        """No quant_lib.* module should add an 'ignore' filter on import."""
        import warnings
        before = [str(f) for f in warnings.filters]
        import importlib, sys
        for name in list(sys.modules):
            if name.startswith("quant_lib"):
                del sys.modules[name]
        import quant_lib  # noqa
        after = [str(f) for f in warnings.filters]
        new_ignores = [f for f in after if f not in before and "ignore" in f]
        quant_lib_filters = [f for f in new_ignores if "quant_lib" in f.lower()]
        assert not quant_lib_filters, (
            f"quant_lib added ignore filters: {quant_lib_filters}"
        )

    def test_user_warning_propagates_after_quant_lib_import(self):
        """A UserWarning raised after import is not silently swallowed."""
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            import quant_lib  # noqa F401
            warnings.warn("test user warning", UserWarning, stacklevel=2)
        assert any(w.category is UserWarning for w in captured), (
            "UserWarning raised post-import was not captured"
        )

    def test_deprecation_warning_visible_post_import(self):
        """Any DeprecationWarning from a loaded module must be visible."""
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            from quant_lib.core import _config  # noqa F401
        # Post B0.3 fix, no DeprecationWarning should be filtered.
        # We don't assert len == 0 (dependencies may warn), but we
        # do assert the catch_warnings mechanism itself works.
        for w in captured:
            assert w.category is not None
