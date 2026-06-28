"""Smoke tests for `python -m quant_lib` entry point.

Covers quant_lib/__main__.py (currently 0% covered) and exercises
the top-level quant_lib import path.
"""
import os
import subprocess
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _run_module(*args, env_extra=None, timeout=30):
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env.setdefault("QUANT_LIB_HMAC_SECRET", "x" * 64)
    if env_extra:
        env.update(env_extra)
    return subprocess.run(
        [sys.executable, "-m", "quant_lib", *args],
        capture_output=True, text=True, encoding="utf-8",
        errors="replace", timeout=timeout, cwd=REPO_ROOT, env=env,
    )


class TestPythonDashM:
    def test_module_runs_no_args(self):
        """`python -m quant_lib` shows help (typer no_args_is_help)."""
        r = _run_module()
        combined = (r.stdout + r.stderr).lower()
        assert "usage" in combined or "command" in combined

    def test_module_help(self):
        r = _run_module("--help")
        assert "quant_exp" in (r.stdout + r.stderr).lower()

    def test_module_list_runs(self):
        r = _run_module("list")
        assert r.returncode in (0, 2)
