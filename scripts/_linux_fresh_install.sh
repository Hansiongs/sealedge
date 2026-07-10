#!/usr/bin/env bash
# Fresh Linux install + targeted then full test suite (WSL Ubuntu).
# From Windows (git-bash): MSYS_NO_PATHCONV=1 wsl.exe -d Ubuntu -- bash /mnt/c/Users/Administrator/Desktop/sealedge/scripts/_linux_fresh_install.sh
# From Ubuntu shell: bash /mnt/c/Users/Administrator/Desktop/sealedge/scripts/_linux_fresh_install.sh
set -euo pipefail
cd /mnt/c/Users/Administrator/Desktop/sealedge
echo "=== env ==="
pwd
python3 --version
uname -a

VENV=/tmp/sealedge-linux-venv
rm -rf "$VENV"

# Ubuntu minimal images often lack ensurepip / python3-venv.
if python3 -m venv "$VENV" 2>/tmp/sealedge-venv-err; then
  # shellcheck disable=SC1091
  source "$VENV/bin/activate"
  python -m pip install -U pip setuptools wheel
else
  echo "venv with ensurepip failed; bootstrapping without-pip:"
  cat /tmp/sealedge-venv-err || true
  rm -rf "$VENV"
  python3 -m venv --without-pip "$VENV"
  curl -sS https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py
  "$VENV/bin/python" /tmp/get-pip.py
  "$VENV/bin/python" -m pip install -U pip setuptools wheel
  # shellcheck disable=SC1091
  source "$VENV/bin/activate"
fi

echo "=== pip install -e .[dev] ==="
python -m pip install -e ".[dev,viz]"
python -c "import quant_lib; print('import', quant_lib.__file__); print('ver', getattr(quant_lib, '__version__', '?'))"

export QUANT_LIB_HMAC_SECRET="${QUANT_LIB_HMAC_SECRET:-sealedge-jss-reproduction-32chars-min}"

echo "=== collect ==="
python -m pytest --collect-only -q | tail -8

echo "=== suite: spa + validation + python_api + experiments ==="
python -m pytest tests/test_spa.py tests/test_spa_validation.py tests/test_python_api.py tests/test_experiments.py -q --tb=line
echo "SUITE_EXIT:$?"

echo "=== full suite (serial, may take long) ==="
python -m pytest -q --tb=line
echo "FULL_EXIT:$?"
