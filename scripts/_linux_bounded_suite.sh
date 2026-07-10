#!/usr/bin/env bash
# Fresh-or-reuse Linux venv + SPA/python_api/experiments suite only.
set -euo pipefail
cd /mnt/c/Users/Administrator/Desktop/sealedge
VENV=/tmp/sealedge-linux-venv
PY="${VENV}/bin/python"

bootstrap() {
  echo "=== recreate venv at ${VENV} ==="
  rm -rf "${VENV}"
  if ! python3 -m venv "${VENV}" 2>/tmp/sealedge-venv-err; then
    echo "ensurepip missing; without-pip bootstrap"
    cat /tmp/sealedge-venv-err || true
    rm -rf "${VENV}"
    python3 -m venv --without-pip "${VENV}"
    curl -sS https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py
    "${PY}" /tmp/get-pip.py
  fi
  "${PY}" -m pip install -U pip setuptools wheel
  "${PY}" -m pip install -e ".[dev,viz]"
}

if [ ! -x "${PY}" ] || ! "${PY}" -c "import quant_lib, pytest, matplotlib, seaborn" 2>/dev/null; then
  bootstrap
else
  echo "=== reuse ${VENV} ==="
fi

"${PY}" -c "import quant_lib; print('ver', quant_lib.__version__)"
export QUANT_LIB_HMAC_SECRET="${QUANT_LIB_HMAC_SECRET:-sealedge-jss-reproduction-32chars-min}"
echo "=== bounded suite ==="
"${PY}" -m pytest tests/test_spa.py tests/test_spa_validation.py tests/test_python_api.py tests/test_experiments.py -q --tb=line
echo "SUITE_EXIT:$?"
