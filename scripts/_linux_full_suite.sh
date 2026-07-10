#!/usr/bin/env bash
# Recreate venv if needed, then full pytest. Log: /tmp/sealedge-full-pytest.txt
set -euo pipefail
cd /mnt/c/Users/Administrator/Desktop/sealedge
VENV=/tmp/sealedge-linux-venv
PY="${VENV}/bin/python"

bootstrap() {
  echo "=== recreate venv ==="
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
  echo "=== reuse existing venv ==="
fi

"${PY}" -c "import quant_lib; print('ver', quant_lib.__version__)"
export QUANT_LIB_HMAC_SECRET="${QUANT_LIB_HMAC_SECRET:-sealedge-jss-reproduction-32chars-min}"

# Prefer xdist if available (installed via [dev]); fall back serial.
NPROC=$("${PY}" -c "import os; print(max(1, (os.cpu_count() or 2) // 2))")
echo "=== full suite start (xdist -n ${NPROC}) ==="
set +e
"${PY}" -m pytest -n "${NPROC}" -q --tb=line --no-header 2>&1 | tee /tmp/sealedge-full-pytest.txt
EC=${PIPESTATUS[0]}
set -e
echo "FULL_EXIT:${EC}"
echo "=== fail lines (head) ==="
grep -E 'FAILED|ERROR' /tmp/sealedge-full-pytest.txt | head -80 || true
echo "=== tail ==="
tail -15 /tmp/sealedge-full-pytest.txt
exit "${EC}"
