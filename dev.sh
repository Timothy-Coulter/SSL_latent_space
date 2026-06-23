#!/usr/bin/env bash
set -euo pipefail

  usage() {
  cat <<'USAGE'
dev.sh <command>

Commands:
  format       Run code formatters (ruff format + black + isort)
  lint         Run static lint (ruff check)
  lint-fix     Run lint and auto-fix (ruff --fix)
  typecheck    Run mypy with strict settings
  test         Run pytest with xdist + reruns
  all-checks   Format, lint-fix, typecheck, then tests
  versions     Print Python, Torch, and CUDA info
  clean        Remove caches and build artifacts
  gpu-check    Check GPU availability
  install      Install package in editable mode
  jupyter      Opening Jupyter Lab

Examples:
  ./dev.sh all-checks
USAGE
}

cmd=${1:-}
if [[ -z "$cmd" ]]; then
  usage
  exit 1
fi

  case "$cmd" in
  format)
    echo "[format] ruff format, black, isort"
    uv run --with ruff,black,isort ruff format .
    uv run --with black black .
    uv run --with isort isort .
    ;;
  lint)
    echo "[lint] ruff check"
    uv run --with ruff ruff check .
    ;;
  lint-fix)
    echo "[lint-fix] ruff check --fix"
    uv run --with ruff ruff check --fix .
    ;;
  typecheck)
    echo "[typecheck] mypy"
    uv run --with mypy mypy .
    ;;
  test)
    echo "[test] pytest (uses pyproject addopts)"
    uv run --with pytest pytest
    ;;
  integration-tests)
    echo "[integration-tests] pytest integration_tests (uses pyproject addopts)"
    uv run --with pytest pytest integration_tests
    ;;
  versions)
    echo "[versions] Python, torch, CUDA"
    python - <<'PY'
import sys
print("python:", sys.version.replace("\n"," "))
try:
    import torch
    print("torch:", torch.__version__)
    print("torch.cuda.is_available():", torch.cuda.is_available())
    print("torch.version.cuda:", torch.version.cuda)
    try:
        from torch.utils.cpp_extension import CUDA_HOME
        print("CUDA_HOME:", CUDA_HOME)
    except Exception as e:
        print("CUDA_HOME: n/a (", e, ")")
except Exception as e:
    print("torch import failed:", e)
PY
    ;;
  clean)
    echo "[clean] removing build and tool caches"
    rm -rf \
      .mypy_cache \
      .pytest_cache \
      .ruff_cache \
      .cache \
      build \
      dist \
      src/*.egg-info \
      **/__pycache__ 2>/dev/null || true
    ;;
  all-checks)
    "$0" format
    "$0" lint-fix
    "$0" typecheck
    "$0" test
    "$0" integration-tests
    ;;
  *)
    usage
    exit 1
    ;;

  gpu-check)
      echo "[gpu-check] Checking GPU availability"
      python -c "import torch; print(f'GPU Available: {torch.cuda.is_available()}'); print(f'GPU Count: {torch.cuda.device_count()}'); [print(f'GPU {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())]"
      ;;

  install)
      echo "[install] Installing package in editable mode"
      uv pip install -e .
      ;;

  jupyter)
      echo "[jupyter] Opening Jupyter Lab"
      open http://localhost:8888
      ;;
esac
