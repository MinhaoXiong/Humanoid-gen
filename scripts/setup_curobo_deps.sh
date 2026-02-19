#!/usr/bin/env bash
# Install CuRobo dependencies into an existing isaaclab conda env.
# Usage: bash scripts/setup_curobo_deps.sh [CONDA_ENV_NAME]
set -euo pipefail

ENV_NAME="${1:-isaaclab_arena}"
PIP="$(conda run -n "$ENV_NAME" which pip)"

echo "[setup] Installing CuRobo deps into conda env: $ENV_NAME"

$PIP install setuptools_scm warp-lang "trimesh>=4.0" yourdfpy h5py scipy

echo "[setup] Done. Verify with:"
echo "  conda run -n $ENV_NAME python -c \"import trimesh, yourdfpy, warp; print('OK')\""
