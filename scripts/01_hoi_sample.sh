#!/usr/bin/env bash
set -euo pipefail

PACK_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

HOI_ROOT="$PACK_ROOT/repos/hoifhli_release"
PYTHON="${HOI_PYTHON:-$(command -v python3)}"

cd "$HOI_ROOT"
"$PYTHON" sample.py

echo "--- HOI results ---"
find "$HOI_ROOT/results" -name human_object_results.pkl 2>/dev/null
