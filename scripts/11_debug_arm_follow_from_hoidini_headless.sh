#!/usr/bin/env bash
set -euo pipefail

PACK_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

HOIDINI_FINAL_PKL=${1:-"$PACK_ROOT/artifacts/hoidini_kitchen_pickplace_run1/sample_final.pickle"}
OUT_DIR=${2:-"$PACK_ROOT/artifacts/debug_schemeA2_hoidini"}
SCENE=${3:-"kitchen_pick_and_place"}
OBJECT=${4:-"cracker_box"}

HOIDINI_FINAL_PKL="$(python3 - "$HOIDINI_FINAL_PKL" <<'PY'
import os, sys
print(os.path.abspath(sys.argv[1]))
PY
)"
OUT_DIR="$(python3 - "$OUT_DIR" <<'PY'
import os, sys
print(os.path.abspath(sys.argv[1]))
PY
)"

mkdir -p "$OUT_DIR"

PYTHON="${ISAAC_PYTHON:-$(command -v python3)}"
HOI_PKL="$OUT_DIR/hoidini_bridge_input.pkl"
HOIDINI_DEBUG_JSON="$OUT_DIR/hoidini_convert_debug.json"

"$PYTHON" "$PACK_ROOT/scripts/10_convert_hoidini_final_to_bridge_pkl.py" \
  --hoidini-final-pickle "$HOIDINI_FINAL_PKL" \
  --output-pickle "$HOI_PKL" \
  --output-debug-json "$HOIDINI_DEBUG_JSON" \
  --object-name-override "$OBJECT"

# HOIDiNi train.fps is 20 by default.
HOI_FPS=${HOI_FPS:-20}

HOI_FPS="$HOI_FPS" \
bash "$PACK_ROOT/scripts/09_debug_arm_follow_from_hoi_headless.sh" \
  "$HOI_PKL" \
  "$OUT_DIR" \
  "$SCENE" \
  "$OBJECT"
