#!/usr/bin/env bash
set -euo pipefail

PACK_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

OUT_DIR=${1:-"$PACK_ROOT/artifacts/debug_scheme1"}
PATTERN=${2:-"lift_place"}
OBJECT=${3:-"brown_box"}
SCENE=${4:-"galileo_g1_locomanip_pick_and_place"}
DEVICE=${DEVICE:-"cuda:0"}

mkdir -p "$OUT_DIR"

ISAAC_ROOT="$PACK_ROOT/repos/IsaacLab-Arena"
PYTHON="${ISAAC_PYTHON:-$(command -v python3)}"

KIN_TRAJ_PATH="$OUT_DIR/object_kinematic_traj.npz"
DEBUG_JSON_PATH="$OUT_DIR/debug_traj.json"

SCENE_PRESET="none"
if [[ "$SCENE" == "galileo_g1_locomanip_pick_and_place" ]]; then
  SCENE_PRESET="galileo_locomanip"
elif [[ "$SCENE" == "kitchen_pick_and_place" ]]; then
  SCENE_PRESET="kitchen_pick_and_place"
fi

"$PYTHON" "$PACK_ROOT/isaac_replay/generate_debug_object_traj.py" \
  --output "$KIN_TRAJ_PATH" \
  --output-debug-json "$DEBUG_JSON_PATH" \
  --object-name "$OBJECT" \
  --pattern "$PATTERN" \
  --scene-preset "$SCENE_PRESET"

cd "$ISAAC_ROOT"
"$PYTHON" "$PACK_ROOT/isaac_replay/policy_runner_kinematic_object_replay.py" \
  --device "$DEVICE" --enable_cameras \
  --object-only \
  --kin-traj-path "$KIN_TRAJ_PATH" \
  --kin-asset-name "$OBJECT" \
  --kin-apply-timing pre_step \
  "$SCENE" \
  --object "$OBJECT" \
  --embodiment g1_wbc_pink
