#!/usr/bin/env bash
set -euo pipefail

PACK_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

HOI_PKL=${1:-"$PACK_ROOT/artifacts/hoifhli/human_object_results_compare_fine_01_p0_o3.pkl"}
OUT_DIR=${2:-"$PACK_ROOT/artifacts/debug_schemeA2_hoi"}
SCENE=${3:-"kitchen_pick_and_place"}
OBJECT=${4:-"cracker_box"}
DEVICE=${DEVICE:-"cpu"}
HEADLESS=${HEADLESS:-"1"}

HOI_PKL="$(python3 - "$HOI_PKL" <<'PY'
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

ISAAC_ROOT="$PACK_ROOT/repos/IsaacLab-Arena"
PYTHON="${ISAAC_PYTHON:-$(command -v python3)}"

KIN_TRAJ_PATH="$OUT_DIR/object_kinematic_traj.npz"
BRIDGE_REPLAY_UNUSED="$OUT_DIR/replay_actions_bridge_unused.hdf5"
BRIDGE_DEBUG_JSON="$OUT_DIR/bridge_debug.json"
REPLAY_HDF5="$OUT_DIR/replay_actions_arm_follow.hdf5"
DEBUG_REPLAY_JSON="$OUT_DIR/debug_replay.json"

# Defaults can be overridden by env vars before running this script.
TRJ_OFFSET_W=${TRJ_OFFSET_W:-"0.0,0.0,0.0"}
ALIGN_LAST_POS_W=${ALIGN_LAST_POS_W:-""}
ALIGN_LAST_RAMP_SEC=${ALIGN_LAST_RAMP_SEC:-"1.0"}

if [[ "$SCENE" == "kitchen_pick_and_place" ]]; then
  BASE_POS_W=${BASE_POS_W:-"0.05,0.0,0.0"}
  G1_INIT_YAW_DEG=${G1_INIT_YAW_DEG:-"0.0"}
  RIGHT_WRIST_QUAT_OBJ_WXYZ=${RIGHT_WRIST_QUAT_OBJ_WXYZ:-"0.70710678,0.0,-0.70710678,0.0"}
  TRJ_SCALE_XYZ=${TRJ_SCALE_XYZ:-"0.13,0.22,0.30"}
  ALIGN_FIRST_POS_W=${ALIGN_FIRST_POS_W:-"0.40,0.00,0.10"}
  CLIP_Z_MIN=${CLIP_Z_MIN:-"0.08"}
  CLIP_Z_MAX=${CLIP_Z_MAX:-"0.38"}
  CLIP_XY_MIN=${CLIP_XY_MIN:-"0.05,-0.45"}
  CLIP_XY_MAX=${CLIP_XY_MAX:-"0.65,0.45"}
elif [[ "$SCENE" == "galileo_g1_locomanip_pick_and_place" ]]; then
  BASE_POS_W=${BASE_POS_W:-"0.0,0.18,0.0"}
  RIGHT_WRIST_QUAT_OBJ_WXYZ=${RIGHT_WRIST_QUAT_OBJ_WXYZ:-"0.70710678,0.0,-0.70710678,0.0"}
  TRJ_SCALE_XYZ=${TRJ_SCALE_XYZ:-"1.0,1.0,1.0"}
  ALIGN_FIRST_POS_W=${ALIGN_FIRST_POS_W:-"0.5785,0.18,0.0707"}
  CLIP_Z_MIN=${CLIP_Z_MIN:-"0.06"}
  CLIP_Z_MAX=${CLIP_Z_MAX:-"0.40"}
  CLIP_XY_MIN=${CLIP_XY_MIN:-""}
  CLIP_XY_MAX=${CLIP_XY_MAX:-""}
else
  BASE_POS_W=${BASE_POS_W:-"0.0,0.0,0.0"}
  RIGHT_WRIST_QUAT_OBJ_WXYZ=${RIGHT_WRIST_QUAT_OBJ_WXYZ:-"0.70710678,0.0,-0.70710678,0.0"}
  TRJ_SCALE_XYZ=${TRJ_SCALE_XYZ:-"1.0,1.0,1.0"}
  ALIGN_FIRST_POS_W=${ALIGN_FIRST_POS_W:-""}
  CLIP_Z_MIN=${CLIP_Z_MIN:-""}
  CLIP_Z_MAX=${CLIP_Z_MAX:-""}
  CLIP_XY_MIN=${CLIP_XY_MIN:-""}
  CLIP_XY_MAX=${CLIP_XY_MAX:-""}
fi

cmd_bridge=(
  "$PYTHON" "$PACK_ROOT/bridge/build_replay.py"
  --hoi-pickle "$HOI_PKL"
  --output-hdf5 "$BRIDGE_REPLAY_UNUSED"
  --output-object-traj "$KIN_TRAJ_PATH"
  --output-debug-json "$BRIDGE_DEBUG_JSON"
  --hoi-fps "${HOI_FPS:-30}"
  --target-fps "${TARGET_FPS:-50}"
  --traj-scale-xyz "$TRJ_SCALE_XYZ"
  --traj-offset-w "$TRJ_OFFSET_W"
  --align-last-ramp-sec "$ALIGN_LAST_RAMP_SEC"
  --object-name-override "$OBJECT"
)

if [[ -n "$ALIGN_FIRST_POS_W" ]]; then
  cmd_bridge+=(--align-first-pos-w "$ALIGN_FIRST_POS_W")
fi
if [[ -n "$ALIGN_LAST_POS_W" ]]; then
  cmd_bridge+=(--align-last-pos-w "$ALIGN_LAST_POS_W")
fi
if [[ -n "$CLIP_Z_MIN" ]]; then
  cmd_bridge+=(--clip-z-min "$CLIP_Z_MIN")
fi
if [[ -n "$CLIP_Z_MAX" ]]; then
  cmd_bridge+=(--clip-z-max "$CLIP_Z_MAX")
fi
if [[ -n "$CLIP_XY_MIN" ]]; then
  cmd_bridge+=(--clip-xy-min "$CLIP_XY_MIN")
fi
if [[ -n "$CLIP_XY_MAX" ]]; then
  cmd_bridge+=(--clip-xy-max "$CLIP_XY_MAX")
fi

"${cmd_bridge[@]}"

"$PYTHON" "$PACK_ROOT/isaac_replay/build_arm_follow_replay.py" \
  --kin-traj-path "$KIN_TRAJ_PATH" \
  --output-hdf5 "$REPLAY_HDF5" \
  --output-debug-json "$DEBUG_REPLAY_JSON" \
  --base-pos-w "$BASE_POS_W" \
  --base-yaw 0.0 \
  --right-wrist-pos-obj=-0.20,-0.03,0.10 \
  --right-wrist-quat-obj-wxyz="$RIGHT_WRIST_QUAT_OBJ_WXYZ" \
  --left-hand-state 0.0 \
  --right-hand-state 0.0

if [[ -z "${MAX_STEPS:-}" ]]; then
  MAX_STEPS="$("$PYTHON" - "$KIN_TRAJ_PATH" <<'PY'
import numpy as np
import sys
data = np.load(sys.argv[1], allow_pickle=True)
print(int(data["object_pos_w"].shape[0]))
PY
)"
fi

cmd_runner=(
  "$PYTHON" "$PACK_ROOT/isaac_replay/policy_runner_kinematic_object_replay.py"
  --device "$DEVICE" --enable_cameras
  --policy_type replay
  --replay_file_path "$REPLAY_HDF5"
  --episode_name demo_0
  --kin-traj-path "$KIN_TRAJ_PATH"
  --kin-asset-name "$OBJECT"
  --kin-apply-timing pre_step
  --max-steps "$MAX_STEPS"
  "$SCENE"
  --object "$OBJECT"
  --embodiment g1_wbc_pink
)

if [[ "$HEADLESS" != "0" ]]; then
  cmd_runner=( "$PYTHON" "$PACK_ROOT/isaac_replay/policy_runner_kinematic_object_replay.py"
    --headless
    --device "$DEVICE" --enable_cameras
    --policy_type replay
    --replay_file_path "$REPLAY_HDF5"
    --episode_name demo_0
    --kin-traj-path "$KIN_TRAJ_PATH"
    --kin-asset-name "$OBJECT"
    --kin-apply-timing pre_step
    --max-steps "$MAX_STEPS"
    "$SCENE"
    --object "$OBJECT"
    --embodiment g1_wbc_pink
  )
fi

if [[ "$SCENE" == "kitchen_pick_and_place" ]]; then
  cmd_runner+=(--g1-init-pos-w "$BASE_POS_W" --g1-init-yaw-deg "$G1_INIT_YAW_DEG")
fi

cd "$ISAAC_ROOT"
"${cmd_runner[@]}"
