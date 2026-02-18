#!/usr/bin/env bash
set -euo pipefail

PACK_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

OUT_DIR=${1:-"$PACK_ROOT/artifacts/debug_schemeA2"}
PATTERN=${2:-"lift_place"}
SCENE=${3:-"kitchen_pick_and_place"}
OBJECT=${4:-"cracker_box"}
DEVICE=${DEVICE:-"cuda:0"}

mkdir -p "$OUT_DIR"

ISAAC_ROOT="$PACK_ROOT/repos/IsaacLab-Arena"
PYTHON="${ISAAC_PYTHON:-$(command -v python3)}"

KIN_TRAJ_PATH="$OUT_DIR/object_kinematic_traj.npz"
REPLAY_HDF5="$OUT_DIR/replay_actions_arm_follow.hdf5"
DEBUG_TRAJ_JSON="$OUT_DIR/debug_traj.json"
DEBUG_REPLAY_JSON="$OUT_DIR/debug_replay.json"

SCENE_PRESET="none"
BASE_POS_W=${BASE_POS_W:-""}
G1_INIT_YAW_DEG=${G1_INIT_YAW_DEG:-"0.0"}
RIGHT_WRIST_QUAT_OBJ_WXYZ=${RIGHT_WRIST_QUAT_OBJ_WXYZ:-"0.70710678,0.0,-0.70710678,0.0"}
RIGHT_WRIST_QUAT_CONTROL=${RIGHT_WRIST_QUAT_CONTROL:-"constant_pelvis"}
RIGHT_WRIST_QUAT_PELVIS_WXYZ=${RIGHT_WRIST_QUAT_PELVIS_WXYZ:-"1.0,0.0,0.0,0.0"}
if [[ "$SCENE" == "galileo_g1_locomanip_pick_and_place" ]]; then
  SCENE_PRESET="galileo_locomanip"
  BASE_POS_W=${BASE_POS_W:-"0.0,0.18,0.0"}
elif [[ "$SCENE" == "kitchen_pick_and_place" ]]; then
  SCENE_PRESET="kitchen_pick_and_place"
  BASE_POS_W=${BASE_POS_W:-"0.05,0.0,0.0"}
else
  BASE_POS_W=${BASE_POS_W:-"0.0,0.0,0.0"}
fi

"$PYTHON" "$PACK_ROOT/isaac_replay/generate_debug_object_traj.py" \
  --output "$KIN_TRAJ_PATH" \
  --output-debug-json "$DEBUG_TRAJ_JSON" \
  --object-name "$OBJECT" \
  --pattern "$PATTERN" \
  --scene-preset "$SCENE_PRESET"

"$PYTHON" "$PACK_ROOT/isaac_replay/build_arm_follow_replay.py" \
  --kin-traj-path "$KIN_TRAJ_PATH" \
  --output-hdf5 "$REPLAY_HDF5" \
  --output-debug-json "$DEBUG_REPLAY_JSON" \
  --base-pos-w "$BASE_POS_W" \
  --base-yaw 0.0 \
  --right-wrist-pos-obj=-0.20,-0.03,0.10 \
  --right-wrist-quat-obj-wxyz="$RIGHT_WRIST_QUAT_OBJ_WXYZ" \
  --right-wrist-quat-control "$RIGHT_WRIST_QUAT_CONTROL" \
  --right-wrist-quat-pelvis-wxyz "$RIGHT_WRIST_QUAT_PELVIS_WXYZ" \
  --left-hand-state 0.0 \
  --right-hand-state 0.0

cmd_runner=(
  "$PYTHON" "$PACK_ROOT/isaac_replay/policy_runner_kinematic_object_replay.py"
  --device "$DEVICE" --enable_cameras
  --policy_type replay
  --replay_file_path "$REPLAY_HDF5"
  --episode_name demo_0
  --kin-traj-path "$KIN_TRAJ_PATH"
  --kin-asset-name "$OBJECT"
  --kin-apply-timing pre_step
  --max-steps 408
  "$SCENE"
  --object "$OBJECT"
  --embodiment g1_wbc_pink
)

if [[ "$SCENE" == "kitchen_pick_and_place" ]]; then
  cmd_runner+=(--g1-init-pos-w "$BASE_POS_W" --g1-init-yaw-deg "$G1_INIT_YAW_DEG")
fi

cd "$ISAAC_ROOT"
"${cmd_runner[@]}"
