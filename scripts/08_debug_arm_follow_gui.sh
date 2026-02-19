#!/usr/bin/env bash
set -euo pipefail

PACK_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

OUT_DIR=${1:-"$PACK_ROOT/artifacts/debug_schemeA2"}
PATTERN=${2:-"lift_place"}
SCENE=${3:-"kitchen_pick_and_place"}
OBJECT=${4:-"cracker_box"}
DEVICE=${DEVICE:-"cuda:0"}
HEADLESS=${HEADLESS:-"0"}

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
RIGHT_WRIST_POS_OBJ=${RIGHT_WRIST_POS_OBJ:-"-0.20,-0.03,0.10"}

# Walk-to-grasp (Feature 1)
WALK_TO_GRASP=${WALK_TO_GRASP:-""}
WALK_TARGET_BASE_POS_W=${WALK_TARGET_BASE_POS_W:-""}
WALK_TARGET_OFFSET_OBJ_W=${WALK_TARGET_OFFSET_OBJ_W:-"-0.35,0.0,0.0"}
WALK_TARGET_OFFSET_FRAME=${WALK_TARGET_OFFSET_FRAME:-"object"}
WALK_TARGET_YAW_MODE=${WALK_TARGET_YAW_MODE:-"face_object"}
WALK_TARGET_YAW_DEG=${WALK_TARGET_YAW_DEG:-"0.0"}
WALK_NAV_MAX_LIN_SPEED=${WALK_NAV_MAX_LIN_SPEED:-"0.22"}
WALK_NAV_MAX_ANG_SPEED=${WALK_NAV_MAX_ANG_SPEED:-"0.55"}
WALK_NAV_DT=${WALK_NAV_DT:-"0.02"}
WALK_PREGRASP_HOLD_STEPS=${WALK_PREGRASP_HOLD_STEPS:-"25"}
RIGHT_WRIST_NAV_POS=${RIGHT_WRIST_NAV_POS:-"0.201,-0.145,0.101"}
RIGHT_WRIST_NAV_QUAT_WXYZ=${RIGHT_WRIST_NAV_QUAT_WXYZ:-"1.0,0.0,0.0,0.0"}

# CEDex grasp import (Feature 2 runtime glue)
CEDEX_GRASP_PT=${CEDEX_GRASP_PT:-""}
CEDEX_GRASP_INDEX=${CEDEX_GRASP_INDEX:-"0"}
CEDEX_WRIST_POS_OFFSET=${CEDEX_WRIST_POS_OFFSET:-"0.0,0.0,0.0"}
CEDEX_WRIST_QUAT_OFFSET_WXYZ=${CEDEX_WRIST_QUAT_OFFSET_WXYZ:-"1.0,0.0,0.0,0.0"}

if [[ "$SCENE" == "galileo_g1_locomanip_pick_and_place" ]]; then
  SCENE_PRESET="galileo_locomanip"
  BASE_POS_W=${BASE_POS_W:-"0.0,0.18,0.0"}
  WALK_TO_GRASP=${WALK_TO_GRASP:-"0"}
elif [[ "$SCENE" == "kitchen_pick_and_place" ]]; then
  SCENE_PRESET="kitchen_pick_and_place"
  BASE_POS_W=${BASE_POS_W:-"0.05,0.0,0.0"}
  WALK_TO_GRASP=${WALK_TO_GRASP:-"1"}
else
  BASE_POS_W=${BASE_POS_W:-"0.0,0.0,0.0"}
  WALK_TO_GRASP=${WALK_TO_GRASP:-"0"}
fi

"$PYTHON" "$PACK_ROOT/isaac_replay/generate_debug_object_traj.py" \
  --output "$KIN_TRAJ_PATH" \
  --output-debug-json "$DEBUG_TRAJ_JSON" \
  --object-name "$OBJECT" \
  --pattern "$PATTERN" \
  --scene-preset "$SCENE_PRESET"

cmd_build=(
  "$PYTHON" "$PACK_ROOT/isaac_replay/build_arm_follow_replay.py"
  --kin-traj-path "$KIN_TRAJ_PATH"
  --output-hdf5 "$REPLAY_HDF5"
  --output-debug-json "$DEBUG_REPLAY_JSON"
  --base-pos-w "$BASE_POS_W"
  --base-yaw 0.0
  --right-wrist-pos-obj="$RIGHT_WRIST_POS_OBJ"
  --right-wrist-quat-obj-wxyz="$RIGHT_WRIST_QUAT_OBJ_WXYZ"
  --right-wrist-quat-control "$RIGHT_WRIST_QUAT_CONTROL"
  --right-wrist-quat-pelvis-wxyz "$RIGHT_WRIST_QUAT_PELVIS_WXYZ"
  --left-hand-state 0.0
  --right-hand-state 0.0
)

if [[ "$WALK_TO_GRASP" == "1" ]]; then
  cmd_build+=(
    --walk-to-grasp
    --walk-target-offset-obj-w="$WALK_TARGET_OFFSET_OBJ_W"
    --walk-target-offset-frame "$WALK_TARGET_OFFSET_FRAME"
    --walk-target-yaw-mode "$WALK_TARGET_YAW_MODE"
    --walk-target-yaw-deg "$WALK_TARGET_YAW_DEG"
    --walk-nav-max-lin-speed "$WALK_NAV_MAX_LIN_SPEED"
    --walk-nav-max-ang-speed "$WALK_NAV_MAX_ANG_SPEED"
    --walk-nav-dt "$WALK_NAV_DT"
    --walk-pregrasp-hold-steps "$WALK_PREGRASP_HOLD_STEPS"
    --right-wrist-nav-pos "$RIGHT_WRIST_NAV_POS"
    --right-wrist-nav-quat-wxyz "$RIGHT_WRIST_NAV_QUAT_WXYZ"
  )
  if [[ -n "$WALK_TARGET_BASE_POS_W" ]]; then
    cmd_build+=(--walk-target-base-pos-w "$WALK_TARGET_BASE_POS_W")
  fi
fi

if [[ -n "$CEDEX_GRASP_PT" ]]; then
  cmd_build+=(
    --cedex-grasp-pt "$CEDEX_GRASP_PT"
    --cedex-grasp-index "$CEDEX_GRASP_INDEX"
    --cedex-wrist-pos-offset "$CEDEX_WRIST_POS_OFFSET"
    --cedex-wrist-quat-offset-wxyz "$CEDEX_WRIST_QUAT_OFFSET_WXYZ"
  )
fi

"${cmd_build[@]}"

KIN_START_STEP=$("$PYTHON" -c 'import json,sys; d=json.load(open(sys.argv[1])); print(int(d.get("replay",{}).get("recommended_kin_start_step",0)))' "$DEBUG_REPLAY_JSON" 2>/dev/null || echo 0)
RUN_MAX_STEPS_DEFAULT=$("$PYTHON" -c 'import json,sys; d=json.load(open(sys.argv[1])); print(int(d.get("replay",{}).get("total_steps",408)))' "$DEBUG_REPLAY_JSON" 2>/dev/null || echo 408)
RUN_MAX_STEPS=${MAX_STEPS:-$RUN_MAX_STEPS_DEFAULT}

cmd_runner=(
  "$PYTHON" "$PACK_ROOT/isaac_replay/policy_runner_kinematic_object_replay.py"
  --device "$DEVICE" --enable_cameras
  --policy_type replay
  --replay_file_path "$REPLAY_HDF5"
  --episode_name demo_0
  --kin-traj-path "$KIN_TRAJ_PATH"
  --kin-asset-name "$OBJECT"
  --kin-start-step "$KIN_START_STEP"
  --kin-apply-timing pre_step
  --max-steps "$RUN_MAX_STEPS"
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
    --kin-start-step "$KIN_START_STEP"
    --kin-apply-timing pre_step
    --max-steps "$RUN_MAX_STEPS"
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
