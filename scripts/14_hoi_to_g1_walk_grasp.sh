#!/usr/bin/env bash
# End-to-end: HOI pkl → G1 retarget → walk-to-grasp replay in Isaac Sim.
#
# Usage:
#   cd "$PACK_ROOT"
#   DEVICE=cuda:0 bash scripts/14_hoi_to_g1_walk_grasp.sh \
#     path/to/human_object_results.pkl \
#     artifacts/hoi_retarget_run1 \
#     kitchen_pick_and_place \
#     cracker_box
set -euo pipefail

PACK_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

HOI_PKL=${1:?"Usage: $0 <hoi_pkl> [out_dir] [scene] [object]"}
OUT_DIR=${2:-"$PACK_ROOT/artifacts/hoi_retarget_gui"}
SCENE=${3:-"kitchen_pick_and_place"}
OBJECT=${4:-"cracker_box"}
DEVICE=${DEVICE:-"cuda:0"}
HEADLESS=${HEADLESS:-"0"}
SPIDER_PYTHON="${SPIDER_PYTHON:-/home/ubuntu/miniconda3/envs/spider/bin/python}"
ISAAC_PYTHON="${ISAAC_PYTHON:-$(command -v python3)}"

HOI_PKL="$(python3 -c "import os,sys; print(os.path.abspath(sys.argv[1]))" "$HOI_PKL")"
OUT_DIR="$(python3 -c "import os,sys; print(os.path.abspath(sys.argv[1]))" "$OUT_DIR")"
mkdir -p "$OUT_DIR"

# Scene defaults
if [[ "$SCENE" == "kitchen_pick_and_place" ]]; then
  BASE_POS_W=${BASE_POS_W:-"-0.52,0.0,0.0"}
  G1_INIT_YAW_DEG=${G1_INIT_YAW_DEG:-"0.0"}
  WALK_TARGET_OFFSET=${WALK_TARGET_OFFSET:-"-0.30,-0.10,0.0"}
  RIGHT_WRIST_POS_OBJ=${RIGHT_WRIST_POS_OBJ:-"-0.16,-0.05,0.06"}
  REPLAY_BASE_HEIGHT=${REPLAY_BASE_HEIGHT:-"0.80"}
else
  BASE_POS_W=${BASE_POS_W:-"0.0,0.0,0.0"}
  G1_INIT_YAW_DEG=${G1_INIT_YAW_DEG:-"0.0"}
  WALK_TARGET_OFFSET=${WALK_TARGET_OFFSET:-"-0.35,0.0,0.0"}
  RIGHT_WRIST_POS_OBJ=${RIGHT_WRIST_POS_OBJ:-"-0.20,-0.03,0.10"}
  REPLAY_BASE_HEIGHT=${REPLAY_BASE_HEIGHT:-"0.78"}
fi

KIN_TRAJ="$OUT_DIR/object_kinematic_traj.npz"
RETARGET_DEBUG="$OUT_DIR/retarget_debug.json"

echo "============================================"
echo "[14] HOI-to-G1 Walk-Grasp Pipeline"
echo "[14] HOI pkl:  $HOI_PKL"
echo "[14] Scene:    $SCENE"
echo "[14] Object:   $OBJECT"
echo "[14] Output:   $OUT_DIR"
echo "============================================"

# --- Step 1: HOI → G1 retarget → object_kinematic_traj.npz ---
echo "[14] Step 1: Retargeting HOI to G1..."
"$SPIDER_PYTHON" "$PACK_ROOT/bridge/hoi_to_g1_retarget.py" \
  --hoi-pickle "$HOI_PKL" \
  --output-npz "$KIN_TRAJ" \
  --output-debug-json "$RETARGET_DEBUG" \
  --scene "$SCENE" \
  --object-name-override "$OBJECT"

echo "[14] Step 1 done: $KIN_TRAJ"

# --- Step 2: Build arm-follow replay HDF5 ---
REPLAY_HDF5="$OUT_DIR/replay_actions_arm_follow.hdf5"
DEBUG_REPLAY_JSON="$OUT_DIR/debug_replay.json"

echo "[14] Step 2: Building arm-follow replay..."
"$ISAAC_PYTHON" "$PACK_ROOT/isaac_replay/build_arm_follow_replay.py" \
  --kin-traj-path "$KIN_TRAJ" \
  --output-hdf5 "$REPLAY_HDF5" \
  --output-debug-json "$DEBUG_REPLAY_JSON" \
  --base-pos-w "$BASE_POS_W" \
  --base-yaw 0.0 \
  --right-wrist-pos-obj="$RIGHT_WRIST_POS_OBJ" \
  --right-wrist-quat-obj-wxyz="0.70710678,0.0,-0.70710678,0.0" \
  --right-wrist-quat-control constant_pelvis \
  --right-wrist-quat-pelvis-wxyz="1.0,0.0,0.0,0.0" \
  --left-hand-state 0.0 \
  --right-hand-state 0.0

echo "[14] Step 2 done: $REPLAY_HDF5"

# --- Step 3: Get max steps from trajectory ---
MAX_STEPS="$("$ISAAC_PYTHON" -c "
import numpy as np, sys
d = np.load(sys.argv[1], allow_pickle=True)
print(int(d['object_pos_w'].shape[0]))
" "$KIN_TRAJ")"

echo "[14] Step 3: Max steps = $MAX_STEPS"

# --- Step 4: Run Isaac Sim replay ---
echo "[14] Step 4: Running Isaac Sim replay..."
ISAAC_ROOT="$PACK_ROOT/repos/IsaacLab-Arena"

cmd_runner=(
  "$ISAAC_PYTHON" "$PACK_ROOT/isaac_replay/policy_runner_kinematic_object_replay.py"
  --device "$DEVICE" --enable_cameras
  --policy_type replay
  --replay_file_path "$REPLAY_HDF5"
  --episode_name demo_0
  --kin-traj-path "$KIN_TRAJ"
  --kin-asset-name "$OBJECT"
  --kin-apply-timing pre_step
  --max-steps "$MAX_STEPS"
  "$SCENE"
  --object "$OBJECT"
  --embodiment g1_wbc_pink
)

if [[ "$HEADLESS" != "0" ]]; then
  cmd_runner=( "${cmd_runner[@]:0:1}" --headless "${cmd_runner[@]:1}" )
fi

if [[ "$SCENE" == "kitchen_pick_and_place" ]]; then
  cmd_runner+=(--g1-init-pos-w "$BASE_POS_W" --g1-init-yaw-deg "$G1_INIT_YAW_DEG")
fi

cd "$ISAAC_ROOT"
"${cmd_runner[@]}"

echo "============================================"
echo "[14] Pipeline complete."
echo "[14] Retarget debug: $RETARGET_DEBUG"
echo "[14] Trajectory:     $KIN_TRAJ"
echo "[14] Replay HDF5:    $REPLAY_HDF5"
echo "============================================"
