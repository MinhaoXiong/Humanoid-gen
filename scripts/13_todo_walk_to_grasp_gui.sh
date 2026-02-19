#!/usr/bin/env bash
# One-click GUI: TODO walk-to-grasp pipeline with CuRobo IK validation.
# Usage:
#   bash scripts/13_todo_walk_to_grasp_gui.sh [OUT_DIR] [PATTERN] [SCENE] [OBJECT]
#
# Example (same style as 08_debug_arm_follow_gui.sh):
#   cd "$PACK_ROOT"
#   DEVICE=cuda:0 BASE_POS_W=-0.55,0.0,0.0 G1_INIT_YAW_DEG=0.0 \
#   bash scripts/13_todo_walk_to_grasp_gui.sh \
#     "$PACK_ROOT/artifacts/todo_curobo_gui" \
#     lift_place \
#     kitchen_pick_and_place \
#     cracker_box
set -euo pipefail

PACK_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

OUT_DIR=${1:-"$PACK_ROOT/artifacts/todo_curobo_gui"}
PATTERN=${2:-"lift_place"}
SCENE=${3:-"kitchen_pick_and_place"}
OBJECT=${4:-"cracker_box"}
DEVICE=${DEVICE:-"cuda:0"}
HEADLESS=${HEADLESS:-"0"}
PLANNER=${PLANNER:-"auto"}
MOMAGEN_STYLE=${MOMAGEN_STYLE:-"1"}
MOMAGEN_START_DIST_MIN=${MOMAGEN_START_DIST_MIN:-"0.4"}
MOMAGEN_START_DIST_MAX=${MOMAGEN_START_DIST_MAX:-"1.0"}
MOMAGEN_TARGET_DIST_MIN=${MOMAGEN_TARGET_DIST_MIN:-"0.4"}
MOMAGEN_TARGET_DIST_MAX=${MOMAGEN_TARGET_DIST_MAX:-"1.0"}
MOMAGEN_MIN_TRAVEL_DIST=${MOMAGEN_MIN_TRAVEL_DIST:-"0.25"}
MOMAGEN_SAMPLE_ATTEMPTS=${MOMAGEN_SAMPLE_ATTEMPTS:-"40"}
MOMAGEN_SAMPLE_SEED=${MOMAGEN_SAMPLE_SEED:-"13"}

# Scene-specific defaults (G1 starts far from table, walks to grasp)
if [[ "$SCENE" == "kitchen_pick_and_place" ]]; then
  BASE_POS_W=${BASE_POS_W:-"-0.52,0.0,0.0"}
  G1_INIT_YAW_DEG=${G1_INIT_YAW_DEG:-"0.0"}
  WALK_TARGET_OFFSET=${WALK_TARGET_OFFSET:-"-0.30,-0.10,0.0"}
  TRAJ_START_POS_W=${TRAJ_START_POS_W:-"0.36,-0.16,0.07"}
  TRAJ_END_POS_W=${TRAJ_END_POS_W:-"0.30,-0.06,0.07"}
  TRAJ_LIFT_HEIGHT=${TRAJ_LIFT_HEIGHT:-"0.08"}
  # TODO: RIGHT_WRIST_POS_OBJ is hardcoded for debug. Should read from BODex .pt file via --cedex-grasp-pt.
  RIGHT_WRIST_POS_OBJ=${RIGHT_WRIST_POS_OBJ:-"-0.16,-0.05,0.06"}
  REPLAY_BASE_HEIGHT=${REPLAY_BASE_HEIGHT:-"0.80"}
elif [[ "$SCENE" == "galileo_g1_locomanip_pick_and_place" ]]; then
  BASE_POS_W=${BASE_POS_W:-"-0.80,0.18,0.0"}
  G1_INIT_YAW_DEG=${G1_INIT_YAW_DEG:-"0.0"}
  WALK_TARGET_OFFSET=${WALK_TARGET_OFFSET:-"-0.45,0.0,0.0"}
  TRAJ_START_POS_W=${TRAJ_START_POS_W:-"0.5785,0.18,0.0707"}
  TRAJ_END_POS_W=${TRAJ_END_POS_W:-"0.52,0.08,0.0707"}
  TRAJ_LIFT_HEIGHT=${TRAJ_LIFT_HEIGHT:-"0.10"}
  RIGHT_WRIST_POS_OBJ=${RIGHT_WRIST_POS_OBJ:-"-0.18,-0.04,0.08"}
  REPLAY_BASE_HEIGHT=${REPLAY_BASE_HEIGHT:-"0.78"}
else
  BASE_POS_W=${BASE_POS_W:-"0.0,0.0,0.0"}
  G1_INIT_YAW_DEG=${G1_INIT_YAW_DEG:-"0.0"}
  WALK_TARGET_OFFSET=${WALK_TARGET_OFFSET:-"-0.35,0.0,0.0"}
  TRAJ_START_POS_W=${TRAJ_START_POS_W:-"0.4,0.0,0.1"}
  TRAJ_END_POS_W=${TRAJ_END_POS_W:-"0.3,0.1,0.1"}
  TRAJ_LIFT_HEIGHT=${TRAJ_LIFT_HEIGHT:-"0.12"}
  RIGHT_WRIST_POS_OBJ=${RIGHT_WRIST_POS_OBJ:-"-0.20,-0.03,0.10"}
  REPLAY_BASE_HEIGHT=${REPLAY_BASE_HEIGHT:-"0.78"}
fi

# CuRobo lives in BODex/src — add to PYTHONPATH so MotionGen works
export PYTHONPATH="${PACK_ROOT}/../BODex/src:${PYTHONPATH:-}"

PYTHON="${ISAAC_PYTHON:-/home/ubuntu/miniconda3/envs/isaaclab_arena/bin/python}"

# Build args for run_walk_to_grasp_todo.py
cmd=(
  "$PYTHON" "$PACK_ROOT/scripts/run_walk_to_grasp_todo.py"
  --out-dir "$OUT_DIR"
  --scene "$SCENE"
  --object "$OBJECT"
  --pattern "$PATTERN"
  --planner "$PLANNER"
  --base-pos-w="$BASE_POS_W"
  --base-yaw-deg "$G1_INIT_YAW_DEG"
  --walk-target-offset-obj-w="$WALK_TARGET_OFFSET"
  --traj-start-pos-w="$TRAJ_START_POS_W"
  --traj-end-pos-w="$TRAJ_END_POS_W"
  --traj-lift-height="$TRAJ_LIFT_HEIGHT"
  --right-wrist-pos-obj="$RIGHT_WRIST_POS_OBJ"
  --replay-base-height="$REPLAY_BASE_HEIGHT"
  --device "$DEVICE"
  --isaac-python "$PYTHON"
)

if [[ "$MOMAGEN_STYLE" != "0" ]]; then
  cmd+=(
    --momagen-style
    --momagen-start-dist-min "$MOMAGEN_START_DIST_MIN"
    --momagen-start-dist-max "$MOMAGEN_START_DIST_MAX"
    --momagen-target-dist-min "$MOMAGEN_TARGET_DIST_MIN"
    --momagen-target-dist-max "$MOMAGEN_TARGET_DIST_MAX"
    --momagen-min-travel-dist "$MOMAGEN_MIN_TRAVEL_DIST"
    --momagen-sample-attempts "$MOMAGEN_SAMPLE_ATTEMPTS"
    --momagen-sample-seed "$MOMAGEN_SAMPLE_SEED"
  )
fi

if [[ "$HEADLESS" != "0" ]]; then
  cmd+=(--headless)
fi

echo "============================================"
echo "[13_todo] Scene:   $SCENE"
echo "[13_todo] Object:  $OBJECT"
echo "[13_todo] Planner: $PLANNER"
echo "[13_todo] G1 start: $BASE_POS_W  yaw=$G1_INIT_YAW_DEG"
echo "[13_todo] Walk offset: $WALK_TARGET_OFFSET"
echo "[13_todo] Traj start/end: $TRAJ_START_POS_W -> $TRAJ_END_POS_W"
echo "[13_todo] Traj lift height: $TRAJ_LIFT_HEIGHT"
echo "[13_todo] Right wrist pos (obj): $RIGHT_WRIST_POS_OBJ"
echo "[13_todo] Replay base height: $REPLAY_BASE_HEIGHT"
echo "[13_todo] MoMaGen-style: $MOMAGEN_STYLE (start=${MOMAGEN_START_DIST_MIN}-${MOMAGEN_START_DIST_MAX}m, target=${MOMAGEN_TARGET_DIST_MIN}-${MOMAGEN_TARGET_DIST_MAX}m, min_nav=${MOMAGEN_MIN_TRAVEL_DIST}m)"
echo "[13_todo] Output:  $OUT_DIR"
echo "============================================"

"${cmd[@]}"

# Print IK result from report
REPORT="$OUT_DIR/todo_run_report.json"
if [[ -f "$REPORT" ]]; then
  echo ""
  echo "============================================"
  "$PYTHON" -c "
import json, sys
r = json.load(open(sys.argv[1]))
p = r.get('planner', {})
ik = p.get('ik_result')
print('[13_todo] Planner used:', p.get('planner_used'))
print('[13_todo] CuRobo available:', p.get('curobo_available'))
if ik:
    print('[13_todo] IK reachable:', ik.get('reachable'))
    print('[13_todo] IK pos_error:', ik.get('position_error'))
else:
    print('[13_todo] IK check: skipped')
mg = p.get('motion_gen_result')
if mg:
    print('[13_todo] MotionGen success:', mg.get('success'))
    print('[13_todo] MotionGen steps:', mg.get('num_steps'))
else:
    print('[13_todo] MotionGen: skipped')
print('[13_todo] Notes:', p.get('notes'))
print('[13_todo] Status:', r.get('status'))
" "$REPORT"
  echo "============================================"
fi
