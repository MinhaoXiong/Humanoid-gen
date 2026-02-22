#!/usr/bin/env bash
# LW-BenchHub walk-to-grasp pipeline.
#
# This script runs the walk-to-grasp pipeline using LW-BenchHub kitchen scenes
# instead of the hardcoded IsaacLab-Arena scenes. It does NOT modify
# scripts/13_todo_walk_to_grasp_gui.sh.
#
# Usage:
#   # With LW-BenchHub SDK (auto-extract scene from layout/style)
#   bash scripts/15_lwbench_walk_to_grasp.sh [OUT_DIR] [OBJECT] [LAYOUT_ID] [STYLE_ID]
#
#   # With Spider SMPL retargeting input
#   SPIDER_NPZ=/path/to/trajectory_mjwp.npz \
#   SPIDER_XML=/path/to/scene.xml \
#   bash scripts/15_lwbench_walk_to_grasp.sh [OUT_DIR] [OBJECT] [LAYOUT_ID] [STYLE_ID]
#
#   # With local USD file
#   LOCAL_USD=/path/to/kitchen.usd \
#   bash scripts/15_lwbench_walk_to_grasp.sh [OUT_DIR] [OBJECT]
#
# Examples:
#   # Kitchen layout 0, style 0, cracker_box
#   bash scripts/15_lwbench_walk_to_grasp.sh artifacts/lwbench_test cracker_box 0 0
#
#   # With Spider SMPL input
#   SPIDER_NPZ=~/spider_output/trajectory_mjwp.npz \
#   SPIDER_XML=~/spider_output/scene.xml \
#   bash scripts/15_lwbench_walk_to_grasp.sh artifacts/lwbench_spider cracker_box 0 0
#
#   # Headless mode
#   HEADLESS=1 bash scripts/15_lwbench_walk_to_grasp.sh artifacts/lwbench_hl cracker_box 0 0

set -euo pipefail

PACK_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------
OUT_DIR=${1:-"$PACK_ROOT/artifacts/lwbench_walk_to_grasp"}
OBJECT=${2:-"cracker_box"}
LAYOUT_ID=${3:-"0"}
STYLE_ID=${4:-"0"}

# ---------------------------------------------------------------------------
# Environment variables
# ---------------------------------------------------------------------------
DEVICE=${DEVICE:-"cuda:0"}
HEADLESS=${HEADLESS:-"0"}
PLANNER=${PLANNER:-"auto"}
PATTERN=${PATTERN:-"lift_place"}

# Spider inputs (optional)
SPIDER_NPZ=${SPIDER_NPZ:-""}
SPIDER_XML=${SPIDER_XML:-""}
SPIDER_OBJECT_BODY=${SPIDER_OBJECT_BODY:-""}

# Local USD (optional, overrides layout/style)
LOCAL_USD=${LOCAL_USD:-""}

# Mock mode (for testing without USD/SDK)
MOCK_SCENE=${MOCK_SCENE:-"0"}

# Robot parameters
ROBOT_STANDOFF=${ROBOT_STANDOFF:-"0.65"}
REPLAY_BASE_HEIGHT=${REPLAY_BASE_HEIGHT:-"0.78"}

# MoMaGen parameters
MOMAGEN_STYLE=${MOMAGEN_STYLE:-"1"}
MOMAGEN_START_DIST_MIN=${MOMAGEN_START_DIST_MIN:-"0.5"}
MOMAGEN_START_DIST_MAX=${MOMAGEN_START_DIST_MAX:-"0.8"}
MOMAGEN_TARGET_DIST_MIN=${MOMAGEN_TARGET_DIST_MIN:-"0.5"}
MOMAGEN_TARGET_DIST_MAX=${MOMAGEN_TARGET_DIST_MAX:-"0.8"}
MOMAGEN_MIN_TRAVEL_DIST=${MOMAGEN_MIN_TRAVEL_DIST:-"0.25"}
MOMAGEN_MAX_TRAVEL_DIST=${MOMAGEN_MAX_TRAVEL_DIST:-"0.90"}
MOMAGEN_SAMPLE_ATTEMPTS=${MOMAGEN_SAMPLE_ATTEMPTS:-"40"}
MOMAGEN_SAMPLE_SEED=${MOMAGEN_SAMPLE_SEED:-"13"}

# Walk parameters
WALK_NAV_MAX_LIN_SPEED=${WALK_NAV_MAX_LIN_SPEED:-"0.16"}
WALK_NAV_MAX_ANG_SPEED=${WALK_NAV_MAX_ANG_SPEED:-"0.35"}
LEFT_WRIST_POS=${LEFT_WRIST_POS:-"0.18,0.22,0.20"}
LEFT_WRIST_QUAT_WXYZ=${LEFT_WRIST_QUAT_WXYZ:-"1.0,0.0,0.0,0.0"}
RIGHT_WRIST_NAV_POS=${RIGHT_WRIST_NAV_POS:-"0.18,-0.22,0.20"}
RIGHT_WRIST_NAV_QUAT_WXYZ=${RIGHT_WRIST_NAV_QUAT_WXYZ:-"1.0,0.0,0.0,0.0"}

# CuRobo
export PYTHONPATH="${PACK_ROOT}/../BODex/src:${PYTHONPATH:-}"

PYTHON="${ISAAC_PYTHON:-/home/ubuntu/miniconda3/envs/isaaclab_arena/bin/python}"
SPIDER_PYTHON="${SPIDER_PYTHON:-/home/ubuntu/miniconda3/envs/spider/bin/python}"

# ---------------------------------------------------------------------------
# Step 0: Extract scene config from LW-BenchHub
# ---------------------------------------------------------------------------
SCENE_JSON="$OUT_DIR/lwbench_scene_config.json"
mkdir -p "$OUT_DIR"

echo "============================================"
echo "[15_lwbench] Step 0: Extract LW-BenchHub scene config"
echo "============================================"

adapter_cmd=(
  "$PYTHON" "$PACK_ROOT/bridge/lwbench_scene_adapter.py"
  --output-json "$SCENE_JSON"
  --robot-standoff "$ROBOT_STANDOFF"
)

if [[ -n "$LOCAL_USD" ]]; then
  adapter_cmd+=(--local-usd "$LOCAL_USD" --standalone)
  echo "[15_lwbench] Using local USD: $LOCAL_USD"
elif [[ "$MOCK_SCENE" != "0" ]]; then
  adapter_cmd+=(--mock --layout-id "$LAYOUT_ID" --style-id "$STYLE_ID")
  echo "[15_lwbench] Using MOCK scene (layout=$LAYOUT_ID style=$STYLE_ID)"
else
  adapter_cmd+=(--layout-id "$LAYOUT_ID" --style-id "$STYLE_ID")
  echo "[15_lwbench] Using LW-BenchHub layout=$LAYOUT_ID style=$STYLE_ID"
fi

"${adapter_cmd[@]}"

# Parse scene config JSON to extract parameters
SCENE_NAME=$("$PYTHON" -c "
import json, sys
d = json.load(open(sys.argv[1]))
print(d['scene_info']['scene_name'])
" "$SCENE_JSON")

BASE_POS_W=$("$PYTHON" -c "
import json, sys
d = json.load(open(sys.argv[1]))
p = d['scene_config']['default_base_pos_w']
print(f'{p[0]},{p[1]},{p[2]}')
" "$SCENE_JSON")

G1_INIT_YAW_DEG=$("$PYTHON" -c "
import json, sys
d = json.load(open(sys.argv[1]))
print(d['scene_config'].get('default_base_yaw_deg', 0.0))
" "$SCENE_JSON")

WALK_TARGET_OFFSET=$("$PYTHON" -c "
import json, sys
d = json.load(open(sys.argv[1]))
o = d['scene_config']['default_walk_target_offset']
print(f'{o[0]},{o[1]},{o[2]}')
" "$SCENE_JSON")

OBJECT_ALIGN_POS=$("$PYTHON" -c "
import json, sys
d = json.load(open(sys.argv[1]))
p = d['scene_config']['object_align_pos']
print(f'{p[0]},{p[1]},{p[2]}')
" "$SCENE_JSON")

TABLE_Z=$("$PYTHON" -c "
import json, sys
d = json.load(open(sys.argv[1]))
print(d['scene_config']['table_z'])
" "$SCENE_JSON")

RIGHT_WRIST_POS_OBJ=$("$PYTHON" -c "
import json, sys
d = json.load(open(sys.argv[1]))
p = d['scene_config'].get('right_wrist_pos_obj', [-0.18, -0.04, 0.08])
print(f'{p[0]},{p[1]},{p[2]}')
" "$SCENE_JSON")

# Compute trajectory start/end from object align pos
TRAJ_START_POS_W="$OBJECT_ALIGN_POS"
# End position: slight offset for lift_place pattern
TRAJ_END_POS_W=$("$PYTHON" -c "
import json, sys
d = json.load(open(sys.argv[1]))
p = d['scene_config']['object_align_pos']
print(f'{p[0]-0.06},{p[1]+0.10},{p[2]}')
" "$SCENE_JSON")
TRAJ_LIFT_HEIGHT=${TRAJ_LIFT_HEIGHT:-"0.10"}

echo ""
echo "============================================"
echo "[15_lwbench] Scene:   $SCENE_NAME"
echo "[15_lwbench] Object:  $OBJECT"
echo "[15_lwbench] Table Z: $TABLE_Z"
echo "[15_lwbench] G1 start: $BASE_POS_W  yaw=$G1_INIT_YAW_DEG"
echo "[15_lwbench] Walk offset: $WALK_TARGET_OFFSET"
echo "[15_lwbench] Object align: $OBJECT_ALIGN_POS"
echo "[15_lwbench] Traj start/end: $TRAJ_START_POS_W -> $TRAJ_END_POS_W"
echo "============================================"

# ---------------------------------------------------------------------------
# Step 0.5 (optional): Convert Spider SMPL output to kinematic trajectory
# ---------------------------------------------------------------------------
if [[ -n "$SPIDER_NPZ" ]]; then
  echo ""
  echo "============================================"
  echo "[15_lwbench] Step 0.5: Convert Spider SMPL → object trajectory"
  echo "============================================"

  SPIDER_OUTPUT_NPZ="$OUT_DIR/object_kinematic_traj_spider.npz"
  SPIDER_DEBUG_JSON="$OUT_DIR/debug_spider_traj.json"

  spider_cmd=(
    "$SPIDER_PYTHON" "$PACK_ROOT/bridge/smpl_to_g1_spider.py"
    --spider-npz "$SPIDER_NPZ"
    --output-npz "$SPIDER_OUTPUT_NPZ"
    --output-debug-json "$SPIDER_DEBUG_JSON"
    --scene "$SCENE_NAME"
    --object-name "$OBJECT"
  )

  if [[ -n "$SPIDER_XML" ]]; then
    spider_cmd+=(--spider-xml "$SPIDER_XML")
  fi

  if [[ -n "$SPIDER_OBJECT_BODY" ]]; then
    spider_cmd+=(--object-body-name "$SPIDER_OBJECT_BODY")
  fi

  "${spider_cmd[@]}"

  # Use Spider output as HOI pickle equivalent
  # The run_walk_to_grasp_todo.py will use this as Step 1 input
  USE_SPIDER_TRAJ="1"
  echo "[15_lwbench] Spider trajectory: $SPIDER_OUTPUT_NPZ"
else
  USE_SPIDER_TRAJ="0"
fi

# ---------------------------------------------------------------------------
# Step 1: Register scene collision and run walk-to-grasp pipeline
# ---------------------------------------------------------------------------
echo ""
echo "============================================"
echo "[15_lwbench] Step 1: Run walk-to-grasp pipeline"
echo "============================================"

# Build the main pipeline command (reuses run_walk_to_grasp_todo.py)
cmd=(
  "$PYTHON" "$PACK_ROOT/scripts/run_walk_to_grasp_todo.py"
  --out-dir "$OUT_DIR"
  --scene "$SCENE_NAME"
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
  --left-wrist-pos="$LEFT_WRIST_POS"
  --left-wrist-quat-wxyz="$LEFT_WRIST_QUAT_WXYZ"
  --right-wrist-nav-pos="$RIGHT_WRIST_NAV_POS"
  --right-wrist-nav-quat-wxyz="$RIGHT_WRIST_NAV_QUAT_WXYZ"
  --walk-nav-max-lin-speed "$WALK_NAV_MAX_LIN_SPEED"
  --walk-nav-max-ang-speed "$WALK_NAV_MAX_ANG_SPEED"
  --replay-base-height="$REPLAY_BASE_HEIGHT"
  --device "$DEVICE"
  --isaac-python "$PYTHON"
  --no-runner
)

# Register scene collision before running pipeline
# Inject scene config JSON path via environment variable
export LWBENCH_SCENE_JSON="$SCENE_JSON"

if [[ "$MOMAGEN_STYLE" != "0" ]]; then
  cmd+=(
    --momagen-style
    --momagen-start-dist-min "$MOMAGEN_START_DIST_MIN"
    --momagen-start-dist-max "$MOMAGEN_START_DIST_MAX"
    --momagen-target-dist-min "$MOMAGEN_TARGET_DIST_MIN"
    --momagen-target-dist-max "$MOMAGEN_TARGET_DIST_MAX"
    --momagen-min-travel-dist "$MOMAGEN_MIN_TRAVEL_DIST"
    --momagen-max-travel-dist "$MOMAGEN_MAX_TRAVEL_DIST"
    --momagen-sample-attempts "$MOMAGEN_SAMPLE_ATTEMPTS"
    --momagen-sample-seed "$MOMAGEN_SAMPLE_SEED"
  )
fi

if [[ "$HEADLESS" != "0" ]]; then
  cmd+=(--headless)
fi

"${cmd[@]}"

# ---------------------------------------------------------------------------
# Print results
# ---------------------------------------------------------------------------
REPORT="$OUT_DIR/todo_run_report.json"
if [[ -f "$REPORT" ]]; then
  echo ""
  echo "============================================"
  "$PYTHON" -c "
import json, sys
r = json.load(open(sys.argv[1]))
p = r.get('planner', {})
ik = p.get('ik_result')
print('[15_lwbench] Planner used:', p.get('planner_used'))
print('[15_lwbench] CuRobo available:', p.get('curobo_available'))
if ik:
    print('[15_lwbench] IK reachable:', ik.get('reachable'))
    print('[15_lwbench] IK pos_error:', ik.get('position_error'))
else:
    print('[15_lwbench] IK check: skipped')
mg = p.get('motion_gen_result')
if mg:
    print('[15_lwbench] MotionGen success:', mg.get('success'))
    print('[15_lwbench] MotionGen steps:', mg.get('num_steps'))
else:
    print('[15_lwbench] MotionGen: skipped')
print('[15_lwbench] Notes:', p.get('notes'))
print('[15_lwbench] Status:', r.get('status'))
" "$REPORT"
  echo "============================================"
fi

echo ""
echo "[15_lwbench] Done. Output: $OUT_DIR"
echo "[15_lwbench] Scene config: $SCENE_JSON"
if [[ "$USE_SPIDER_TRAJ" == "1" ]]; then
  echo "[15_lwbench] Spider trajectory: $SPIDER_OUTPUT_NPZ"
fi
echo "[15_lwbench] NOTE: Isaac Sim runner was skipped (--no-runner)."
echo "[15_lwbench] To run with Isaac Sim, remove --no-runner from the pipeline command"
echo "[15_lwbench] and ensure the LW-BenchHub scene is registered in IsaacLab-Arena."
