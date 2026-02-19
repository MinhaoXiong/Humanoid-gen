#!/usr/bin/env bash
set -euo pipefail

PACK_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${ISAAC_PYTHON:-$(command -v python3)}"

OUT_DIR="${1:-$PACK_ROOT/artifacts/todo_walk_to_grasp}"
SCENE="${2:-kitchen_pick_and_place}"
OBJECT="${3:-cracker_box}"
PATTERN="${4:-lift_place}"

PLANNER="${PLANNER:-auto}"                  # auto|curobo|open_loop
STRICT_CUROBO="${STRICT_CUROBO:-0}"         # 1 = fail if curobo mode cannot be honored
HEADLESS="${HEADLESS:-0}"                   # 1 = run Isaac in headless mode
NO_RUNNER="${NO_RUNNER:-0}"                 # 1 = only build replay artifacts
DEVICE="${DEVICE:-auto}"                    # auto picks the most idle GPU
MAX_STEPS="${MAX_STEPS:-}"
BASE_POS_W="${BASE_POS_W:-}"                # optional override, e.g. -0.55,0.0,0.0
BASE_YAW_DEG="${BASE_YAW_DEG:-0.0}"

WALK_TARGET_BASE_POS_W="${WALK_TARGET_BASE_POS_W:-}"
WALK_TARGET_OFFSET_OBJ_W="${WALK_TARGET_OFFSET_OBJ_W:-}"
WALK_TARGET_OFFSET_FRAME="${WALK_TARGET_OFFSET_FRAME:-object}"
WALK_TARGET_YAW_MODE="${WALK_TARGET_YAW_MODE:-face_object}"
WALK_TARGET_YAW_DEG="${WALK_TARGET_YAW_DEG:-0.0}"
WALK_NAV_MAX_LIN_SPEED="${WALK_NAV_MAX_LIN_SPEED:-0.22}"
WALK_NAV_MAX_ANG_SPEED="${WALK_NAV_MAX_ANG_SPEED:-0.55}"
WALK_NAV_DT="${WALK_NAV_DT:-0.02}"
WALK_PREGRASP_HOLD_STEPS="${WALK_PREGRASP_HOLD_STEPS:-25}"

RIGHT_WRIST_POS_OBJ="${RIGHT_WRIST_POS_OBJ:-"-0.20,-0.03,0.10"}"
RIGHT_WRIST_QUAT_OBJ_WXYZ="${RIGHT_WRIST_QUAT_OBJ_WXYZ:-"0.70710678,0.0,-0.70710678,0.0"}"
RIGHT_WRIST_QUAT_CONTROL="${RIGHT_WRIST_QUAT_CONTROL:-constant_pelvis}"
RIGHT_WRIST_QUAT_PELVIS_WXYZ="${RIGHT_WRIST_QUAT_PELVIS_WXYZ:-"1.0,0.0,0.0,0.0"}"

CEDEX_GRASP_PT="${CEDEX_GRASP_PT:-}"
CEDEX_GRASP_INDEX="${CEDEX_GRASP_INDEX:-0}"
CEDEX_WRIST_POS_OFFSET="${CEDEX_WRIST_POS_OFFSET:-"0.0,0.0,0.0"}"
CEDEX_WRIST_QUAT_OFFSET_WXYZ="${CEDEX_WRIST_QUAT_OFFSET_WXYZ:-"1.0,0.0,0.0,0.0"}"

pick_idle_gpu() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    return 1
  fi
  local best_idx
  best_idx=$(
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits 2>/dev/null \
      | awk -F',' '
          $1 ~ /^[[:space:]]*[0-9]+[[:space:]]*$/ {
            gsub(/[[:space:]]+/, "", $1);
            gsub(/[[:space:]]+/, "", $2);
            gsub(/[[:space:]]+/, "", $3);
            score = ($2 + 0) * 1000 + ($3 + 0);
            if (!seen || score < best_score) {
              best_score = score;
              best_idx = $1;
              seen = 1;
            }
          }
          END {
            if (seen && best_idx != "") {
              print best_idx;
            }
          }'
  )
  [[ -n "$best_idx" ]] || return 1
  echo "cuda:${best_idx}"
}

if [[ "$DEVICE" == "auto" ]]; then
  DEVICE="$(pick_idle_gpu || true)"
  if [[ -z "$DEVICE" ]]; then
    echo "[todo_pipeline] Failed to detect idle GPU with nvidia-smi. Please set DEVICE=cuda:<id> explicitly." >&2
    exit 2
  fi
fi
if [[ "$DEVICE" == cpu* ]]; then
  echo "[todo_pipeline] DEVICE must be a CUDA device, got: $DEVICE" >&2
  exit 2
fi
echo "[todo_pipeline] DEVICE=$DEVICE"

cmd=(
  "$PYTHON" "$PACK_ROOT/scripts/run_walk_to_grasp_todo.py"
  --pack-root "$PACK_ROOT"
  --out-dir "$OUT_DIR"
  --scene "$SCENE"
  --object "$OBJECT"
  --pattern "$PATTERN"
  --planner "$PLANNER"
  --base-yaw-deg "$BASE_YAW_DEG"
  --walk-target-offset-frame "$WALK_TARGET_OFFSET_FRAME"
  --walk-target-yaw-mode "$WALK_TARGET_YAW_MODE"
  --walk-target-yaw-deg "$WALK_TARGET_YAW_DEG"
  --walk-nav-max-lin-speed "$WALK_NAV_MAX_LIN_SPEED"
  --walk-nav-max-ang-speed "$WALK_NAV_MAX_ANG_SPEED"
  --walk-nav-dt "$WALK_NAV_DT"
  --walk-pregrasp-hold-steps "$WALK_PREGRASP_HOLD_STEPS"
  --right-wrist-pos-obj="$RIGHT_WRIST_POS_OBJ"
  --right-wrist-quat-obj-wxyz="$RIGHT_WRIST_QUAT_OBJ_WXYZ"
  --right-wrist-quat-control "$RIGHT_WRIST_QUAT_CONTROL"
  --right-wrist-quat-pelvis-wxyz="$RIGHT_WRIST_QUAT_PELVIS_WXYZ"
  --device "$DEVICE"
)

if [[ "$STRICT_CUROBO" == "1" ]]; then
  cmd+=(--strict-curobo)
fi
if [[ "$HEADLESS" == "1" ]]; then
  cmd+=(--headless)
fi
if [[ "$NO_RUNNER" == "1" ]]; then
  cmd+=(--no-runner)
fi
if [[ -n "$MAX_STEPS" ]]; then
  cmd+=(--max-steps "$MAX_STEPS")
fi
if [[ -n "$BASE_POS_W" ]]; then
  cmd+=(--base-pos-w="$BASE_POS_W")
fi
if [[ -n "$WALK_TARGET_BASE_POS_W" ]]; then
  cmd+=(--walk-target-base-pos-w="$WALK_TARGET_BASE_POS_W")
fi
if [[ -n "$WALK_TARGET_OFFSET_OBJ_W" ]]; then
  cmd+=(--walk-target-offset-obj-w="$WALK_TARGET_OFFSET_OBJ_W")
fi
if [[ -n "$CEDEX_GRASP_PT" ]]; then
  cmd+=(
    --cedex-grasp-pt "$CEDEX_GRASP_PT"
    --cedex-grasp-index "$CEDEX_GRASP_INDEX"
    --cedex-wrist-pos-offset="$CEDEX_WRIST_POS_OFFSET"
    --cedex-wrist-quat-offset-wxyz="$CEDEX_WRIST_QUAT_OFFSET_WXYZ"
  )
fi

"${cmd[@]}"
