#!/usr/bin/env bash
# Collect a reproducible debug bundle for walk-to-grasp GUI runs.
set -euo pipefail

PACK_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT_DIR=${1:-"$PACK_ROOT/artifacts/todo_debug_bundle"}
PATTERN=${2:-"lift_place"}
SCENE=${3:-"kitchen_pick_and_place"}
OBJECT=${4:-"cracker_box"}

export RUNNER_DEBUG_DIR="${RUNNER_DEBUG_DIR:-$OUT_DIR/runner_debug}"
export RUNNER_DEBUG_EVERY="${RUNNER_DEBUG_EVERY:-1}"
export RUNNER_DEBUG_MAX_STEPS="${RUNNER_DEBUG_MAX_STEPS:-2000}"

mkdir -p "$OUT_DIR" "$RUNNER_DEBUG_DIR"

SNAPSHOT="$OUT_DIR/env_snapshot.txt"
if command -v rg >/dev/null 2>&1; then
  SEARCH_CMD="rg -n"
else
  SEARCH_CMD="grep -nE"
fi
{
  echo "date=$(date -Iseconds)"
  echo "pack_root=$PACK_ROOT"
  echo "python=${ISAAC_PYTHON:-<default from 13 script>}"
  echo "runner_debug_dir=$RUNNER_DEBUG_DIR"
  echo "runner_debug_every=$RUNNER_DEBUG_EVERY"
  echo "runner_debug_max_steps=$RUNNER_DEBUG_MAX_STEPS"
  echo ""
  echo "[git]"
  git -C "$PACK_ROOT" rev-parse --short HEAD
  git -C "$PACK_ROOT/repos/IsaacLab-Arena" rev-parse --short HEAD
  echo ""
  echo "[mapping checks]"
  $SEARCH_CMD "actions\\[:, -7:\\]" "$PACK_ROOT/repos/IsaacLab-Arena/isaaclab_arena/embodiments/g1/g1.py" || true
  $SEARCH_CMD "_LEGACY_HAND_TO_INSPIRE_HAND|_INSPIRE_HAND_TO_LEGACY_HAND" \
    "$PACK_ROOT/repos/IsaacLab-Arena/isaaclab_arena_g1/g1_whole_body_controller/wbc_policy/run_policy.py" || true
} >"$SNAPSHOT" 2>&1 || true

echo "[debug_bundle] Running walk-to-grasp..."
bash "$PACK_ROOT/scripts/13_todo_walk_to_grasp_gui.sh" "$OUT_DIR" "$PATTERN" "$SCENE" "$OBJECT"

FILES=(
  "env_snapshot.txt"
  "todo_run_report.json"
  "debug_traj.json"
  "debug_replay.json"
  "replay_actions_arm_follow.hdf5"
  "runner_debug/runner_debug_meta.json"
  "runner_debug/runner_debug_steps.jsonl"
)

EXISTING=()
for f in "${FILES[@]}"; do
  if [[ -e "$OUT_DIR/$f" ]]; then
    EXISTING+=("$f")
  fi
done

BUNDLE="$OUT_DIR/debug_bundle.tar.gz"
if [[ ${#EXISTING[@]} -gt 0 ]]; then
  tar -czf "$BUNDLE" -C "$OUT_DIR" "${EXISTING[@]}"
  echo "[debug_bundle] Created: $BUNDLE"
else
  echo "[debug_bundle] Warning: no expected files found under $OUT_DIR"
fi

echo "[debug_bundle] Done. Output dir: $OUT_DIR"
