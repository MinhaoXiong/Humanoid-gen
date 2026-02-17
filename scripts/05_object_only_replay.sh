#!/usr/bin/env bash
set -euo pipefail

PACK_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

OUT_DIR=${1:-"$PACK_ROOT/artifacts/g1_bridge_run1"}
VIDEO_PREFIX=${2:-"object_only"}
VIDEO_DIR=${3:-"$PACK_ROOT/artifacts/videos"}

mkdir -p "$VIDEO_DIR"

ISAAC_ROOT="$PACK_ROOT/repos/IsaacLab-Arena"
PYTHON="${ISAAC_PYTHON:-$(command -v python3)}"

cd "$ISAAC_ROOT"
"$PYTHON" isaaclab_arena/examples/policy_runner_kinematic_object_replay.py \
  --headless --device cpu --enable_cameras \
  --object-only \
  --kin-traj-path "$OUT_DIR/object_kinematic_traj.npz" \
  --kin-apply-timing pre_step \
  --use-hoi-object \
  --hoi-root "$PACK_ROOT/repos/hoifhli_release" \
  --hoi-usd-cache-dir "$PACK_ROOT/artifacts/hoi_runtime_usd" \
  --save-video \
  --save-third-person \
  --video-output-dir "$VIDEO_DIR" \
  --video-prefix "$VIDEO_PREFIX" \
  galileo_g1_locomanip_pick_and_place \
  --embodiment g1_wbc_pink
