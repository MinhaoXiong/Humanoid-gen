#!/usr/bin/env bash
set -euo pipefail

PACK_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

OUT_DIR=${1:-"$PACK_ROOT/artifacts/g1_bridge_run1"}
VIDEO_PREFIX=${2:-"g1_bridge_run1"}
VIDEO_DIR=${3:-"$PACK_ROOT/artifacts/videos"}

mkdir -p "$VIDEO_DIR"

ISAAC_ROOT="$PACK_ROOT/repos/IsaacLab-Arena"
PYTHON="${ISAAC_PYTHON:-$(command -v python3)}"

cd "$ISAAC_ROOT"
"$PYTHON" "$PACK_ROOT/isaac_replay/policy_runner_kinematic_object_replay.py" \
  --headless --device cpu --enable_cameras \
  --policy_type replay \
  --replay_file_path "$OUT_DIR/replay_actions.hdf5" \
  --episode_name demo_0 \
  --kin-traj-path "$OUT_DIR/object_kinematic_traj.npz" \
  --kin-apply-timing pre_step \
  --use-hoi-object \
  --hoi-root "$PACK_ROOT/repos/hoifhli_release" \
  --hoi-usd-cache-dir "$PACK_ROOT/artifacts/hoi_runtime_usd" \
  --max-steps 408 \
  --save-video \
  --video-output-dir "$VIDEO_DIR" \
  --video-prefix "$VIDEO_PREFIX" \
  galileo_g1_locomanip_pick_and_place \
  --embodiment g1_wbc_pink
