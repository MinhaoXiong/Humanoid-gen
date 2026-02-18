#!/usr/bin/env bash
set -euo pipefail

PACK_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

HOI_PKL=${1:-"/path/to/human_object_results.pkl"}
OUT_DIR=${2:-"$PACK_ROOT/artifacts/g1_bridge_run1"}

mkdir -p "$OUT_DIR"

ISAAC_ROOT="$PACK_ROOT/repos/IsaacLab-Arena"
PYTHON="${ISAAC_PYTHON:-$(command -v python3)}"

cd "$ISAAC_ROOT"
"$PYTHON" "$PACK_ROOT/bridge/build_replay.py" \
  --hoi-pickle "$HOI_PKL" \
  --output-hdf5 "$OUT_DIR/replay_actions.hdf5" \
  --output-object-traj "$OUT_DIR/object_kinematic_traj.npz" \
  --output-debug-json "$OUT_DIR/bridge_debug.json" \
  --hoi-fps 30 \
  --target-fps 50 \
  --yaw-face-object
