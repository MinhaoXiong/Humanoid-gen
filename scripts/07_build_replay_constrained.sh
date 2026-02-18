#!/usr/bin/env bash
set -euo pipefail

PACK_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

HOI_PKL=${1:-"/path/to/human_object_results.pkl"}
OUT_DIR=${2:-"$PACK_ROOT/artifacts/g1_bridge_constrained"}

mkdir -p "$OUT_DIR"

PYTHON="${ISAAC_PYTHON:-$(command -v python3)}"

# Defaults tuned for galileo_g1_locomanip_pick_and_place.
TRJ_SCALE_XYZ=${TRJ_SCALE_XYZ:-"1.0,1.0,1.0"}
TRJ_OFFSET_W=${TRJ_OFFSET_W:-"0.0,0.0,0.0"}
ALIGN_FIRST_POS_W=${ALIGN_FIRST_POS_W:-"0.5785,0.18,0.0707"}
ALIGN_LAST_POS_W=${ALIGN_LAST_POS_W:-""}
ALIGN_LAST_RAMP_SEC=${ALIGN_LAST_RAMP_SEC:-"1.0"}
CLIP_Z_MIN=${CLIP_Z_MIN:-"0.06"}
CLIP_Z_MAX=${CLIP_Z_MAX:-"0.40"}
CLIP_XY_MIN=${CLIP_XY_MIN:-""}
CLIP_XY_MAX=${CLIP_XY_MAX:-""}

cmd=(
  "$PYTHON" "$PACK_ROOT/bridge/build_replay.py"
  --hoi-pickle "$HOI_PKL"
  --output-hdf5 "$OUT_DIR/replay_actions.hdf5"
  --output-object-traj "$OUT_DIR/object_kinematic_traj.npz"
  --output-debug-json "$OUT_DIR/bridge_debug.json"
  --hoi-fps 30
  --target-fps 50
  --yaw-face-object
  --traj-scale-xyz "$TRJ_SCALE_XYZ"
  --traj-offset-w "$TRJ_OFFSET_W"
  --align-first-pos-w "$ALIGN_FIRST_POS_W"
  --align-last-ramp-sec "$ALIGN_LAST_RAMP_SEC"
  --clip-z-min "$CLIP_Z_MIN"
  --clip-z-max "$CLIP_Z_MAX"
)

if [[ -n "$ALIGN_LAST_POS_W" ]]; then
  cmd+=(--align-last-pos-w "$ALIGN_LAST_POS_W")
fi
if [[ -n "$CLIP_XY_MIN" ]]; then
  cmd+=(--clip-xy-min "$CLIP_XY_MIN")
fi
if [[ -n "$CLIP_XY_MAX" ]]; then
  cmd+=(--clip-xy-max "$CLIP_XY_MAX")
fi

"${cmd[@]}"
