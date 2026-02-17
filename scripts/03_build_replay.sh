#!/usr/bin/env bash
set -euo pipefail

HOI_PKL=${1:-"/path/to/human_object_results.pkl"}
GRASP_NPY=${2:-"/path/to/grasp.npy"}
OUT_DIR=${3:-"/tmp/g1_bridge_run1"}

mkdir -p "$OUT_DIR"

cd /home/ubuntu/DATA2/workspace/xmh/IsaacLab-Arena
/home/ubuntu/miniconda3/envs/isaaclab_arena/bin/python tools/hoi_bodex_g1_bridge/build_replay.py \
  --hoi-pickle "$HOI_PKL" \
  --bodex-grasp-npy "$GRASP_NPY" \
  --output-hdf5 "$OUT_DIR/replay_actions.hdf5" \
  --output-object-traj "$OUT_DIR/object_kinematic_traj.npz" \
  --output-debug-json "$OUT_DIR/bridge_debug.json" \
  --hoi-fps 30 \
  --target-fps 50 \
  --yaw-face-object
