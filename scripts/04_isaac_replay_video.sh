#!/usr/bin/env bash
set -euo pipefail

OUT_DIR=${1:-"/tmp/g1_bridge_run1"}
VIDEO_PREFIX=${2:-"g1_bridge_run1"}
VIDEO_DIR=${3:-"/home/ubuntu/DATA2/workspace/xmh/IsaacLab-Arena/.workflow_data/videos"}

mkdir -p "$VIDEO_DIR"

cd /home/ubuntu/DATA2/workspace/xmh/IsaacLab-Arena
/home/ubuntu/miniconda3/envs/isaaclab_arena/bin/python isaaclab_arena/examples/policy_runner_kinematic_object_replay.py \
  --headless --device cpu --enable_cameras \
  --policy_type replay \
  --replay_file_path "$OUT_DIR/replay_actions.hdf5" \
  --episode_name demo_0 \
  --kin-traj-path "$OUT_DIR/object_kinematic_traj.npz" \
  --kin-asset-name brown_box \
  --kin-apply-timing pre_step \
  --max-steps 408 \
  --save-video \
  --video-output-dir "$VIDEO_DIR" \
  --video-prefix "$VIDEO_PREFIX" \
  galileo_g1_locomanip_pick_and_place \
  --object brown_box \
  --embodiment g1_wbc_pink
