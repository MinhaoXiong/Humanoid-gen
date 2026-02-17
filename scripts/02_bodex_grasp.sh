#!/usr/bin/env bash
set -euo pipefail

export BODEX_ROOT=/home/ubuntu/DATA2/workspace/xmh/BODex
export PYTHONPATH=$BODEX_ROOT/src:${PYTHONPATH:-}
export LD_LIBRARY_PATH=/home/ubuntu/miniconda3/envs/objdex/lib/python3.8/site-packages/cmeel.prefix/lib:${LD_LIBRARY_PATH:-}

cd "$BODEX_ROOT"
/home/ubuntu/miniconda3/envs/objdex/bin/python example_grasp/plan_batch_env.py \
  -c sim_shadow/fc.yml \
  -w 1 \
  -m npy \
  -f run_hoi_bridge

find "$BODEX_ROOT/src/curobo/content/assets/output/run_hoi_bridge" -name '*grasp.npy'
