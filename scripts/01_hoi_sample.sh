#!/usr/bin/env bash
set -euo pipefail

cd /home/ubuntu/DATA2/workspace/xmh/hoifhli_release
/home/ubuntu/miniconda3/envs/hoifhli_env/bin/bash scripts/sample.sh

find /home/ubuntu/DATA2/workspace/xmh/hoifhli_release/results -name human_object_results.pkl
