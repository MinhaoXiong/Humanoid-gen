#!/usr/bin/env bash
set -euo pipefail

PACK_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${ISAAC_PYTHON:-$(command -v python3)}"

OBJECT_NAME="${1:-003_cracker_box}"
DATASET="${DATASET:-ycb}"
DATASET_PATH="${DATASET_PATH:-}"
NUM_PARTICLES="${NUM_PARTICLES:-64}"
MAX_ITER="${MAX_ITER:-300}"
SAVE_TOP_K="${SAVE_TOP_K:-16}"
LR="${LR:-1e-2}"
DEVICE="${DEVICE:-}"
OUTPUT_PT="${OUTPUT_PT:-}"
OUTPUT_JSON="${OUTPUT_JSON:-}"
CEDEX_ROOT="${CEDEX_ROOT:-/home/ubuntu/DATA2/workspace/xmh/CEDex-Grasp}"

cmd=(
  "$PYTHON" "$PACK_ROOT/scripts/generate_inspirehand_grasps.py"
  --cedex-root "$CEDEX_ROOT"
  --dataset "$DATASET"
  --object-name "$OBJECT_NAME"
  --num-particles "$NUM_PARTICLES"
  --max-iter "$MAX_ITER"
  --save-top-k "$SAVE_TOP_K"
  --learning-rate "$LR"
)

if [[ -n "$DATASET_PATH" ]]; then
  cmd+=(--dataset-path "$DATASET_PATH")
fi
if [[ -n "$DEVICE" ]]; then
  cmd+=(--device "$DEVICE")
fi
if [[ -n "$OUTPUT_PT" ]]; then
  cmd+=(--output-pt "$OUTPUT_PT")
fi
if [[ -n "$OUTPUT_JSON" ]]; then
  cmd+=(--output-json "$OUTPUT_JSON")
fi

"${cmd[@]}"
