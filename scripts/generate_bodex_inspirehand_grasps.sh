#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

MESH_FILE="${1:?Usage: $0 <mesh_file> [spider_npz] [output_pt]}"
SPIDER_NPZ="${2:-}"
OUTPUT_PT="${3:-}"

ARGS=(--mesh-file "$MESH_FILE")
[ -n "$SPIDER_NPZ" ] && ARGS+=(--seed-from-spider "$SPIDER_NPZ" --rank-by-human)
[ -n "$OUTPUT_PT" ] && ARGS+=(--output-pt "$OUTPUT_PT")

exec python3 "$SCRIPT_DIR/generate_bodex_inspirehand_grasps.py" "${ARGS[@]}"
