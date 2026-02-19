#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PACK_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

MESH_FILE="${1:?Usage: $0 <mesh_file> [output_pt]}"
OUTPUT_PT="${2:-}"

ARGS=(--mesh-file "$MESH_FILE")
[ -n "$OUTPUT_PT" ] && ARGS+=(--output-pt "$OUTPUT_PT")

exec python3 "$SCRIPT_DIR/generate_bodex_inspirehand_grasps.py" "${ARGS[@]}"
