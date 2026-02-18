#!/usr/bin/env bash
set -euo pipefail

PACK_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEVICE="${DEVICE:-cuda:0}"

HEADLESS=0 DEVICE="$DEVICE" \
bash "$PACK_ROOT/scripts/11_debug_arm_follow_from_hoidini_headless.sh" "$@"
