#!/usr/bin/env bash
# Deploy InspireHand assets into a BODex installation.
# Usage: bash scripts/setup_bodex_inspirehand.sh [BODEX_ROOT]
#   BODEX_ROOT defaults to ../BODex relative to Humanoid-gen-pack.
set -euo pipefail

PACK_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BODEX_ROOT="${1:-$(cd "$PACK_ROOT/.." && pwd)/BODex}"
ASSETS="$PACK_ROOT/configs/bodex_inspire_hand_assets"

if [[ ! -d "$BODEX_ROOT/src/curobo" ]]; then
  echo "[ERROR] BODex not found at $BODEX_ROOT"
  echo "  Clone it first: git clone https://github.com/JYChen18/BODex.git $BODEX_ROOT"
  exit 1
fi

ROBOT_DIR="$BODEX_ROOT/src/curobo/content/assets/robot"
CFG_DIR="$BODEX_ROOT/src/curobo/content/configs/robot"

echo "[setup] Deploying InspireHand assets to $BODEX_ROOT ..."

# 1. URDF + meshes
cp -r "$ASSETS/inspire_hand_description" "$ROBOT_DIR/"

# 2. CuRobo robot config
cp "$ASSETS/inspire_hand.yml" "$CFG_DIR/"

# 3. Hand pose transfer config
mkdir -p "$CFG_DIR/hand_pose_transfer"
cp "$ASSETS/hand_pose_transfer_inspire_hand.yml" "$CFG_DIR/hand_pose_transfer/inspire_hand.yml"

# 4. Collision spheres config
mkdir -p "$CFG_DIR/spheres"
cp "$ASSETS/spheres_inspire_hand.yml" "$CFG_DIR/spheres/inspire_hand.yml"

echo "[setup] Done. Deployed files:"
echo "  $ROBOT_DIR/inspire_hand_description/"
echo "  $CFG_DIR/inspire_hand.yml"
echo "  $CFG_DIR/hand_pose_transfer/inspire_hand.yml"
echo "  $CFG_DIR/spheres/inspire_hand.yml"
