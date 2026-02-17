#!/usr/bin/env bash
# IsaacLab-Arena 环境安装指引
# 前置条件：NVIDIA Isaac Sim 已安装（需要 GPU + NVIDIA 驱动）
set -euo pipefail

PACK_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ISAAC_ROOT="$PACK_ROOT/repos/IsaacLab-Arena"

echo "=== IsaacLab-Arena 环境安装 ==="
echo ""
echo "1. 确保已安装 Isaac Sim (https://docs.isaacsim.omniverse.nvidia.com/)"
echo ""
echo "2. 创建 conda 环境并安装 IsaacLab："
echo "   cd $ISAAC_ROOT/submodules/IsaacLab"
echo "   ./isaaclab.sh --install"
echo ""
echo "3. 安装 IsaacLab-Arena："
echo "   cd $ISAAC_ROOT"
echo "   pip install -e ."
echo ""
echo "4. 安装 G1 WBC 控制器："
echo "   cd $ISAAC_ROOT/isaaclab_arena_g1"
echo "   pip install -e ."
echo ""
echo "详细文档参考: $ISAAC_ROOT/docs/"
