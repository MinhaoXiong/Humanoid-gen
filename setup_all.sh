#!/usr/bin/env bash
# Humanoid-gen-pack 一键初始化
# 用法: bash setup_all.sh
set -euo pipefail

PACK_ROOT="$(cd "$(dirname "$0")" && pwd)"

echo "============================================"
echo "  Humanoid-gen-pack Setup"
echo "============================================"
echo ""

# ---- Step 1: 初始化 submodule ----
echo "[1/4] 初始化 git submodule ..."
cd "$PACK_ROOT"
git submodule update --init --recursive
echo "      done."

# ---- Step 2: Apply patches ----
echo "[2/4] 应用上游修改 patch ..."

if [ -f "$PACK_ROOT/patches/hoifhli_release.patch" ]; then
    cd "$PACK_ROOT/repos/hoifhli_release"
    if git apply --check "$PACK_ROOT/patches/hoifhli_release.patch" 2>/dev/null; then
        git apply "$PACK_ROOT/patches/hoifhli_release.patch"
        echo "      hoifhli_release.patch applied."
    else
        echo "      hoifhli_release.patch already applied or conflict, skipping."
    fi
fi

if [ -f "$PACK_ROOT/patches/IsaacLab-Arena-modified.patch" ]; then
    cd "$PACK_ROOT/repos/IsaacLab-Arena"
    if git apply --check "$PACK_ROOT/patches/IsaacLab-Arena-modified.patch" 2>/dev/null; then
        git apply "$PACK_ROOT/patches/IsaacLab-Arena-modified.patch"
        echo "      IsaacLab-Arena-modified.patch applied."
    else
        echo "      IsaacLab-Arena-modified.patch already applied or conflict, skipping."
    fi
fi

if [ -f "$PACK_ROOT/patches/IsaacLab-Arena-new-files.patch" ]; then
    cd "$PACK_ROOT/repos/IsaacLab-Arena"
    if git apply --check "$PACK_ROOT/patches/IsaacLab-Arena-new-files.patch" 2>/dev/null; then
        git apply "$PACK_ROOT/patches/IsaacLab-Arena-new-files.patch"
        echo "      IsaacLab-Arena-new-files.patch applied."
    else
        echo "      IsaacLab-Arena-new-files.patch already applied or conflict, skipping."
    fi
fi

# ---- Step 3: Conda 环境 ----
echo "[3/4] Conda 环境提示 ..."
echo ""
echo "  HOIFHLI 环境:"
echo "    conda env create -f $PACK_ROOT/envs/hoifhli_env.yml"
echo ""
echo "  IsaacLab-Arena 环境:"
echo "    bash $PACK_ROOT/envs/setup_isaaclab_arena.sh"
echo ""

# ---- Step 4: 数据下载 ----
echo "[4/4] 数据下载提示 ..."
echo ""
echo "  请编辑 $PACK_ROOT/data/download_data.sh 填入下载地址后运行："
echo "    bash $PACK_ROOT/data/download_data.sh"
echo ""
echo "  或手动将以下数据放到 repos/hoifhli_release/ 下："
echo "    experiments/   (~380M, 模型权重)"
echo "    data/processed_data/  (~37G)"
echo "    data/smpl_all_models/ (~1.9G)"
echo ""

echo "============================================"
echo "  Setup 完成！"
echo "============================================"
echo ""
echo "执行流程："
echo "  1) bash scripts/01_hoi_sample.sh"
echo "  2) bash scripts/03_build_replay.sh <hoi_pkl> <output_dir>"
echo "  3) bash scripts/04_isaac_replay_video.sh <output_dir>"
