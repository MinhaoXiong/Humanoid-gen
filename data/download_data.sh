#!/usr/bin/env bash
# HOIFHLI 数据下载脚本
# 需要下载的数据：
#   1. experiments/  (~380M) - 模型权重（navi_release, cnet_release, rnet_release, fnet_release）
#   2. data/processed_data/ (~37G) - 训练/推理数据
#   3. data/smpl_all_models/ (~1.9G) - SMPL 模型文件
#
# 请将下载地址填入下方变量，然后运行此脚本。
set -euo pipefail

PACK_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
HOI_ROOT="$PACK_ROOT/repos/hoifhli_release"

# ============================================================
# 请在此处填入实际下载地址（HuggingFace / 网盘 / scp 等）
# ============================================================
EXPERIMENTS_URL=""    # 模型权重压缩包 URL
PROCESSED_DATA_URL="" # processed_data 压缩包 URL
SMPL_MODELS_URL=""    # SMPL 模型压缩包 URL

download_and_extract() {
    local url="$1"
    local target_dir="$2"
    local name="$3"

    if [ -z "$url" ]; then
        echo "[SKIP] $name: 未设置下载地址，请手动放置到 $target_dir"
        return
    fi

    echo "[DOWN] $name -> $target_dir"
    mkdir -p "$target_dir"

    if command -v wget &>/dev/null; then
        wget -q --show-progress -O "/tmp/_download_tmp.tar.gz" "$url"
    elif command -v curl &>/dev/null; then
        curl -L -o "/tmp/_download_tmp.tar.gz" "$url"
    else
        echo "[ERROR] 需要 wget 或 curl"
        return 1
    fi

    tar -xzf "/tmp/_download_tmp.tar.gz" -C "$target_dir"
    rm -f "/tmp/_download_tmp.tar.gz"
    echo "[DONE] $name"
}

echo "=== HOIFHLI 数据下载 ==="
echo "目标目录: $HOI_ROOT"
echo ""

download_and_extract "$EXPERIMENTS_URL" "$HOI_ROOT/experiments" "模型权重 (experiments)"
download_and_extract "$PROCESSED_DATA_URL" "$HOI_ROOT/data/processed_data" "训练数据 (processed_data)"
download_and_extract "$SMPL_MODELS_URL" "$HOI_ROOT/data/smpl_all_models" "SMPL 模型 (smpl_all_models)"

echo ""
echo "=== 下载完成 ==="
echo "如有未下载项，请手动将数据放到对应目录。"
echo "目录结构："
echo "  $HOI_ROOT/experiments/{navi_release,cnet_release,rnet_release,fnet_release}"
echo "  $HOI_ROOT/data/processed_data/"
echo "  $HOI_ROOT/data/smpl_all_models/"
