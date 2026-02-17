# Humanoid-gen-pack

HOIFHLI + G1 replay 最小可运行链路。

在 IsaacLab 仿真中，将 HOIFHLI 生成的人-物交互轨迹转化为 Unitree G1 机器人的回放动作，并导出视频。

## 目录结构

```
├── bridge/build_replay.py              HOI → G1 23D 回放动作
├── isaac_replay/                       Isaac 回放脚本
├── scripts/
│   ├── 01_hoi_sample.sh                HOI 采样
│   ├── 03_build_replay.sh              构建回放文件
│   └── 04_isaac_replay_video.sh        回放 + 视频导出
├── repos/
│   ├── hoifhli_release/                [submodule] HOI 生成模型
│   └── IsaacLab-Arena/                 [submodule] Isaac 仿真平台
├── patches/                            上游仓库的本地修改
├── envs/                               Conda 环境配置
├── data/download_data.sh               数据下载脚本（需填入地址）
├── docs/                               详细文档
└── setup_all.sh                        一键初始化
```

## 快速开始

### 1. Clone

```bash
git clone --recursive https://github.com/MinhaoXiong/Humanoid-gen.git
cd Humanoid-gen
```

### 2. 初始化

```bash
bash setup_all.sh
```

这会：初始化 submodule → apply patch → 提示创建 conda 环境和下载数据。

### 3. 创建环境

```bash
# HOIFHLI 环境
conda env create -f envs/hoifhli_env.yml

# IsaacLab-Arena 环境（需要 Isaac Sim）
bash envs/setup_isaaclab_arena.sh
```

### 4. 下载数据

编辑 `data/download_data.sh` 填入下载地址后运行，或手动将数据放到 `repos/hoifhli_release/` 下：
- `experiments/` (~380M, 模型权重)
- `data/processed_data/` (~37G)
- `data/smpl_all_models/` (~1.9G)

### 5. 执行

```bash
# Step 1: HOI 生成
HOI_PYTHON=/path/to/hoifhli_env/bin/python bash scripts/01_hoi_sample.sh

# Step 2: 构建 G1 回放
ISAAC_PYTHON=/path/to/isaaclab_arena/bin/python \
  bash scripts/03_build_replay.sh /path/to/human_object_results.pkl /tmp/run1

# Step 3: Isaac 回放 + 视频
ISAAC_PYTHON=/path/to/isaaclab_arena/bin/python \
  bash scripts/04_isaac_replay_video.sh /tmp/run1
```

视频输出到 `output/videos/`。

## Pipeline

```
HOIFHLI (文本 → 人物交互轨迹)
    ↓ human_object_results.pkl (obj_pos, obj_rot_mat)
build_replay.py (物体轨迹 → G1 23D action)
    ↓ replay_actions.hdf5 + object_kinematic_traj.npz
IsaacLab-Arena (运动学回放 + 视频)
```

## 环境变量

脚本通过环境变量指定 Python 解释器：
- `HOI_PYTHON`：HOIFHLI 环境的 python（默认 `python3`）
- `ISAAC_PYTHON`：IsaacLab-Arena 环境的 python（默认 `python3`）

## 文档

- `docs/END_TO_END_FROM_HOI_BODEX_CN.md` — 端到端流程说明
- `docs/PROJECT_STATUS_CN.md` — 项目状态
- `docs/README_bridge.md` — bridge 脚本详解
