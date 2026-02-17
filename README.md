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

## 转化流程详解：HOIFHLI → G1 回放动作

整个转化发生在 `bridge/build_replay.py` 中，分为 6 个阶段。

核心思路：**不做人体→机器人的运动重定向**，而是取 HOIFHLI 的物体轨迹作为"地面真值"，在物体坐标系下定义固定的手抓取位姿，沿物体轨迹展开并经过坐标变换链生成 G1 控制器能直接消费的 23D action 序列。

### 阶段 1：加载 HOI 物体轨迹

HOIFHLI 输出的 `human_object_results.pkl` 包含人体全身运动和物体运动，但 `build_replay.py` **只取物体轨迹**：
- `obj_pos` [T, 3] — 物体质心世界坐标
- `obj_rot_mat` [T, 3, 3] — 物体旋转矩阵

人体关节数据完全不用。

### 阶段 2：帧率重采样

```
30 FPS (HOIFHLI) → 50 FPS (Isaac Sim)
```

- 位置：线性插值（`np.interp`）
- 旋转：转为四元数后 SLERP 球面插值

### 阶段 3：确定手相对物体的抓取位姿

定义手在**物体坐标系**下的 pregrasp（准备抓）和 grasp（抓住）两个位姿。来源：
- BODex 求解（可选，当前未使用）
- 手动指定（默认）：pregrasp `(-0.35, -0.08, 0.10)`，grasp `(-0.28, -0.05, 0.06)`

这两个位姿是相对于物体的，物体怎么动，手就跟着怎么动。

### 阶段 4：规划 base 轨迹 + 导航指令

对每帧，在物体坐标系下施加固定偏移（base 在物体正后方 0.55m），转到世界坐标系得到 base 位置。base yaw 始终朝向物体。对相邻帧差分得到速度指令 `[vx, vy, wz]`，转到 base 局部坐标系。

### 阶段 5：计算右手腕在 pelvis 坐标系下的轨迹

这是核心数学。G1 的 `g1_wbc_pink` 控制器接受 pelvis 坐标系下的手腕目标。

对每帧：
1. 根据时间线在 pregrasp/grasp 之间插值（位置线性，旋转 SLERP）
2. 物体坐标系 → 世界坐标系：`T_hand_w = T_obj_w @ T_hand_obj`
3. 世界坐标系 → pelvis 坐标系：`p_hand_pelvis = R_base^T @ (p_hand_w - p_base_w)`

完整变换链：

```
物体坐标系 ──T_obj_w──→ 世界坐标系 ──T_base_w⁻¹──→ pelvis 坐标系
   ↑                                                    ↓
手的 pregrasp/grasp                              G1 控制器输入
相对位姿 (固定)                                  (每帧变化)
```

### 阶段 6：组装 23D action 向量

每帧 23 个维度：

```
索引      含义                    数据来源
─────────────────────────────────────────────────────
[0]       左手开合状态             固定 0.0（张开）
[1]       右手开合状态             按阶段切换
[2:5]     左手腕位置 (pelvis)      固定值
[5:9]     左手腕四元数 (pelvis)    固定值
[9:12]    右手腕位置 (pelvis)      ← 阶段 5 计算
[12:16]   右手腕四元数 (pelvis)    ← 阶段 5 计算
[16:19]   导航指令 (vx, vy, wz)   ← 阶段 4 计算
[19]      base 高度                固定 0.75
[20:23]   躯干 RPY                 固定 (0, 0, 0)
```

右手开合按阶段切换：

```
帧 0 ─── nav_end ── approach_start ──── grasp_idx ── close_end ── hold_end ── N-1
  │ 导航阶段 │ pregrasp │    approach     │  close   │   hold   │  (保持)  │
  │ 手张开   │ 手张开   │  手张开(接近中) │  手闭合  │  手闭合  │  手张开  │
  │ base移动 │ base停   │  base停        │  base停  │  base停  │  base停  │
```

### 输出文件

- `replay_actions.hdf5` — `[N, 23]` float32 action 序列
- `object_kinematic_traj.npz` — 物体轨迹（Isaac 回放时强制写入物体位姿，不走物理引擎）
- `bridge_debug.json` — 参数、阶段划分、sanity check

## 可视化

### 1. HOIFHLI 人体+物体 motion 可视化

HOIFHLI 上游仓库默认开启可视化（`VISUALIZE=True`），采样时自动渲染人体+物体交互视频（pyrender → PNG → MP4），无需额外配置。

运行 `scripts/01_hoi_sample.sh` 后，视频输出到：
```
repos/hoifhli_release/visualizer_results/<vis_wdir>/
```

### 2. Isaac 物体 motion 单独可视化

在 Isaac 回放时加 `--object-only` 参数，机器人站着不动，只回放物体轨迹。

**Headless 模式（无显示器 / 远程服务器）**— 保存视频文件：

```bash
ISAAC_PYTHON=/path/to/isaaclab_arena/bin/python \
  bash scripts/05_object_only_replay.sh /tmp/run1
```

视频输出到 `output/videos/`。

**GUI 模式（有显示器）**— 实时在线观看：

去掉 `--headless` 参数即可打开 Isaac Sim GUI 窗口，实时观看物体运动：

```bash
ISAAC_PYTHON=/path/to/isaaclab_arena/bin/python \
  python repos/IsaacLab-Arena/isaaclab_arena/examples/policy_runner_kinematic_object_replay.py \
  --device cuda:0 --enable_cameras \
  --object-only \
  --kin-traj-path /tmp/run1/object_kinematic_traj.npz \
  --kin-asset-name brown_box \
  --kin-apply-timing pre_step \
  galileo_g1_locomanip_pick_and_place \
  --object brown_box --embodiment g1_wbc_pink
```

也可以同时加 `--save-video` 在 GUI 观看的同时保存视频。

## 环境变量

脚本通过环境变量指定 Python 解释器：
- `HOI_PYTHON`：HOIFHLI 环境的 python（默认 `python3`）
- `ISAAC_PYTHON`：IsaacLab-Arena 环境的 python（默认 `python3`）

## 文档

- `docs/END_TO_END_FROM_HOI_BODEX_CN.md` — 端到端流程说明
- `docs/PROJECT_STATUS_CN.md` — 项目状态
- `docs/README_bridge.md` — bridge 脚本详解
- `docs/CUROBO_G1_MOTION_PLANNING_CN.md` — G1 适配 CuRobo 运动规划方案

## 已同步产物（artifacts）

当前仓库已同步中间和结果文件到 `artifacts/`：

- `artifacts/g1_bridge_run1/`
- `artifacts/g1_bridge_run2/`
- `artifacts/hoifhli/`
- `artifacts/videos/`
- `artifacts/MANIFEST.txt`

其中包含你提到的：

- `object_kinematic_traj.npz`
- `replay_actions.hdf5`（motion 回放文件）
- `human_object_results.pkl`
- 回放视频 `*.mp4`
