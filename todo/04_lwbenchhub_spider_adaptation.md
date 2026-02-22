# LW-BenchHub + Spider 场景适配方案

## 1. 目标

将 Humanoid-gen-pack 的 walk-to-grasp 管线从仅支持 2 个硬编码场景，扩展为支持 LW-BenchHub 的 100 个厨房场景，并通过 Spider 实现 SMPL 人体动作到 Unitree G1 的 retargeting。

## 2. 整体数据流

```
人体视频/MoCap (SMPL格式)
    │
    ▼
Spider retarget (SMPL → G1 29DOF 关节角 + 物体轨迹)
    │  spider/examples/run_mjwp.py --robot_type=unitree_g1
    │  输出: trajectory_mjwp.npz (qpos, qvel, ctrl)
    │
    ▼
bridge/smpl_to_g1_spider.py (Spider NPZ → object_kinematic_traj.npz)
    │  提取物体位姿序列，转换坐标系 (MuJoCo Z-up → Isaac Z-up)
    │  输出: object_kinematic_traj.npz (object_pos_w, object_quat_wxyz)
    │
    ▼
bridge/lwbench_scene_adapter.py (LW-BenchHub 场景 → SceneConfig + 碰撞体)
    │  从 Lightwheel SDK 加载厨房 USD
    │  提取 fixture AABB → CuRobo collision cuboids
    │  生成 SceneConfig (桌面高度、工作空间边界、机器人位置)
    │  输出: scene_config.json
    │
    ▼
run_walk_to_grasp_todo.py (现有 8 步管线，不修改)
    │  Step1: 物体轨迹 (来自 Spider 或 synthetic)
    │  Step2: 加载物体初始位姿
    │  Step3: CuRobo 规划 (使用动态碰撞体)
    │  Step4: 构建 replay HDF5
    │  Step5: 读取 replay 元数据
    │  Step6: Isaac Sim 仿真 (使用 LW-BenchHub 场景)
    │  Step7: 验证产物
    │  Step8: 报告
    │
    ▼
scripts/15_lwbench_walk_to_grasp.sh (新脚本，不修改 13_todo_walk_to_grasp_gui.sh)
```

## 3. 新增文件清单

| 文件 | 用途 |
|------|------|
| `bridge/lwbench_scene_adapter.py` | LW-BenchHub 场景 → SceneConfig + 碰撞体提取 |
| `bridge/smpl_to_g1_spider.py` | Spider NPZ → object_kinematic_traj.npz 转换 |
| `scripts/15_lwbench_walk_to_grasp.sh` | LW-BenchHub 场景的 walk-to-grasp 入口脚本 |
| `todo/04_lwbenchhub_spider_adaptation.md` | 本文档 |

## 4. 修改文件清单

| 文件 | 修改内容 |
|------|---------|
| `bridge/scene_config.py` | 新增 `register_lwbench_scene()` 函数，支持动态注册场景 |
| `isaac_replay/g1_curobo_planner.py` | 新增动态碰撞体注册接口 |

## 5. 关键设计决策

### 5.1 为什么选 LW-BenchHub 而不是 GenieSim

- LW-BenchHub 和 Humanoid-gen-pack 共享 `isaaclab_arena` 框架，原生兼容
- LW-BenchHub 有 100 个厨房场景 (10 layouts × 10 styles)，GenieSim 仅 1 个 G1 场景
- LW-BenchHub 的 G1 embodiment 已配好 WBC 控制器
- LW-BenchHub 的 fixture AABB 可自动提取，不需要手动测量碰撞体

### 5.2 坐标系约定

- Spider (MuJoCo): Z-up，单位 m
- Isaac Sim / IsaacLab: Z-up，单位 m
- LW-BenchHub 场景: Z-up，场景原点由 USD 定义
- 物体轨迹坐标: 世界坐标系 (Z-up)

Spider 和 Isaac 都是 Z-up，不需要坐标系旋转，只需要平移对齐到场景桌面。

### 5.3 碰撞体提取策略

从 LW-BenchHub 的 `KitchenArena` 提取 fixture AABB：
1. 加载 USD stage
2. 遍历 fixture prims，提取 bounding box
3. 筛选 "counter" 类型的 fixture 作为桌面碰撞体
4. 转换为 CuRobo cuboid 格式: `{"name": ..., "pose": [x,y,z, qw,qx,qy,qz], "dims": [dx,dy,dz]}`

### 5.4 机器人配置

保持 Humanoid-gen-pack 的 G1 + InspireHand 配置不变：
- URDF: `g1_29dof_with_inspire_hand.urdf`
- CuRobo: `g1_right_arm.yml` (7 DOF 右臂)
- Embodiment: `g1_wbc_pink`
- 手: InspireHand 6 DOF

LW-BenchHub 的 Dex3 夹爪配置不使用。

## 6. Spider Retargeting 流程

### 6.1 前置条件

```bash
# Spider 环境
conda activate spider
cd /home/ubuntu/DATA2/workspace/xmh/spider
```

### 6.2 完整 Pipeline

```bash
# 1. 处理 SMPL 数据集 → 标准 NPZ
uv run spider/process_datasets/gigahand.py \
  --task=TASK --embodiment-type=humanoid --data-id=0

# 2. 网格分解
uv run spider/preprocess/decompose_fast.py \
  --task=TASK --dataset-name=NAME --data-id=0 --embodiment-type=humanoid

# 3. 接触检测
uv run spider/preprocess/detect_contact.py \
  --task=TASK --dataset-name=NAME --data-id=0 --embodiment-type=humanoid

# 4. 生成场景 XML
uv run spider/preprocess/generate_xml.py \
  --task=TASK --dataset-name=NAME --data-id=0 \
  --embodiment-type=humanoid --robot-type=unitree_g1

# 5. IK 初始化
uv run spider/preprocess/ik.py \
  --task=TASK --dataset-name=NAME --data-id=0 \
  --embodiment-type=humanoid --robot-type=unitree_g1 --open-hand

# 6. MPC 物理优化
uv run examples/run_mjwp.py +override=humanoid \
  task=TASK data_id=0 robot_type=unitree_g1 embodiment_type=humanoid

# 7. 转换为 Humanoid-gen-pack 格式
python bridge/smpl_to_g1_spider.py \
  --spider-npz /path/to/trajectory_mjwp.npz \
  --output-npz /path/to/object_kinematic_traj.npz \
  --scene lwbench_kitchen_0_0
```

### 6.3 Spider 输出格式

```python
# trajectory_mjwp.npz 内容
{
    "qpos": np.ndarray,      # [T, nq] 关节位置 (含浮动基座 7 + 机器人 DOF + 物体 DOF)
    "qvel": np.ndarray,      # [T, nv] 关节速度
    "ctrl": np.ndarray,      # [T, nu] 控制信号
    "contact": np.ndarray,   # [T, ...] 接触状态
}
```

### 6.4 转换到 Humanoid-gen-pack 格式

```python
# object_kinematic_traj.npz 内容 (Humanoid-gen-pack 需要的)
{
    "object_pos_w": np.ndarray,       # [T, 3] 物体世界坐标位置
    "object_quat_wxyz": np.ndarray,   # [T, 4] 物体世界坐标四元数 (w,x,y,z)
    "object_name": str,               # 物体名称
}
```

## 7. LW-BenchHub 场景适配流程

### 7.1 场景命名约定

```
lwbench_kitchen_{layout_id}_{style_id}
```

例如: `lwbench_kitchen_0_0`, `lwbench_kitchen_3_5`

### 7.2 场景参数自动提取

`bridge/lwbench_scene_adapter.py` 负责：

1. 调用 `lightwheel_sdk.loader.floorplan_loader.acquire_usd()` 获取 USD 路径
2. 用 OpenUSD API 提取 fixture bounding boxes
3. 识别 counter/table 类型 fixture 作为工作台面
4. 计算桌面高度 (counter top Z)
5. 计算机器人初始位置 (counter 前方 0.6-0.8m)
6. 生成 SceneConfig 和 CuRobo collision cuboids
7. 输出 JSON 配置文件

### 7.3 碰撞体格式

```json
{
  "scene_name": "lwbench_kitchen_0_0",
  "scene_config": {
    "arena_scene_name": "lwbench_kitchen_0_0",
    "arena_background": "lwbench_kitchen_0_0",
    "table_z": 0.85,
    "object_align_pos": [0.5, 0.0, 0.85],
    "base_height": 0.75,
    "default_base_pos_w": [-0.3, 0.0, 0.0],
    "replay_base_height": 0.78
  },
  "collision_cuboids": [
    {"name": "counter_0", "pose": [0.5, 0.0, 0.425, 1, 0, 0, 0], "dims": [0.8, 1.2, 0.85]}
  ],
  "obstacles_2d": [
    [[0.1, -0.6], [0.9, 0.6]]
  ]
}
```
