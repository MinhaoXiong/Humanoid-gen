# BODex InspireHand 抓取生成流程详解

## 1. 概述

BODex 是一个基于 CuRobo 的灵巧手抓取位姿生成框架。给定一个物体 mesh，BODex 优化出多个候选抓取方案，每个包含：
- **wrist pose**：手腕在物体坐标系下的 6D 位姿
- **finger joint angles**：InspireHand 12 个关节角（6 独立 + 6 mimic）

一次运行可生成多个候选（默认 top-k=16），按抓取质量排序。

## 2. InspireHand 关节结构

InspireHand 右手共 12 个关节，分为 6 个独立 DOF 和 6 个 mimic（从动）关节：

| 独立关节 | mimic 关节 | 倍率 |
|----------|-----------|------|
| R_thumb_proximal_yaw_joint | — | — |
| R_thumb_proximal_pitch_joint | R_thumb_intermediate_joint | 1.6x |
| | R_thumb_distal_joint | 2.4x |
| R_index_proximal_joint | R_index_intermediate_joint | 1.0x |
| R_middle_proximal_joint | R_middle_intermediate_joint | 1.0x |
| R_ring_proximal_joint | R_ring_intermediate_joint | 1.0x |
| R_pinky_proximal_joint | R_pinky_intermediate_joint | 1.0x |

BODex 只优化 6 个独立 DOF，mimic 关节通过 `expand_mimic_joints()` 按倍率自动计算。

## 3. 生成流程

```
物体 mesh (.obj)
    → CuRobo WorldConfig（碰撞世界）
    → GraspSolverConfig（加载 inspire_hand.yml）
    → GraspSolver.solve_batch_env()（多 seed 并行优化）
    → 按 grasp_error 排序取 top-k
    → expand_mimic_joints()（6 DOF → 12 关节角）
    → 保存 .pt
```

核心优化目标：
- 5 个指尖（thumb/index/middle/ring/pinky）接触物体表面
- 最小化接触距离误差（dist_error）
- 最小化力闭合误差（grasp_error）
- 满足关节限位约束

## 4. 运行命令

```bash
cd Humanoid-gen-pack

# 生成 cracker_box 的 16 个候选抓取
python scripts/generate_bodex_inspirehand_grasps.py \
  --mesh-file /path/to/cracker_box.obj \
  --top-k 16

# 或用 shell wrapper
bash scripts/generate_bodex_inspirehand_grasps.sh /path/to/cracker_box.obj
```

输出：`artifacts/bodex_inspire_grasps/<object>_top16_<timestamp>.pt`

## 5. .pt 文件格式

```python
{
    "object_name": "cracker_box",
    "mesh_path": "/path/to/cracker_box.obj",
    "wrist_pose_7d": tensor [K, 7],       # [x, y, z, w, qx, qy, qz] 物体坐标系
    "joint_angles_12": tensor [K, 12],     # 12 个关节角（含 mimic）
    "independent_q_6": tensor [K, 6],      # 6 个独立关节角
    "grasp_error": tensor [K],             # 抓取质量（越小越好）
    "dist_error": tensor [K],              # 接触距离误差
    "joint_names_12": list[str],           # 12 个关节名顺序
    "independent_joint_names": list[str],  # 6 个独立关节名
}
```

K = top-k 候选数（默认 16）。index 0 是最优抓取。

## 6. 与管道的衔接

```
阶段1: BODex 生成 .pt（离线）
  ↓
阶段2: 管道读取 .pt → wrist pose 转世界坐标系 → IK 检查 → MotionGen 轨迹
  ↓
阶段3: replay 构建 → Isaac Sim 回放（手臂轨迹 + 手指关节角）
```

在管道中使用 BODex 抓取：

```bash
python scripts/run_walk_to_grasp_todo.py \
  --bodex-grasp-pt artifacts/bodex_inspire_grasps/cracker_box_top16.pt \
  --bodex-grasp-index 0 \
  --out-dir artifacts/todo_bodex_test \
  --scene kitchen_pick_and_place \
  --object cracker_box \
  --planner auto
```

`--bodex-grasp-index` 选择第几个候选（0=最优）。

## 7. 查看 .pt 内容

```python
import torch
data = torch.load("artifacts/bodex_inspire_grasps/cracker_box_top16.pt", map_location="cpu")
print(f"候选数: {data['wrist_pose_7d'].shape[0]}")
print(f"最优 grasp_error: {data['grasp_error'][0]:.6f}")
print(f"最优 wrist pose: {data['wrist_pose_7d'][0].tolist()}")
print(f"最优 finger angles: {data['joint_angles_12'][0].tolist()}")
```

## 8. 关键文件

| 文件 | 作用 |
|------|------|
| `scripts/generate_bodex_inspirehand_grasps.py` | 抓取生成主脚本 |
| `configs/bodex_inspire_hand_grasp.yml` | GraspSolver 配置（接触点、力约束） |
| `configs/bodex_inspire_hand_assets/` | InspireHand URDF + mesh + CuRobo 配置 |
| `scripts/setup_bodex_inspirehand.sh` | 部署资产到 BODex 仓库 |
