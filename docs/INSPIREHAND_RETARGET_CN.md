# InspireHand 集成 & SMPLX→InspireHand 重定向 操作手册

## 概述

本文档记录了将 G1 机器人的 Unitree 手替换为 InspireHand，以及使用 spider 进行 SMPLX→InspireHand 重定向的完整流程。

---

## 1. G1 + InspireHand URDF 生成

### 1.1 生成合并 URDF

```bash
cd Humanoid-gen-pack
python scripts/build_g1_inspirehand_urdf.py
```

输出：`configs/g1_inspirehand/g1_29dof_with_inspire_hand.urdf`（53 DOF）

### 1.2 验证

```bash
python -c "
import pinocchio as pin
model = pin.buildModelFromUrdf('configs/g1_inspirehand/g1_29dof_with_inspire_hand.urdf')
print(f'DOF: {model.nq}')  # 应为 53
"
```

---

## 2. IsaacLab-Arena 修改清单

以下文件已修改以支持 InspireHand（12 DOF/手，6 独立 + 6 mimic）：

| 文件 | 修改内容 |
|------|---------|
| `g1_env/g1_constants.py` | 24 个 InspireHand 关节限位（替换 Unitree 14 个） |
| `g1_env/g1_supplemental_info.py` | 关节名、关节组、限位映射（URDF 顺序） |
| `g1_env/config/loco_manip_g1_joints_order_43dof.yaml` | 53-DOF 关节顺序 |
| `g1_env/config/lab_g1_joints_order_43dof.yaml` | 53-DOF 关节顺序（交错左右） |
| `wbc_policy/policy/action_constants.py` | 新增 `RIGHT_FINGER_ANGLES` 索引 (23-35) |
| `g1_env/mdp/actions/g1_decoupled_wbc_pink_action.py` | 支持 35D action 提取 12 指关节 |
| `wbc_policy/g1_wbc_upperbody_ik/g1_wbc_upperbody_controller.py` | 12-DOF `get_hand_joint_pos()` + BODex→URDF 映射 |
| `wbc_policy/utils/g1.py` | 加载本地 G1+InspireHand URDF |

### 关节顺序映射

```
BODex 顺序: [thumb_yaw, thumb_pitch, index, middle, ring, pinky,
             thumb_inter, thumb_distal, index_inter, middle_inter, ring_inter, pinky_inter]

URDF 顺序:  [index_prox, index_inter, middle_prox, middle_inter, pinky_prox, pinky_inter,
             ring_prox, ring_inter, thumb_yaw, thumb_pitch, thumb_inter, thumb_distal]

BODex→URDF: [2, 8, 3, 9, 5, 11, 4, 10, 0, 1, 6, 7]
```

---

## 3. 使用 Spider 进行 SMPLX→InspireHand 重定向

Spider 是 Meta/FAIR 的物理重定向框架，已内置 InspireHand 支持（`robot_type=inspire`）。

### 3.1 环境准备

```bash
conda activate spider
cd /home/ubuntu/DATA2/workspace/xmh/spider
```

### 3.2 准备输入数据

Spider 需要 MANO 格式的手部关键点数据 `trajectory_keypoints.npz`：

```
qpos_wrist_right:  (T, 7)     # [x, y, z, qw, qx, qy, qz]
qpos_finger_right: (T, 5, 7)  # 5 指尖 × [x, y, z, qw, qx, qy, qz]
qpos_obj_right:    (T, 7)     # 物体位姿
qpos_wrist_left:   (T, 7)
qpos_finger_left:  (T, 5, 7)
qpos_obj_left:     (T, 7)
contact_right:     (T, 5)     # 可选：每指接触标记
contact_left:      (T, 5)
```

指尖顺序：thumb, index, middle, ring, pinky

### 3.3 从 SMPLX 生成输入数据

如果你的数据是 SMPLX 格式（如 hoifhli 输出），需要先转换：

```python
import smplx, torch, numpy as np
from scipy.spatial.transform import Rotation as R

# 加载 SMPLX 模型
model = smplx.create(model_path="data/smpl_all_models", model_type="smplx",
                     gender="male", flat_hand_mean=True, use_pca=False)

# 前向运动学得到关节位置
output = model(global_orient=..., transl=..., right_hand_pose=..., body_pose=...)
joints = output.joints[0].detach().numpy()  # (J, 3)

# SMPLX 右手指尖索引: thumb=40, index=41, middle=42, ring=43, pinky=44
# 右手腕: joint 21
wrist_pos = joints[21]
fingertips = joints[[40, 41, 42, 43, 44]]  # (5, 3)
```

### 3.4 运行 Spider 重定向管线

```bash
DATASET_DIR=example_datasets
DATASET=oakink          # 或你的数据集名
TASK=pick_spoon_bowl    # 任务名
DATA_ID=0

# Step 1: 物体 mesh 凸分解
uv run spider/preprocess/decompose_fast.py \
  --dataset-dir=$DATASET_DIR --dataset-name=$DATASET \
  --task=$TASK --data-id=$DATA_ID --embodiment-type=right

# Step 2: 接触检测
uv run spider/preprocess/detect_contact.py \
  --dataset-dir=$DATASET_DIR --dataset-name=$DATASET \
  --task=$TASK --data-id=$DATA_ID --embodiment-type=right

# Step 3: 生成场景 XML（加载 InspireHand 模型 + 物体）
uv run spider/preprocess/generate_xml.py \
  --dataset-dir=$DATASET_DIR --dataset-name=$DATASET \
  --robot-type=inspire --embodiment-type=right \
  --task=$TASK --data-id=$DATA_ID

# Step 4: 约束 IK（MANO 关键点 → InspireHand 关节角）
uv run spider/preprocess/ik.py \
  --dataset-dir=$DATASET_DIR --dataset-name=$DATASET \
  --robot-type=inspire --embodiment-type=right \
  --task=$TASK --data-id=$DATA_ID --open-hand

# Step 5: MPC 物理优化（可选，提升物理真实性）
uv run examples/run_mjwp.py \
  +override=$DATASET task=$TASK data_id=$DATA_ID \
  robot_type=inspire embodiment_type=right
```

### 3.5 输出格式

IK 输出 `trajectory_kinematic.npz`：
```
qpos: (T, nq)        # InspireHand 关节角（6 wrist DOF + 12 finger DOF + 7 object）
qvel: (T, nv)        # 关节速度
contact: (T, 5)      # 接触标记
contact_pos: (T,5,3) # 接触点位置
```

MPC 输出 `trajectory_mjwp.npz`：
```
qpos: (T, nq)        # 物理优化后的关节角
ctrl: (T, nu)        # 控制指令
```

### 3.6 提取 InspireHand 关节角用于 BODex/Isaac

```python
import numpy as np

data = np.load("trajectory_kinematic.npz")
qpos = data["qpos"]  # (T, nq)

# Spider InspireHand 关节顺序 (qpos[6:18]):
# [thumb_yaw, thumb_pitch, thumb_inter, thumb_distal,
#  index_prox, index_inter, middle_prox, middle_inter,
#  ring_prox, ring_inter, pinky_prox, pinky_inter]
finger_q = qpos[:, 6:18]   # (T, 12)
wrist_q = qpos[:, :6]      # (T, 6) [pos_xyz + rot_xyz_euler]

# 转 BODex 6 独立 DOF 顺序:
# [thumb_yaw, thumb_pitch, index, middle, ring, pinky]
SPIDER_TO_BODEX_6 = [0, 1, 4, 6, 8, 10]
bodex_6dof = finger_q[:, SPIDER_TO_BODEX_6]

# 转 URDF 顺序 (Isaac Sim WBC):
SPIDER_TO_URDF = [4, 5, 6, 7, 10, 11, 8, 9, 0, 1, 2, 3]
urdf_12dof = finger_q[:, SPIDER_TO_URDF]
```

---

## 4. BODex 抓取生成

### 4.1 基础抓取生成

```bash
cd Humanoid-gen-pack
conda activate bodex

python scripts/generate_bodex_inspirehand_grasps.py \
  --mesh-file path/to/object.obj \
  --output-pt grasps.pt
```

### 4.2 模块 B：Spider Seed 注入

使用 spider IK 输出作为 BODex 优化的初始种子，提升抓取质量：

```bash
python scripts/generate_bodex_inspirehand_grasps.py \
  --mesh-file path/to/object.obj \
  --seed-from-spider path/to/trajectory_ikrollout.npz \
  --seed-frame -1
```

内部转换流程：
- Spider qpos `[wrist_xyz(3), euler_xyz(3), finger_12dof, obj_7dof]`
- → BODex seed `[xyz(3), quat_wxyz(4), 6_independent_dof]` = 13D
- Spider 12 指关节 → BODex 6 独立 DOF 索引: `[0, 1, 4, 6, 8, 10]`

### 4.3 模块 C：人手距离排序

同时使用 seed 注入和人手距离排序，选择最接近人类抓取的结果：

```bash
python scripts/generate_bodex_inspirehand_grasps.py \
  --mesh-file path/to/object.obj \
  --seed-from-spider path/to/trajectory_ikrollout.npz \
  --rank-by-human \
  --lambda-rot 1.0 --lambda-finger 0.5
```

距离度量（论文 Eq.5）：
```
d = ||t_robot - t_human||₂ + λ_rot · arccos(|q_robot · q_human|) + λ_finger · ||q_finger_diff||₂
```

Shell 快捷方式（自动启用 seed + ranking）：
```bash
bash scripts/generate_bodex_inspirehand_grasps.sh object.obj spider_ik.npz output.pt
```

---

## 5. 端到端管线

```
SMPLX 人类动作
    │
    ├─→ SMPLX FK → MANO 关键点 (trajectory_keypoints.npz)
    │
    ├─→ Spider preprocess (decompose → detect_contact → generate_xml → ik)
    │       │
    │       ▼
    │   trajectory_kinematic.npz (InspireHand 12 关节角)
    │       │
    │       ├─→ [可选] Spider MPC 优化 → trajectory_mjwp.npz
    │       │
    │       ▼
    │   提取抓取帧 → BODex seed_config
    │       │
    │       ▼
    │   BODex 抓取优化 → 最优抓取 (wrist SE3 + 12 finger DOF)
    │       │
    │       ▼
    │   距离度量排序 (选最接近人手的抓取)
    │
    ▼
G1 全身 IK (CuRobo/PINK) → Isaac Sim 回放
```

---

## 6. 文件清单

| 文件 | 说明 |
|------|------|
| `scripts/build_g1_inspirehand_urdf.py` | 生成 G1+InspireHand 合并 URDF |
| `scripts/smplx_to_inspirehand.py` | 独立 IK 重定向模块（备用，非 spider 管线） |
| `scripts/generate_bodex_inspirehand_grasps.py` | BODex InspireHand 抓取生成 |
| `configs/g1_inspirehand/` | URDF + mesh 资源 |
| `configs/bodex_inspire_hand_grasp.yml` | BODex 抓取配置 |
| `isaac_replay/build_arm_follow_replay.py` | 35D action 构建（含 12 指关节） |
| `isaac_replay/g1_curobo_planner.py` | CuRobo IK + MotionGen 规划 |
