# 功能2: 替换Dex3-1为InspireHand + CEDex-Grasp抓取生成

## 1. 当前状态分析

### 1.1 当前灵巧手: Unitree Dex3-1
- **URDF**: `Humanoid-Teleop/assets/unitree_hand/unitree_dex3_right.urdf`
- **配置**: `Humanoid-Teleop/assets/unitree_hand/unitree_dex3.yml`
- **控制**: `Humanoid-Teleop/teleop/robot_control/unitree_hand.py`

### 1.2 当前抓取生成: BODex
- **路径**: `BODex/`
- **方法**: CuRobo grasp solver，基于物体几何+重心生成抓取
- **支持的手**: ShadowHand, Allegro, LeapHand
- **不支持**: InspireHand

### 1.3 目标灵巧手: InspireHand
- **URDF**: `ManipTrans/maniptrans_envs/assets/inspire_hand/inspire_hand_right.urdf`
- **Mesh**: `ManipTrans/maniptrans_envs/assets/inspire_hand/meshes/`
- **特点**: 5指，16个语义分区（掌 + 5指×3段）

### 1.4 目标抓取生成: CEDex-Grasp
- **路径**: `CEDex-Grasp/`
- **方法**: 接触图引导的Adam优化
- **已支持InspireHand**: 是（见 `INSPIREHAND_ADAPTATION.md`）
- **状态**: 代码已适配，尚未端到端验证

## 2. CEDex-Grasp的InspireHand抓取生成流程

### 2.1 已完成的适配（INSPIREHAND_ADAPTATION.md）

CEDex-Grasp已对InspireHand做了以下适配：

1. **资源配置** (`data/urdf/urdf_assets_meta.json`)
   - 注册了inspirehand的URDF和mesh路径

2. **手模型** (`utils_model/HandModel.py`)
   - 16-part语义映射（掌+5指×3段）
   - SPF关键点（dis_key_point）和SRF关键点（keypoints）

3. **优化器** (`utils_model/CMapAdam.py`)
   - `n_robot_part=16`, `n_human_part=16`
   - 初始旋转策略：绕Z轴随机旋转

4. **控制器** (`utils/controller.py`)
   - yaw关节方向: `[0,1,0]`，其他: `[1,0,0]`
   - 开手→lower limit，收手→upper limit

5. **入口** (`generate_data.py`, `vis_generated_grasp.py`)
   - `--robot_name inspirehand` 已加入choices

### 2.2 CEDex-Grasp抓取生成命令

对cracker_box物体生成InspireHand抓取：
```bash
cd CEDex-Grasp
python generate_data.py \
    --robot_name inspirehand \
    --object_name 003_cracker_box \
    --n_particles 64 \
    --max_iter 300
```

输出: `logs/dataset_generation_*/generated_grasps/003_cracker_box.npy`
格式: top-16抓取的关节角配置 `(16, n_joints)`

### 2.3 CEDex-Grasp生成流程详解

```
Stage 1: 接触图预测（PointNet++）
  输入: 物体点云 + 人手接触图
  输出: InspireHand的接触图预测 contact_map_goal (N, 7)
         ↓
Stage 2: Adam优化（CMapAdam）
  输入: contact_map_goal + contact_part标签
  优化: 手的全局位姿(6D) + 关节角(n_joints)
  损失: contact_loss + 1000*(ERF + SPF + SRF)
  迭代: 300次，最后50次仅优化穿透
         ↓
Stage 3: 控制器闭合（controller.py）
  输入: 优化后的关节角
  操作: 按InspireHand开合策略微调关节
         ↓
Stage 4: Isaac Gym验证（可选）
  输入: 最终关节角 + 物体
  测试: 6方向力测试，位移<2cm为成功
```

## 3. 机器人换手方案：Dex3-1 → InspireHand

### 3.1 URDF替换

**当前G1使用的手**: `g1_29dof_with_hand_rev_1_0.usd`（含Dex3-1）

**需要做的**:
1. 获取或制作G1+InspireHand的组合URDF/USD
2. 方案A: 修改G1 URDF，将Dex3-1的wrist link替换为InspireHand
3. 方案B: 使用已有的 `pickplace_unitree_g1_inspire_hand_env_cfg.py` 中的 `G1_INSPIRE_FTP_CFG`

**推荐方案B**，因为IsaacLab已有G1+InspireHand的完整配置：
- 文件: `isaaclab_tasks/manager_based/manipulation/pick_place/pickplace_unitree_g1_inspire_hand_env_cfg.py`
- 机器人配置: `G1_INSPIRE_FTP_CFG`
- 24个手关节（每手12个: 5指×2+拇指额外2）
- PINK IK控制器已配置双臂EEF

### 3.2 Embodiment配置修改

**修改 `g1.py` 中的 `G1SceneCfg`**:
```python
# 替换USD路径为G1+InspireHand版本
robot: ArticulationCfg = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=G1_INSPIRE_FTP_CFG.usd_path,  # G1+InspireHand USD
        ...
    ),
    actuators={
        ...  # 保留legs/feet/waist/arms
        "hands": IdealPDActuatorCfg(
            joint_names_expr=[".*_hand_.*"],  # InspireHand关节名
            effort_limit=5.0,
            velocity_limit=10.0,
            stiffness=4.0,
            damping=0.5,
        ),
    },
)
```

### 3.3 抓取位姿从CEDex-Grasp到IsaacLab的转换

CEDex-Grasp输出的是**浮动手**的关节角+全局位姿，需要转换为**机械臂末端**的目标位姿+手指关节角。

**转换流程**:
```python
# CEDex输出
hand_global_pose = grasp_result['hand_pose']  # (4,4) 手掌全局位姿
hand_joint_angles = grasp_result['joint_angles']  # (n_joints,)

# 转换为IsaacLab动作
# 1. hand_global_pose → EEF目标位姿（考虑wrist到hand_base的offset）
eef_target = hand_global_pose @ inv(wrist_to_hand_offset)

# 2. hand_joint_angles → 手指关节目标
finger_targets = hand_joint_angles  # 直接使用
```

## 4. 为物体obj生成InspireHand抓取的完整流程

### 4.1 准备物体mesh

```bash
# 确保物体mesh在CEDex-Grasp可访问的路径
# YCB物体(如cracker_box)已在ContactDB/YCB数据集中
ls CEDex-Grasp/data/object_meshes/003_cracker_box/
```

### 4.2 运行CEDex-Grasp生成

```bash
cd CEDex-Grasp

# Step 1: 生成接触图（如果没有预训练模型则跳过Stage1，直接用GT接触图）
python generate_data.py \
    --robot_name inspirehand \
    --object_name 003_cracker_box \
    --n_particles 64 \
    --max_iter 300 \
    --gpu 0
```

### 4.3 Isaac Gym验证（可选）

```bash
cd CEDex-Grasp
python validation/isaac_main.py \
    --robot_name inspirehand \
    --object_name 003_cracker_box \
    --batch_size 16 \
    --gpu 0
```

### 4.4 将抓取结果导入IsaacLab场景

```python
# 加载CEDex生成的抓取
grasps = np.load("CEDex-Grasp/logs/.../003_cracker_box.npy")
best_grasp = grasps[0]  # 能量最低的抓取

# 提取手掌位姿和关节角
hand_pose = best_grasp[:7]   # pos(3) + quat(4)
joint_q = best_grasp[7:]     # 关节角

# 在IsaacLab中设置预抓取位姿
pre_grasp_eef_pose = compute_eef_from_hand_pose(hand_pose)
```

## 5. 需要修改/新建的文件清单

| 操作 | 文件路径 | 说明 |
|------|----------|------|
| 修改 | `IsaacLab-Arena/isaaclab_arena/embodiments/g1/g1.py` | USD换为G1+InspireHand |
| 新建 | `scripts/generate_inspirehand_grasps.py` | 调用CEDex为obj生成抓取 |
| 新建 | `scripts/convert_cedex_to_isaaclab.py` | CEDex抓取→IsaacLab位姿 |
| 修改 | `CEDex-Grasp/generate_data.py` | 确认inspirehand路径正确 |
| 验证 | `CEDex-Grasp/validation/isaac_main.py` | 验证InspireHand抓取质量 |
