# 给Codex的分步代码实现指令（CuRobo版）

## 前置条件

- 工作目录: `/home/ubuntu/DATA2/workspace/xmh`
- CuRobo源码: `BODex/src/curobo/` (已有，可直接import)
- MoMaGen CuRobo wrapper参考: `MoMaGen/BEHAVIOR-1K/OmniGibson/omnigibson/action_primitives/curobo.py`
- InspireHand URDF: `ManipTrans/maniptrans_envs/assets/inspire_hand/inspire_hand_right.urdf`
- G1+InspireHand配置参考: `IsaacLab-Arena/submodules/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/pick_place/pickplace_unitree_g1_inspire_hand_env_cfg.py`

---

## Step 1: G1 CuRobo Robot Config

**新建文件**: `IsaacLab-Arena/isaaclab_arena/planning/g1_inspire_curobo.yaml`

为G1+InspireHand编写CuRobo robot config，定义两种embodiment模式：
- `base_mode`: 只规划底盘(x,y,θ)，锁定手臂和腿
- `arm_mode`: 只规划手臂，锁定底盘和腿

**参考**: `MoMaGen/BEHAVIOR-1K/OmniGibson/omnigibson/action_primitives/curobo.py:77-264` 中为R1机器人创建多embodiment MotionGen的方式

**关键字段**:
- `urdf_path`: G1+InspireHand URDF
- `base_link`: `pelvis`
- `ee_link`: `right_wrist_yaw_link`
- `collision_spheres`: 从G1 URDF自动生成或手动标注
- `lock_joints`: 按模式分组

---

## Step 2: CuRobo规划Wrapper

**新建文件**: `IsaacLab-Arena/isaaclab_arena/planning/g1_curobo_planner.py`

参考 `MoMaGen/.../curobo.py` 的 `CuRoboMotionGenerator` 类，实现三个核心方法：

1. `sample_base_target(obj_pos)` — 采样base位置 + IK可达验证 + 碰撞检查
2. `plan_base_path(start, target)` — BASE模式无碰撞底盘路径
3. `plan_arm_trajectory(joint_pos, eef_target)` — ARM_NO_TORSO模式手臂轨迹

详见 `01_feature1_walk_to_grasp.md` 第4.2节的伪代码。

---

## Step 3: G1+InspireHand Embodiment

**修改文件**: `IsaacLab-Arena/isaaclab_arena/embodiments/g1/g1.py`

1. 新增 `G1InspireHandSceneCfg`，USD换为G1+InspireHand版本
2. 参考 `pickplace_unitree_g1_inspire_hand_env_cfg.py` 的 `G1_INSPIRE_FTP_CFG`
3. `init_state.pos=(0.8, -3.0, 0.78)` — 远离桌子
4. 新增 `G1InspireHandWBCPinkEmbodiment` 类

---

## Step 4: 三阶段策略（CuRobo规划 + WBC执行）

**新建文件**: `IsaacLab-Arena/isaaclab_arena/policy/walk_to_grasp_policy.py`

核心逻辑：
1. 初始化时调用CuRobo planner离线规划base路径和手臂轨迹
2. 导航阶段：CuRobo路径点 → WBC navigation_subgoals → P控制器跟踪
3. 手臂阶段：CuRobo手臂轨迹 → PINK IK action
4. 操作阶段：MoMaGen位姿相对变换replay

---

## Step 5: CEDex-Grasp生成InspireHand抓取

**新建文件**: `scripts/generate_inspirehand_grasps.sh`

```bash
#!/bin/bash
cd /home/ubuntu/DATA2/workspace/xmh/CEDex-Grasp
python generate_data.py \
    --robot_name inspirehand \
    --object_name 003_cracker_box \
    --n_particles 64 --max_iter 300 --gpu 0
```

---

## Step 6: 抓取格式转换 + 场景集成

**新建文件**: `scripts/convert_cedex_to_isaaclab.py`

CEDex输出浮动手位姿+关节角 → 转为IsaacLab EEF目标+手指关节目标。

---

## Step 7: 运行入口

**新建文件**: `IsaacLab-Arena/isaaclab_arena/scripts/run_walk_to_grasp.py`

一键运行：加载场景 → CuRobo离线规划 → WBC在线执行三阶段。

---

## Step 8: 验证

1. CEDex抓取质量：Isaac Gym验证成功率>50%
2. CuRobo路径：可视化base路径确认避障
3. 导航：机器人稳定走到桌子附近
4. 手臂：无碰撞过渡到预抓取位姿
5. 抓取：手指闭合抓住cracker_box

---

## 文件总表

| Step | 操作 | 文件 |
|------|------|------|
| 1 | 新建 | `isaaclab_arena/planning/g1_inspire_curobo.yaml` |
| 2 | 新建 | `isaaclab_arena/planning/g1_curobo_planner.py` |
| 3 | 修改 | `isaaclab_arena/embodiments/g1/g1.py` |
| 4 | 新建 | `isaaclab_arena/policy/walk_to_grasp_policy.py` |
| 5 | 新建 | `scripts/generate_inspirehand_grasps.sh` |
| 6 | 新建 | `scripts/convert_cedex_to_isaaclab.py` |
| 7 | 新建 | `isaaclab_arena/scripts/run_walk_to_grasp.py` |

## 依赖关系

```
Step 1 (CuRobo config) → Step 2 (Planner) ─┐
Step 3 (Embodiment)  ───────────────────────┤
Step 5 (CEDex生成) → Step 6 (格式转换) ────┼→ Step 7 (入口) → Step 8 (验证)
Step 4 (策略类)  ───────────────────────────┘
```
