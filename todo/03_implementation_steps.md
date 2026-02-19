# 给Codex的分步代码实现指令

## 前置条件

- 工作目录: `/home/ubuntu/DATA2/workspace/xmh`
- 核心项目: IsaacLab-Arena, CEDex-Grasp, MoMaGen, BODex
- InspireHand URDF: `ManipTrans/maniptrans_envs/assets/inspire_hand/inspire_hand_right.urdf`
- 已有G1+InspireHand配置参考: `IsaacLab-Arena/submodules/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/pick_place/pickplace_unitree_g1_inspire_hand_env_cfg.py`

---

## Step 1: 创建G1+InspireHand的WBC Embodiment

**目标**: 在IsaacLab-Arena中新建一个使用InspireHand的G1 embodiment配置

**修改文件**: `IsaacLab-Arena/isaaclab_arena/embodiments/g1/g1.py`

**操作**:
1. 在 `G1SceneCfg` 中新增 `G1InspireHandSceneCfg`，将 `usd_path` 替换为G1+InspireHand的USD
2. 参考 `pickplace_unitree_g1_inspire_hand_env_cfg.py` 中的 `G1_INSPIRE_FTP_CFG` 获取正确的USD路径和关节名
3. 手部actuator的 `joint_names_expr` 改为InspireHand的关节名模式（参考URDF中的joint名称）
4. 新增 `G1InspireHandWBCPinkEmbodiment` 类，继承 `G1EmbodimentBase`，使用新的SceneCfg

**关键参考**:
- InspireHand关节名格式: `R_thumb_proximal_yaw_joint`, `R_index_proximal_joint` 等（见URDF）
- 已有配置中手关节数: 每手12个DOF

---

## Step 2: 修改机器人初始位置（远离桌子）

**目标**: 让G1 spawn在离桌子较远的位置

**修改文件**: `IsaacLab-Arena/isaaclab_arena/embodiments/g1/g1.py`

**操作**:
1. 在 `G1InspireHandSceneCfg`（Step1新建）中设置 `init_state.pos=(0.8, -3.0, 0.78)`
2. 保持朝向不变 `rot=(0.0, 0.0, 0.0, 1.0)`
3. 这样机器人距桌子约1.6m，需要先走过去

---

## Step 3: 实现三阶段轨迹策略（Walk-to-Grasp）

**目标**: 新建策略类，实现 导航→预抓取→操作回放 三阶段

**新建文件**: `IsaacLab-Arena/isaaclab_arena/policy/walk_to_grasp_policy.py`

**核心类结构**:
```python
class WalkToGraspPolicy(PolicyBase):
    """三阶段: Navigation → PreGrasp → Replay"""

    def __init__(self, nav_target, pre_grasp_pose, replay_traj_file, ...):
        self.phase = "navigation"
        self.nav_target = nav_target          # (x, y) 导航目标
        self.pre_grasp_pose = pre_grasp_pose  # (4,4) 预抓取EEF位姿
        self.replay_traj = torch.load(replay_traj_file)
        self.step_idx = 0

    def get_action(self, env, obs):
        if self.phase == "navigation":
            return self._nav_action(env, obs)
        elif self.phase == "pre_grasp":
            return self._pre_grasp_action(env, obs)
        else:
            return self._replay_action(env, obs)
```

**导航阶段 `_nav_action` 实现要点**:
- 计算机器人当前base位置到nav_target的方向向量
- 生成navigate_cmd: `[vx, vy, omega_z]`，使用P控制器
- 手臂保持默认姿态（navigate_cmd非零时WBC自动处理下肢）
- 停止条件: `dist < 0.15m` → 切换到pre_grasp阶段

**预抓取阶段 `_pre_grasp_action` 实现要点**:
- navigate_cmd设为零（停止行走）
- 用线性插值从当前EEF位姿过渡到pre_grasp_pose
- 插值步数: 约30-50步
- 完成条件: EEF位姿误差 < 阈值 → 切换到replay阶段

**回放阶段 `_replay_action` 实现要点**:
- 参考 `replay_automoma_trajectory_policy.py` 的replay逻辑
- 使用MoMaGen的位姿相对变换: `T_new = T_cur_obj @ inv(T_src_obj) @ T_src_eef[t]`
- 每步从replay_traj取出目标EEF位姿，转为action
- navigate_cmd保持零

---

## Step 4: CEDex-Grasp为Cracker Box生成InspireHand抓取

**目标**: 用CEDex-Grasp为003_cracker_box生成InspireHand的抓取位姿

**新建文件**: `scripts/generate_inspirehand_grasps.sh`

**内容**:
```bash
#!/bin/bash
cd /home/ubuntu/DATA2/workspace/xmh/CEDex-Grasp

# 为cracker_box生成InspireHand抓取
python generate_data.py \
    --robot_name inspirehand \
    --object_name 003_cracker_box \
    --n_particles 64 \
    --max_iter 300 \
    --gpu 0
```

**前置检查**:
1. 确认 `CEDex-Grasp/data/urdf/urdf_assets_meta.json` 中inspirehand路径正确
2. 确认 `ManipTrans/maniptrans_envs/assets/inspire_hand/inspire_hand_right.urdf` 存在
3. 确认 `pytorch_kinematics` 已安装（CEDex依赖）

---

## Step 5: 抓取结果转换脚本（CEDex → IsaacLab）

**目标**: 将CEDex输出的浮动手抓取转为IsaacLab可用的EEF位姿+手指关节角

**新建文件**: `scripts/convert_cedex_to_isaaclab.py`

**核心逻辑**:
```python
import numpy as np
import torch
from pytorch_kinematics import chain

def convert_grasp(grasp_file, urdf_path):
    """
    CEDex输出: hand_global_pose(4x4) + joint_angles(n_joints)
    IsaacLab需要: eef_target_pose(4x4) + finger_joint_targets(n_joints)
    """
    data = np.load(grasp_file, allow_pickle=True)
    # 取能量最低的抓取
    best = data[0]
    hand_pose = best[:16].reshape(4,4)  # 手掌全局位姿
    joint_q = best[16:]                  # 关节角

    # 计算wrist→hand_base offset (从URDF获取)
    # eef_pose = hand_pose @ inv(hand_base_to_wrist)
    return {"eef_pose": hand_pose, "joint_q": joint_q}
```

---

## Step 6: 集成到Cracker Box场景

**目标**: 将三阶段策略和InspireHand抓取集成到kitchen_pick_and_place场景

**修改文件**: `IsaacLab-Arena/isaaclab_arena/examples/example_environments/kitchen_pick_and_place_environment.py`

**操作**:
1. 新增 `--embodiment g1_inspire_wbc_pink` 选项
2. 加载CEDex生成的抓取结果作为pre_grasp_pose
3. 使用 `WalkToGraspPolicy` 替代默认的静态操作策略

---

## Step 7: 运行入口脚本

**新建文件**: `IsaacLab-Arena/isaaclab_arena/scripts/run_walk_to_grasp.py`

**功能**: 一键运行cracker_box场景的walk-to-grasp流程

```python
"""运行入口: G1+InspireHand walk-to-grasp cracker_box"""
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--grasp_file", required=True,
                        help="CEDex生成的抓取文件路径")
    parser.add_argument("--replay_traj", default=None,
                        help="操作回放轨迹文件(.pt)")
    parser.add_argument("--num_envs", type=int, default=1)
    args = parser.parse_args()

    # 1. 加载场景(G1+InspireHand + kitchen + cracker_box)
    # 2. 加载CEDex抓取 → pre_grasp_pose
    # 3. 创建WalkToGraspPolicy
    # 4. 运行仿真循环
```

---

## Step 8: 验证与调试

**验证清单**:

1. **CEDex抓取质量**: 运行Isaac Gym验证，成功率应>50%
```bash
cd CEDex-Grasp
python validation/isaac_main.py --robot_name inspirehand --object_name 003_cracker_box
```

2. **导航阶段**: 机器人能稳定走到桌子附近并停下
3. **预抓取过渡**: 手臂能平滑移动到预抓取位姿
4. **操作回放**: 手指能按CEDex生成的关节角闭合抓住物体

---

## 文件修改/新建总表

| Step | 操作 | 文件 |
|------|------|------|
| 1 | 修改 | `IsaacLab-Arena/isaaclab_arena/embodiments/g1/g1.py` |
| 2 | 同上 | 同上（init pos） |
| 3 | 新建 | `IsaacLab-Arena/isaaclab_arena/policy/walk_to_grasp_policy.py` |
| 4 | 新建 | `scripts/generate_inspirehand_grasps.sh` |
| 5 | 新建 | `scripts/convert_cedex_to_isaaclab.py` |
| 6 | 修改 | `IsaacLab-Arena/.../kitchen_pick_and_place_environment.py` |
| 7 | 新建 | `IsaacLab-Arena/isaaclab_arena/scripts/run_walk_to_grasp.py` |

## 依赖关系

```
Step 1 (Embodiment) ──┐
Step 2 (Init pos)  ───┤
Step 4 (CEDex生成) ───┼──→ Step 6 (集成) → Step 7 (入口) → Step 8 (验证)
Step 5 (格式转换)  ───┤
Step 3 (策略类)    ───┘
```
