# 功能1: MoMaGen式移动操作轨迹规划

## 1. 当前状态分析

### 当前场景配置
- **机器人位置**: G1 spawn在 `(0.8, -1.38, 0.78)`，紧贴桌子边缘
- **桌子位置**: `(0.0, 0.55, 0.0)`
- **Cracker Box位置**: `(0.4, 0.0, 0.1)` 相对桌面
- **操作方式**: 纯手臂操作（Differential IK），无移动底盘

### 当前问题
机器人直接生成在桌子旁，只做手臂操作，缺少"走过去"的过程。

## 2. 目标架构：三阶段流水线

参考MoMaGen的三阶段设计：

```
阶段1: Navigation（导航）
  机器人从远处走向桌子/物体
  ↓ 到达物体附近时停止
阶段2: Pre-Grasp Transition（预抓取过渡）
  切换到手臂控制，规划到预抓取位姿
  ↓ 手臂到达预抓取位姿
阶段3: Manipulation Replay（操作回放）
  手臂跟随物体执行抓取/放置轨迹
```

## 3. MoMaGen核心机制分析

### 3.1 MoMaGen的三阶段数据流

**关键文件**:
- `MoMaGen/momagen/datagen/data_generator.py` — 主生成循环
- `MoMaGen/momagen/datagen/waypoint.py` — 导航/MP/Replay执行

**核心算法 — 位姿相对变换**:
```python
# 对每个时间步t:
T_new_eef[t] = T_current_obj @ inv(T_source_obj) @ T_source_eef[t]
```
源demo的EEF轨迹存储为相对于物体的位姿，通过变换适配到新物体位置。

### 3.2 导航阶段细节

MoMaGen导航阶段 (`waypoint.py:1252-1460`):
1. 调用 `env.primitive._navigate_to_obj()` 采样目标base位置
2. 采样条件：手臂IK可达 + 物体可见 + 无碰撞
3. CuRobo BASE模式（3 DOF: x, y, θ）规划无碰撞路径
4. 执行导航并收集 states/obs/actions

### 3.3 IsaacLab-Arena中G1的WBC导航能力

G1 embodiment已有导航支持（`g1.py`中的WBC Pink配置）:
- `is_navigating` 观测项 — 判断是否在导航中
- `navigation_goal_reached` 观测项 — 判断是否到达目标
- `navigate_cmd` 动作项 — 导航命令 (3D: vx, vy, ωz)
- `base_height_cmd` — 底盘高度命令
- `torso_orientation_rpy_cmd` — 躯干朝向命令

## 4. 具体实现方案

### 4.1 修改机器人初始位置

**文件**: `IsaacLab-Arena/isaaclab_arena/embodiments/g1/g1.py`

将G1 spawn位置从桌子边缘移到远处：
```python
# 当前: pos=(0.8, -1.38, 0.78) — 紧贴桌子
# 修改为: pos=(0.8, -3.0, 0.78) — 离桌子约1.6m远
init_state=ArticulationCfg.InitialStateCfg(
    pos=(0.8, -3.0, 0.78),
    ...
)
```

### 4.2 新建三阶段轨迹策略

**新文件**: `IsaacLab-Arena/isaaclab_arena/policy/walk_to_grasp_policy.py`

```python
class WalkToGraspPolicy(PolicyBase):
    """三阶段策略: 导航 → 预抓取 → 操作回放"""

    class Phase(Enum):
        NAVIGATION = 0      # 走向物体
        PRE_GRASP = 1       # 切换到预抓取位姿
        MANIPULATION = 2    # 手臂replay

    def __init__(self, ...):
        self.phase = Phase.NAVIGATION
        self.nav_target_pos = None      # 导航目标(桌子附近)
        self.pre_grasp_pose = None      # 预抓取EEF位姿
        self.replay_trajectory = None   # 操作回放轨迹
```

### 4.3 阶段1: Navigation实现

利用G1 WBC已有的导航能力，通过`navigate_cmd`发送速度指令走向物体。

**导航目标计算**:
```python
# 物体世界坐标
obj_pos_world = table_pos + cracker_box_offset  # 约(0.4, 0.55, 0.1+桌高)
# 导航停止位置: 物体前方约0.5m（手臂可达范围内）
nav_target = obj_pos_world - forward_dir * 0.5
```

**停止条件**:
```python
dist_to_target = torch.norm(robot_base_pos[:2] - nav_target[:2])
if dist_to_target < 0.15:  # 15cm阈值
    self.phase = Phase.PRE_GRASP
```

**关键依赖**:
- `G1DecoupledWBCPinkActionCfg` — 已支持navigate_cmd
- `g1_observations_mdp.is_navigating` — 导航状态观测
- `g1_observations_mdp.navigation_goal_reached` — 到达判断

### 4.4 阶段2: Pre-Grasp Transition实现

导航停止后，发送navigate_cmd=0停止行走，切换到手臂PINK IK控制。

**预抓取位姿计算**:
```python
# 预抓取位姿 = 物体上方10-15cm，手掌朝下
pre_grasp_pos = obj_pos + torch.tensor([0, 0, 0.12])
pre_grasp_rot = top_down_orientation  # 手掌朝下的四元数
```

**过渡逻辑**:
```python
# 线性插值从当前EEF位姿到预抓取位姿
alpha = step / num_interpolation_steps
target_eef = (1 - alpha) * current_eef + alpha * pre_grasp_pose
# 通过PINK IK解算关节角
action = pink_ik_solve(target_eef)
```

### 4.5 阶段3: Manipulation Replay实现

参考 `replay_automoma_trajectory_policy.py` 的replay机制。

**位姿相对变换（MoMaGen核心）**:
```python
# 加载源demo的EEF轨迹和物体位姿
src_eef_traj = load_source_demo()       # (T, 4, 4)
src_obj_pose = src_eef_traj_obj_pose    # (4, 4) 第0帧物体位姿

# 当前环境物体位姿
cur_obj_pose = get_current_obj_pose()   # (4, 4)

# 变换每个时间步
for t in range(T):
    new_eef[t] = cur_obj_pose @ inv(src_obj_pose) @ src_eef_traj[t]
```

**动作格式（G1 WBC Pink）**:
```python
action = torch.cat([
    left_gripper,           # (1,)
    right_gripper,          # (1,)
    left_eef_pos,           # (3,)
    left_eef_quat,          # (4,)
    right_eef_pos,          # (3,)
    right_eef_quat,         # (4,)
    navigate_cmd_zero,      # (3,) 全零=不导航
    base_height_cmd,        # (1,)
    torso_orientation_cmd,  # (3,)
], dim=0)
```

## 5. 需要修改/新建的文件清单

| 操作 | 文件路径 | 说明 |
|------|----------|------|
| 修改 | `IsaacLab-Arena/isaaclab_arena/embodiments/g1/g1.py` | 修改init pos远离桌子 |
| 新建 | `IsaacLab-Arena/isaaclab_arena/policy/walk_to_grasp_policy.py` | 三阶段策略 |
| 修改 | `IsaacLab-Arena/isaaclab_arena/examples/.../kitchen_pick_and_place_environment.py` | 集成新策略 |
| 新建 | `IsaacLab-Arena/isaaclab_arena/scripts/run_walk_to_grasp.py` | 运行入口脚本 |
