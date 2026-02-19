# 功能1: MoMaGen式移动操作轨迹规划（详细对比分析版）

## 1. 两种实现方案的核心区别

### 1.1 MoMaGen方案（CuRobo框架）

MoMaGen的三阶段全部基于CuRobo运动规划框架，通过`CuRoboEmbodimentSelection`枚举切换不同的规划模式：

```python
# MoMaGen/BEHAVIOR-1K/OmniGibson/omnigibson/action_primitives/curobo.py:38-42
class CuRoboEmbodimentSelection(str, Enum):
    BASE = "base"           # 只规划底盘(x, y, θ)
    ARM = "arm"             # 规划手臂+底盘
    DEFAULT = "default"     # 全身规划
    ARM_NO_TORSO = "arm_no_torso"  # 只规划手臂(锁定底盘和躯干)
```

**MoMaGen三阶段CuRobo调用链**:

| 阶段 | CuRobo模式 | 规划内容 | 关键方法 |
|------|-----------|---------|---------|
| 导航 | `BASE` | 底盘(x,y,θ)无碰撞路径 | `_navigate_to_pose()` → `_plan_joint_motion(emb_sel=BASE)` |
| 手臂MP | `ARM_NO_TORSO` | 手臂关节空间无碰撞轨迹 | `_plan_joint_motion(emb_sel=ARM_NO_TORSO)` |
| 操作 | `DEFAULT/ARM` | 全身/手臂跟踪轨迹 | `_plan_joint_motion(emb_sel=DEFAULT)` |

**MoMaGen导航阶段的CuRobo细节**:

```python
# starter_semantic_action_primitives.py:2061-2100
def _navigate_to_pose(self, pose_2d, ...):
    q_traj = self._plan_joint_motion(
        target_pos, target_quat,
        embodiment_selection=CuRoboEmbodimentSelection.BASE,  # 只规划底盘
    )
```

CuRobo内部流程：
1. **采样目标base位置** — 在物体周围[0.4m, 1.0m]范围内采样(x,y,yaw)
2. **IK可达性验证** — 用ARM模式检查手臂能否从该base位置够到物体
3. **碰撞检查** — SDF球体碰撞检测，确保路径无碰撞
4. **轨迹优化** — Graph-based(RRT*) + TrajOpt两阶段优化
5. **输出** — 关节空间轨迹 `q_traj: (T, n_dof)`

### 1.2 之前规划的简单方案（IsaacLab P-Controller）

IsaacLab-Arena G1的WBC导航用的是纯P控制器：

```python
# p_controller.py:104-134
vx = kp_linear_x * dx_local   # P控制
vy = kp_linear_y * dy_local
wz = kp_angular * angle_error
# 速度裁剪到 [-0.4, 0.4] m/s
```

只有速度指令，没有路径规划，没有碰撞检测。

## 2. 逐维度对比

| 维度 | MoMaGen (CuRobo) | 简单方案 (P-Controller) |
|------|-------------------|------------------------|
| **导航路径** | RRT* + TrajOpt全局规划 | 直线P控制，无路径 |
| **碰撞避障** | SDF球体碰撞检测，保证无碰撞 | 无碰撞检测，可能撞桌子/椅子 |
| **目标位置采样** | 自动采样+IK可达性验证 | 手动指定固定目标点 |
| **手臂过渡** | ARM_NO_TORSO模式规划无碰撞轨迹 | 线性插值，可能穿模 |
| **速度/加速度约束** | 关节速度/加速度限制内优化 | 仅速度裁剪[-0.4, 0.4] |
| **重规划能力** | 失败自动重试(max_attempts) | 无重试机制 |
| **附着物体处理** | 支持attached_obj碰撞检测 | 不支持 |
| **可见性约束** | 支持相机视野约束 | 不支持 |
| **输出格式** | 关节空间轨迹(T, n_dof) | 速度指令(vx, vy, wz) |
| **复杂度** | 高（需配置CuRobo robot config） | 低（直接用WBC API） |

## 3. 两者优劣分析

### 3.1 CuRobo方案的优势

1. **碰撞安全**: SDF球体碰撞检测覆盖全身，导航和手臂运动都保证无碰撞
2. **智能目标采样**: 自动在物体周围采样base位置，并用IK验证手臂可达性
3. **轨迹质量**: RRT* + TrajOpt两阶段优化，轨迹平滑且满足关节限制
4. **统一框架**: 导航/手臂MP/操作三阶段用同一套CuRobo API，切换embodiment即可
5. **鲁棒性**: 内置重试机制，规划失败自动重新采样

### 3.2 CuRobo方案的劣势

1. **配置复杂**: 需要为G1人形机器人编写CuRobo robot config（URDF解析、碰撞球体、关节分组）
2. **GPU依赖**: CuRobo的CUDA Graph加速需要GPU，初始化warmup耗时
3. **人形适配**: CuRobo原生支持机械臂，人形机器人的腿部需要特殊处理（锁定/解锁关节）
4. **与WBC的衔接**: CuRobo输出关节轨迹，但G1下肢由WBC控制，需要桥接两套系统

### 3.3 P-Controller方案的优势

1. **实现简单**: 直接用IsaacLab已有的WBC API，几十行代码
2. **无额外依赖**: 不需要安装/配置CuRobo
3. **WBC原生集成**: navigate_cmd直接喂给WBC，下肢控制自然

### 3.4 P-Controller方案的劣势

1. **无碰撞检测**: 直线走向目标，遇到障碍物直接撞上去
2. **无IK可达验证**: 不知道停下来的位置手臂能不能够到物体
3. **手臂过渡粗糙**: 线性插值可能穿过桌面或物体
4. **不可泛化**: 换场景/换物体位置需要手动调参数

## 4. 推荐方案：混合架构（CuRobo规划 + WBC执行）

结合两者优势：用CuRobo做离线规划（碰撞安全+IK验证），用WBC做在线执行（稳定行走）。

```
┌─────────────────────────────────────────────────┐
│              CuRobo 离线规划层                    │
│                                                   │
│  1. 采样base目标位置(IK可达+无碰撞)              │
│  2. BASE模式规划底盘路径(避障)                    │
│  3. ARM_NO_TORSO模式规划手臂到预抓取(避障)       │
│                                                   │
│  输出: base_waypoints[], arm_trajectory[]         │
└──────────────────────┬──────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────┐
│              WBC 在线执行层                       │
│                                                   │
│  1. 导航阶段: base_waypoints → navigate_cmd      │
│     (P控制器跟踪CuRobo规划的路径点)              │
│  2. 手臂阶段: arm_trajectory → PINK IK action    │
│     (跟踪CuRobo规划的无碰撞手臂轨迹)            │
│  3. 操作阶段: replay transformed demo             │
└─────────────────────────────────────────────────┘
```

### 4.1 为什么不能纯用CuRobo

MoMaGen的机器人(R1)是轮式底盘，CuRobo BASE模式直接输出(x,y,θ)轨迹就能执行。
G1是双足人形，下肢行走必须由WBC策略控制（平衡、步态），CuRobo无法直接生成行走步态。

所以混合架构是必须的：CuRobo负责"去哪里"和"怎么避障"，WBC负责"怎么走过去"。

### 4.2 CuRobo规划层实现

参考MoMaGen的`CuRoboMotionGenerator`（curobo.py:77-264），为G1创建类似的wrapper。

**新建文件**: `IsaacLab-Arena/isaaclab_arena/planning/g1_curobo_planner.py`

```python
class G1CuRoboPlanner:
    """参考MoMaGen CuRoboMotionGenerator，为G1+InspireHand适配"""

    def __init__(self, robot_cfg_path, world_model):
        # 初始化CuRobo MotionGen（参考curobo.py:77-264）
        self.mg = MotionGen(MotionGenConfig.load_from_robot_config(
            robot_cfg_path, world_model=world_model,
            interpolation_steps=1000,
            collision_activation_distance=0.01,
        ))
        self.mg.warmup()  # CUDA graph warmup
```

**Step A: 采样base目标位置（参考MoMaGen _sample_pose_near_object）**:

```python
def sample_base_target(self, obj_pos, n_samples=64):
    """在物体周围采样base位置，验证IK可达+无碰撞"""
    candidates = []
    for _ in range(n_samples):
        dist = random.uniform(0.4, 1.0)
        yaw = random.uniform(-pi, pi)
        x = obj_pos[0] + dist * cos(yaw)
        y = obj_pos[1] + dist * sin(yaw)
        candidates.append([x, y, yaw + pi])  # 面朝物体

    # CuRobo碰撞检查 + IK可达验证
    valid = self.mg.check_collisions(candidates)
    ik_ok = self.mg.solve_ik_batch(candidates, obj_pos)
    return candidates[valid & ik_ok][0]
```

**Step B: 规划底盘路径（参考MoMaGen _navigate_to_pose）**:

```python
def plan_base_path(self, start_xy_yaw, target_xy_yaw):
    """CuRobo BASE模式规划无碰撞底盘路径"""
    result = self.mg.plan_single(
        start_state=JointState(position=start_xy_yaw),
        goal_pose=Pose(position=target_xy_yaw),
        plan_config=MotionGenPlanConfig(
            max_attempts=3,
            num_trajopt_seeds=4,
        ),
    )
    # 输出: 路径点序列 [(x,y,yaw), ...]
    return result.interpolated_plan
```

**Step C: 规划手臂无碰撞轨迹（参考MoMaGen ARM_NO_TORSO模式）**:

```python
def plan_arm_to_pregrasp(self, current_joint_pos, pregrasp_eef_pose):
    """CuRobo ARM_NO_TORSO模式规划手臂到预抓取位姿"""
    result = self.mg.plan_single(
        start_state=JointState(position=current_joint_pos),
        goal_pose=pregrasp_eef_pose,
        plan_config=MotionGenPlanConfig(max_attempts=3),
    )
    return result.interpolated_plan  # (T, arm_dof)
```

### 4.3 WBC执行层实现

CuRobo规划的base路径点转为WBC的navigation_subgoals：

```python
# CuRobo输出的路径点 → WBC subgoals格式
navigation_subgoals = []
for waypoint in base_path[::10]:  # 每10帧取一个subgoal
    navigation_subgoals.append(
        ([waypoint[0], waypoint[1], waypoint[2]], False)
    )
```

手臂轨迹通过PINK IK跟踪CuRobo规划的EEF位姿序列。

### 4.4 世界模型构建

参考BODex的WorldConfig，将IsaacLab场景导出为CuRobo碰撞模型：

```python
world_model = WorldConfig(
    cuboid=[
        Cuboid(name="table", pose=[0,0.55,0.45,...], dims=[1.2,0.8,0.9]),
        Cuboid(name="ground", pose=[0,0,0,...], dims=[10,10,0.01]),
    ],
    mesh=[
        # 可选：加入kitchen mesh做精确碰撞
    ]
)
```

## 5. G1 CuRobo Robot Config

需要为G1+InspireHand编写CuRobo robot config YAML：

```yaml
robot_cfg:
  kinematics:
    urdf_path: "g1_29dof_with_inspire_hand.urdf"
    base_link: "pelvis"
    ee_link: "right_wrist_yaw_link"
    lock_joints:  # 导航时锁定手臂，手臂规划时锁定腿
      navigation_mode: ["left_hip_*", "right_hip_*", ...]
      arm_mode: ["*_hip_*", "*_knee_*", "*_ankle_*"]
  collision:
    # 球体碰撞模型（参考BODex配置）
    spheres: [...]
```

## 6. 文件清单（更新版）

| 操作 | 文件 | 说明 |
|------|------|------|
| 新建 | `isaaclab_arena/planning/g1_curobo_planner.py` | CuRobo规划wrapper |
| 新建 | `isaaclab_arena/planning/g1_curobo_robot.yaml` | G1 CuRobo配置 |
| 新建 | `isaaclab_arena/policy/walk_to_grasp_policy.py` | 三阶段策略(CuRobo+WBC) |
| 修改 | `isaaclab_arena/embodiments/g1/g1.py` | init pos + InspireHand |
| 新建 | `scripts/run_walk_to_grasp.py` | 运行入口 |
