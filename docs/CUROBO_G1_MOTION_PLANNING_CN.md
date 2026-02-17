# G1 适配 CuRobo 运动规划方案

## 背景：当前 pipeline 的问题

当前 `bridge/build_replay.py` 从初始位置到 pregrasp 位姿**没有任何运动规划**。

`build_replay.py:528-536` 的核心逻辑：

```python
for i in range(n):
    if i <= approach_start:
        alpha = 0.0          # 手固定在 pregrasp 位姿
    elif i >= grasp_idx:
        alpha = 1.0          # 手固定在 grasp 位姿
    else:
        alpha = (i - approach_start) / max(1, grasp_idx - approach_start)
    rel_pos = (1 - alpha) * pregrasp_pos + alpha * grasp_pos   # 线性插值
    rel_quat = slerp(pregrasp_quat, grasp_quat, alpha)         # 球面插值
```

问题：
- 手腕目标可能超出 G1 工作空间（"右手飞出视角"）
- 手臂可能穿过物体或自身（无碰撞检测）
- 没有从 rest pose 到 pregrasp 的过渡轨迹——手直接"瞬移"
- 没有 IK 求解、没有可达性检查

## 参考：MoMaGen 的做法

`/home/ubuntu/DATA2/workspace/xmh/MoMaGen/` 仓库用了完全不同的方法。

每个 subtask 分两阶段执行：

**阶段 A — Motion Planning (MP)**：用 CuRobo 规划无碰撞路径

```python
# CuRoboMotionGenerator (omnigibson/action_primitives/curobo.py)
# GPU 加速轨迹优化 + 场景碰撞检测 + 自碰撞检测 + IK 求解
# PRM* 图搜索作为 fallback
mp_trajectory = curobo.plan(current_pose, first_waypoint_pose)
```

**阶段 B — Trajectory Replay**：回放变换后的源示教轨迹

```python
# 核心公式: T_current_obj × T_source_obj⁻¹ × T_source_eef
for waypoint in remaining_waypoints:
    action = env_interface.waypoint_to_action(waypoint)
```

### 对比

| 维度 | build_replay.py (当前) | MoMaGen |
|---|---|---|
| 初始→pregrasp | 无过渡，直接设定 | CuRobo 规划无碰撞路径 |
| pregrasp→grasp | 线性/SLERP 插值 | 源示教轨迹回放 |
| 碰撞检测 | 无 | CuRobo mesh collision |
| IK 求解 | 无 | CuRobo IK solver |
| 可达性检查 | 无 | 有，不可达时触发导航 |
| 机器人 | G1 (23D WBC+PINK) | R1/Tiago (OmniGibson IK/OSC) |
| 仿真环境 | IsaacLab-Arena | OmniGibson (BEHAVIOR-1K) |

## 整体架构

```
build_replay.py (离线)
    │
    ├─ 阶段 1-3: 不变（加载 HOI、重采样、确定 grasp 位姿）
    │
    ├─ 【新增】阶段 3.5: CuRobo 规划 rest → pregrasp 轨迹
    │   ├─ 加载 G1 CuRobo config + URDF
    │   ├─ IK 检查 pregrasp/grasp 可达性
    │   ├─ MotionGen.plan_single(rest_q → pregrasp_pose)
    │   └─ 输出: 关节空间轨迹 → FK → pelvis 坐标系手腕轨迹
    │
    ├─ 阶段 4-6: 基本不变（base 规划、手腕轨迹、组装 23D）
    │   └─ 但 approach 阶段改用 CuRobo 轨迹替代线性插值
    │
    └─ 输出: replay_actions.hdf5
```

核心思路：CuRobo 在离线阶段运行，规划关节空间轨迹，再通过 FK 转为 WBC+PINK 控制器能消费的 pelvis 坐标系手腕目标。不改动 Isaac 回放侧的任何代码。

## 第一步：创建 G1 CuRobo Robot Config

### 可复用的现有资源

| 资源 | 路径 | 用途 |
|---|---|---|
| G1 URDF (29DOF) | `tmp/unitree_rl_gym/resources/robots/g1_description/g1_29dof.urdf` | 运动学链 |
| G1 碰撞球 | `Dex_loco/genie_sim/.../G1/G1_omnipicker_fixed_right.yaml` | 已有 ~200 个碰撞球 |
| G1 关节顺序 | `IsaacLab-Arena/isaaclab_arena_g1/.../lab_g1_joints_order_43dof.yaml` | 关节映射 |
| CuRobo 格式参考 | `BEHAVIOR-1K/.../r1/curobo/r1_description_curobo_arm.yaml` | YAML 格式模板 |
| CuRobo 库源码 | `BODex/src/curobo/` | 完整 CuRobo 库 |
| WBC PINK 控制器 | `IsaacLab-Arena/isaaclab_arena_g1/.../g1_wbc_upperbody_controller.py` | pelvis 坐标系 IK |
| WBC 配置 | `IsaacLab-Arena/isaaclab_arena_g1/.../config/g1_homie_v2.yaml` | 控制器参数 |

### 需要创建的文件

`Humanoid-gen-pack/curobo_configs/g1_right_arm.yaml`

关键配置结构：

```yaml
robot_cfg:
  kinematics:
    base_link: "pelvis"
    ee_link: "right_wrist_yaw_link"   # G1 右手腕末端
    collision_link_names:
      - pelvis
      - torso_link
      - right_shoulder_pitch_link
      - right_shoulder_roll_link
      - right_shoulder_yaw_link
      - right_elbow_link
      - right_wrist_roll_link
      - right_wrist_pitch_link
      - right_wrist_yaw_link
    collision_spheres:
      # 从 G1_omnipicker_fixed_right.yaml 转换
      # 该文件已有 arm_r_link1~6 + body 的碰撞球
      pelvis:
        - {center: [-0.109, 0.0, 0.425], radius: 0.15}
        # ...
      right_shoulder_pitch_link:
        - {center: [0.0, 0.0, 0.0], radius: 0.03}
        # ...
    cspace:
      joint_names:
        - right_shoulder_pitch    # G1 右臂 7 DOF
        - right_shoulder_roll
        - right_shoulder_yaw
        - right_elbow
        - right_wrist_roll
        - right_wrist_pitch
        - right_wrist_yaw
      max_acceleration: [10, 10, 10, 10, 10, 10, 10]
      max_jerk: [10000, 10000, 10000, 10000, 10000, 10000, 10000]
    self_collision_ignore:
      right_shoulder_pitch_link: ["pelvis", "torso_link"]
      right_shoulder_roll_link: ["right_shoulder_pitch_link"]
      # ... 相邻 link 对
```

### Link Name 映射

碰撞球数据在 `G1_omnipicker_fixed_right.yaml` 中使用 genie_sim 命名，需要映射到 G1 URDF 标准命名：

```
genie_sim 命名          →  G1 URDF 标准命名
─────────────────────────────────────────────
base_link               →  pelvis
body_link1              →  waist_yaw_link (或 torso_link)
body_link2              →  torso_link
arm_r_base_link         →  right_shoulder_pitch_link
arm_r_link1             →  right_shoulder_roll_link
arm_r_link2             →  right_shoulder_yaw_link
arm_r_link3             →  right_elbow_link
arm_r_link4             →  right_wrist_roll_link
arm_r_link5             →  right_wrist_pitch_link
arm_r_link6             →  right_wrist_yaw_link
```

> 注意：需要对照实际使用的 URDF 确认映射关系。不同 G1 URDF 变体的 link 命名可能不同。

## 第二步：集成到 build_replay.py

在 `build_replay.py` 中新增 CuRobo 规划函数，在 pregrasp/grasp 位姿确定后、组装 23D action 之前调用：

```python
def _plan_approach_with_curobo(
    rest_q: np.ndarray,           # G1 右臂 7DOF 初始关节角
    pregrasp_pose_w: np.ndarray,  # pregrasp 世界坐标系 4x4
    grasp_pose_w: np.ndarray,     # grasp 世界坐标系 4x4
    robot_cfg_path: str,          # CuRobo config YAML
    urdf_path: str,               # G1 URDF
    device: str = "cuda:0",
) -> tuple[np.ndarray, np.ndarray] | None:
    """
    用 CuRobo 规划 rest → pregrasp → grasp 的关节空间轨迹。
    返回 (joint_traj [T, 7], ee_traj_pelvis [T, 7]) 或 None（规划失败时 fallback）。
    """
    from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig
    from curobo.types.robot import JointState
    from curobo.types.math import Pose

    # 1. 初始化 MotionGen
    config = MotionGenConfig.load_from_robot_config(
        robot_cfg=robot_cfg_path,
        num_ik_seeds=32,
        num_trajopt_seeds=4,
        interpolation_dt=0.02,  # 50 Hz，匹配 Isaac Sim
    )
    motion_gen = MotionGen(config)
    motion_gen.warmup(enable_graph=True, batch=1)

    # 2. IK 可达性检查
    pregrasp_goal = Pose(
        position=pregrasp_pose_w[:3, 3],
        quaternion=rotmat_to_quat_xyzw(pregrasp_pose_w[:3, :3]),
    )
    ik_result = motion_gen.solve_ik(pregrasp_goal)
    if not ik_result.success.item():
        print("[CuRobo] pregrasp IK failed, falling back to linear interp")
        return None

    # 3. 规划 rest → pregrasp
    start_state = JointState.from_position(
        torch.tensor([rest_q], device=device)
    )
    plan_result = motion_gen.plan_single(
        start_state, pregrasp_goal,
        MotionGenPlanConfig(enable_graph=True, max_attempts=30)
    )
    if not plan_result.success.item():
        print("[CuRobo] motion planning failed, falling back to linear interp")
        return None

    # 4. 提取插值后的轨迹
    joint_traj = plan_result.get_interpolated_plan()  # [T, 7]

    # 5. FK → pelvis 坐标系手腕位姿
    ee_traj = motion_gen.compute_kinematics(joint_traj)
    return joint_traj.position.cpu().numpy(), ee_traj
```

### CuRobo MotionGen API 要点

| 方法 | 用途 |
|---|---|
| `MotionGenConfig.load_from_robot_config()` | 从 YAML 加载机器人配置 |
| `MotionGen.warmup()` | 预热 CUDA graph（首次调用慢，后续快 10x） |
| `MotionGen.solve_ik(goal_pose)` | 纯 IK 求解（可达性检查） |
| `MotionGen.plan_single(start, goal)` | 单次运动规划（轨迹优化 + 碰撞避免） |
| `MotionGen.plan_batch(start, goal)` | 批量规划 |
| `result.get_interpolated_plan()` | 获取密集插值轨迹 |
| `result.success` | 规划是否成功 |
| `result.motion_time` | 轨迹执行时间 |

### MotionGenPlanConfig 关键参数

```python
MotionGenPlanConfig(
    enable_graph=True,       # 启用 PRM* 图搜索（作为 fallback）
    enable_opt=True,         # 启用轨迹优化
    max_attempts=30,         # 最大尝试次数
    timeout=10.0,            # 超时（秒）
    enable_finetune_trajopt=True,  # 精细化优化
)
```

## 第三步：关节空间 → pelvis 坐标系手腕目标

这是最关键的衔接点。CuRobo 输出关节空间轨迹，但 G1 WBC+PINK 控制器接受 pelvis 坐标系下的手腕位姿。

### 转换链

```
CuRobo 关节轨迹 q[t]
    ↓ FK (pinocchio / CuRobo kinematics)
手腕在 base_link 坐标系下的位姿 T_wrist_base[t]
    ↓ 坐标变换（base_link ≈ pelvis for G1）
手腕在 pelvis 坐标系下的位姿 → 直接填入 23D action [9:16]
```

G1 的 WBC+PINK 控制器（`g1_wbc_upperbody_controller.py`）内部用 pinocchio + PINK 做 IK，接受的输入就是 pelvis 坐标系下的手腕目标位姿。所以 CuRobo 规划出的轨迹经过 FK 转换后，可以直接替代原来的线性插值结果。

### pinocchio FK 实现

```python
import pinocchio as pin

model = pin.buildModelFromUrdf(urdf_path)
data = model.createData()
wrist_frame_id = model.getFrameId("right_wrist_yaw_link")

ee_poses_pelvis = []
for t in range(len(joint_traj)):
    pin.forwardKinematics(model, data, joint_traj[t])
    pin.updateFramePlacements(model, data)
    T_wrist_pelvis = data.oMf[wrist_frame_id]  # SE3
    pos = T_wrist_pelvis.translation            # [3]
    quat = pin.Quaternion(T_wrist_pelvis.rotation)  # wxyz
    ee_poses_pelvis.append((pos, quat))
    # 填入 action[t, 9:12] = pos
    # 填入 action[t, 12:16] = quat (wxyz)
```

## 第四步：修改 build_replay.py 主循环

当前 `build_replay.py:528-561` 的 per-frame 循环改为：

```python
# 尝试 CuRobo 规划
curobo_result = None
if args.use_curobo:
    # 计算 approach_start 帧的 pregrasp 世界坐标系位姿
    t_obj_w = make_pose(obj_pos[approach_start], obj_rot[approach_start])
    t_pregrasp_w = t_obj_w @ make_pose(rel_pose.pregrasp_pos_obj,
                                        rel_pose.pregrasp_quat_obj_wxyz)
    t_grasp_w = t_obj_w @ make_pose(rel_pose.grasp_pos_obj,
                                     rel_pose.grasp_quat_obj_wxyz)

    curobo_result = _plan_approach_with_curobo(
        rest_q=G1_RIGHT_ARM_REST_Q,
        pregrasp_pose_w=t_pregrasp_w,
        grasp_pose_w=t_grasp_w,
        robot_cfg_path=args.curobo_config,
        urdf_path=args.g1_urdf,
    )

if curobo_result is not None:
    # 用 CuRobo 轨迹替代 approach 阶段的线性插值
    curobo_joint_traj, curobo_ee_pelvis = curobo_result
    # 将 CuRobo 轨迹插入 [0, approach_start] 区间
    # approach_start 之后仍用原逻辑（跟随物体运动）
else:
    # fallback: 原来的线性插值逻辑（完全不变）
    for i in range(n):
        alpha = ...
```

新增命令行参数：

```python
parser.add_argument("--use-curobo", action="store_true",
    help="Use CuRobo for rest→pregrasp motion planning (requires CUDA).")
parser.add_argument("--curobo-config",
    default="curobo_configs/g1_right_arm.yaml",
    help="Path to G1 CuRobo robot config YAML.")
parser.add_argument("--g1-urdf",
    default="repos/g1_description/g1_29dof.urdf",
    help="Path to G1 URDF for FK computation.")
```

## 文件变更清单

| 文件 | 变更类型 | 说明 |
|---|---|---|
| `curobo_configs/g1_right_arm.yaml` | 新增 | G1 右臂 CuRobo 配置 |
| `curobo_configs/g1_right_arm_collision_spheres.yaml` | 新增 | 碰撞球数据（从 genie_sim 转换） |
| `bridge/build_replay.py` | 修改 | 新增 `--use-curobo` + `_plan_approach_with_curobo()` |
| `bridge/g1_fk_utils.py` | 新增 | pinocchio FK 工具函数 |
| `envs/hoifhli_env.yml` | 修改 | 加 curobo + pinocchio 依赖 |
| `scripts/03_build_replay.sh` | 修改 | 加 `--use-curobo` 选项 |

## 关键难点和风险

### 1. URDF Link Name 映射

G1 有多个 URDF 变体（29dof、23dof、with_hand），不同项目用不同的 link 命名。需要确认 IsaacLab-Arena 用的是哪个 URDF，保持一致。

### 2. Pelvis 不是固定的

CuRobo 假设 base_link 固定，但 G1 是移动机器人，pelvis 在走路时会晃动。

解决方案：CuRobo 只规划上半身（pelvis 作为 fixed base），规划时假设 pelvis 静止。这在 approach 阶段（base 已停止移动）是合理的。

### 3. 碰撞球精度

`G1_omnipicker_fixed_right.yaml` 里的碰撞球是给 omnipicker 夹爪配的，不是 G1 原装手。如果 G1 用的是 Unitree 原装手或 Inspire 灵巧手，碰撞球需要重新标定。

### 4. CuRobo 依赖

CuRobo 需要 CUDA + PyTorch，编译时间较长。可以考虑先用 pinocchio 做纯 IK 可达性检查（轻量级），CuRobo 作为可选增强。

### 5. 离线 vs 在线

当前方案是离线规划（`build_replay.py` 里跑 CuRobo），不影响 Isaac 回放。如果未来要做在线闭环，需要把 CuRobo 集成到 Isaac 环境里，那是另一个量级的工作。

## 推荐实施顺序

1. **pinocchio IK 可达性检查**（不需要 CuRobo，验证 pregrasp 位姿是否在 G1 工作空间内）
2. **创建 G1 CuRobo config YAML**（从现有碰撞球数据转换 link name 映射）
3. **集成 CuRobo MotionGen** 到 `build_replay.py`
4. **测试 + 调参**（碰撞球 buffer、IK seeds、trajopt steps）

## TODO 落地执行清单（按优先级）

### TODO-1: IK 可达性守门（1-2 天）

目标：先解决“右手飞出视角”，在不接入 CuRobo 的情况下，过滤不可达 wrist 目标。

实现：
- 在 `bridge/build_replay.py` 新增 `--use-pin-ik-check`。
- 使用 `pinocchio` 对每个 pregrasp/grasp pose 做右臂 7DOF IK（或数值逆解）。
- IK 失败时回退：
  - 先扩大 `pregrasp` 偏置（远离物体）
  - 再降低 `grasp_frame_ratio`（让 approach 更早开始）
  - 仍失败则标记该样本 `unreachable` 并跳过

验收标准：
- `bridge_debug.json` 新增字段：`ik_reachable`, `ik_fail_reason`
- 回放不再出现明显手腕瞬移/飞出

### TODO-2: G1 右臂 CuRobo 配置（2-3 天）

目标：把 G1 右臂建立为可被 MotionGen 调用的机器人模型。

实现：
- 新增 `curobo_configs/g1_right_arm.yaml`
- 新增 `curobo_configs/g1_right_arm_collision_spheres.yaml`
- 从 `G1_omnipicker_fixed_right.yaml` 导出碰撞球并做 link 名映射

验收标准：
- 用独立脚本可完成一次 `rest -> pregrasp` 规划并返回 `success=True`
- 保存轨迹 `npz`（joint + ee pose）

### TODO-3: 集成 MotionGen 到 bridge（2-4 天）

目标：替换当前线性插值 approach 段。

实现：
- `bridge/build_replay.py` 新增参数：
  - `--use-curobo`
  - `--curobo-config`
  - `--g1-urdf`
  - `--curobo-device`（`cuda:0` / `cpu`）
- 仅在 `[0, approach_start]` 段使用 CuRobo 轨迹，后续仍保持原逻辑。
- 规划失败自动 fallback 到原线性插值。

验收标准：
- `bridge_debug.json` 新增字段：`use_curobo`, `curobo_success`, `curobo_fallback`
- 与原 pipeline 对比，失败率下降、轨迹更平滑

### TODO-4: 端到端回放回归（1-2 天）

目标：保证改动不破坏现有 Path A 回放流程。

实现：
- 固定输入：
  - `human_object_results.pkl`
  - `grasp.npy`（或手工 fallback grasp）
- 分别跑：
  - `--use-curobo` 开
  - `--use-curobo` 关
- 对比：终点误差、回放稳定性、视频可视效果

验收标准：
- 两条链路都能生成 `replay_actions.hdf5` + `object_kinematic_traj.npz`
- `--use-curobo` 版本在 approach 段不再出现明显穿模/突变

## 附录：CuRobo MotionGenResult 输出结构

```python
result = motion_gen.plan_single(start_state, goal_pose, plan_config)

# 成功判断
result.success          # torch.Tensor, bool
result.status           # MotionGenStatus (IK_FAIL / GRAPH_FAIL / TRAJOPT_FAIL / SUCCESS)

# 轨迹数据
result.optimized_plan   # JointState, 稀疏 waypoints
result.optimized_dt     # 优化后的时间步
result.get_interpolated_plan()  # JointState, 密集插值轨迹
result.interpolation_dt # 插值时间步

# 误差指标
result.position_error   # L2 距离 (米)
result.rotation_error   # 四元数距离

# 计时
result.solve_time       # 总求解时间
result.ik_time          # IK 求解时间
result.trajopt_time     # 轨迹优化时间
result.motion_time      # 轨迹执行时间
```
