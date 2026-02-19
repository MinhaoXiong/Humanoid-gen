# 功能1: Walk-to-Grasp（基于现有08 pipeline扩展）

## 1. 现有pipeline分析

### 1.1 当前 `08_debug_arm_follow_gui.sh` 做了什么

三步链路：

1. **`generate_debug_object_traj.py`** — 生成合成物体轨迹（lift_place模式：抬起→平移→放下）
   - 输出: `object_kinematic_traj.npz`（`object_pos_w[T,3]`, `object_quat_wxyz[T,4]`）
   - kitchen场景默认: start=(0.4,0.0,0.1), end=(0.2,0.4,0.1), lift_height=0.35

2. **`build_arm_follow_replay.py`** — 物体轨迹 → 23D G1动作序列
   - 右手腕在物体坐标系下固定偏移: `--right-wrist-pos-obj=-0.20,-0.03,0.10`
   - **`nav_cmd`全部设为0**（第220行: `actions[:, NAV_CMD_START_IDX:NAV_CMD_END_IDX] = 0.0`）
   - base固定在 `BASE_POS_W=0.05,0.0,0.0`
   - 输出: `replay_actions_arm_follow.hdf5`

3. **`policy_runner_kinematic_object_replay.py`** — IsaacLab仿真执行
   - 加载hdf5 replay动作 + npz物体轨迹
   - G1初始位置通过 `--g1-init-pos-w` 设置
   - embodiment: `g1_wbc_pink`

### 1.2 当前的限制

- 机器人固定在桌边，不走动
- `nav_cmd=0` 意味着WBC的P-controller不会被触发
- 没有"先走过去再操作"的阶段划分

## 2. 改动方案

### 2.1 核心思路

在 `build_arm_follow_replay.py` 的输出中加入**导航阶段**：

```
时间轴: [0 -------- nav_end -------- T]
         |  导航阶段  |  手臂跟随阶段  |
nav_cmd:  [非零速度]    [0,0,0]
arm:      [固定姿态]    [跟随物体]
```

### 2.2 参考 `build_replay.py` 的nav_cmd计算

`bridge/build_replay.py:374-390` 已有完整的nav_cmd计算逻辑：
```python
dxy_w = (base_xy[i+1] - base_xy[i]) * target_fps
vx = cos(yaw)*dxy_w[0] + sin(yaw)*dxy_w[1]   # 世界→base坐标系
vy = -sin(yaw)*dxy_w[0] + cos(yaw)*dxy_w[1]
nav_cmd[i] = clip([vx, vy, dyaw], -0.4, 0.4)
```

### 2.3 WBC P-controller已有的能力

`g1_decoupled_wbc_pink_action.py` 中：
- 当 `navigate_cmd > NAVIGATE_THRESHOLD(1e-4)` 时自动触发导航
- `navigation_subgoals` 列表定义路径点
- P-controller跟踪每个subgoal直到到达阈值

## 3. 具体改动文件

### 3.1 修改 `isaac_replay/build_arm_follow_replay.py`

新增参数：
- `--nav-target-pos-w` — 导航目标位置(x,y)，即桌边停靠点
- `--nav-start-pos-w` — 导航起始位置(x,y)，即机器人出生点
- `--nav-duration-sec` — 导航阶段时长（默认4秒）

改动逻辑：
1. 前 `nav_frames` 帧：计算从start到target的nav_cmd（参考build_replay.py）
2. 前 `nav_frames` 帧：手臂保持固定姿态（不跟随物体）
3. `nav_frames` 之后：nav_cmd=0，手臂开始跟随物体（现有逻辑）

### 3.2 修改 `scripts/08_debug_arm_follow_gui.sh`

新增环境变量：
- `NAV_START_POS_W` — 机器人出生位置（远离桌子）
- `NAV_TARGET_POS_W` — 导航目标（桌边）

传递给 `build_arm_follow_replay.py` 和 `policy_runner` 的 `--g1-init-pos-w`。

### 3.3 不需要改的文件

- `generate_debug_object_traj.py` — 物体轨迹不变
- `policy_runner_kinematic_object_replay.py` — 已支持replay + kinematic object
- WBC/P-controller — 已有导航能力，只要action里nav_cmd非零就会触发

## 4. 23D动作向量各阶段值

| 维度 | 导航阶段 | 手臂跟随阶段 |
|------|---------|-------------|
| [0] left_hand | 0.0 | 0.0 |
| [1] right_hand | 0.0 | 0.0 |
| [2:9] left_wrist | 固定pelvis姿态 | 固定pelvis姿态 |
| [9:16] right_wrist | 固定pelvis姿态 | 跟随物体（现有逻辑） |
| [16:19] nav_cmd | 非零(vx,vy,wz) | 0,0,0 |
| [19] base_height | 0.75 | 0.75 |
| [20:23] torso_rpy | 0,0,0 | 0,0,0 |
