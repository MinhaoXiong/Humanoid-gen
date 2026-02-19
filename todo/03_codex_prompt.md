# 给Codex的Prompt（基于08 pipeline扩展Walk-to-Grasp）

## 项目背景

工作目录: `/home/ubuntu/DATA2/workspace/xmh/Humanoid-gen-pack`

现有pipeline `scripts/08_debug_arm_follow_gui.sh` 实现了G1机器人**固定站位+手臂跟随物体**的debug场景。三步链路：
1. `isaac_replay/generate_debug_object_traj.py` — 生成合成物体轨迹(lift_place)
2. `isaac_replay/build_arm_follow_replay.py` — 物体轨迹→23D G1动作序列(hdf5)
3. `isaac_replay/policy_runner_kinematic_object_replay.py` — IsaacLab仿真执行

运行命令：
```bash
cd "$PACK_ROOT"
DEVICE="cuda:0" BASE_POS_W="0.05,0.0,0.0" G1_INIT_YAW_DEG="0.0" \
bash scripts/08_debug_arm_follow_gui.sh \
  "$PACK_ROOT/artifacts/debug_schemeA2_gui" \
  lift_place kitchen_pick_and_place cracker_box
```

## 当前限制

`build_arm_follow_replay.py` 第220行把 `nav_cmd` 全部设为0：
```python
actions[:, NAV_CMD_START_IDX:NAV_CMD_END_IDX] = 0.0  # keep base/pelvis fixed
```
机器人固定在桌边不走动，只有手臂跟随物体。

## 任务目标

扩展为**Walk-to-Grasp**：机器人从远处出生 → 走向桌子 → 停下后手臂跟随物体。

---

## Step 1: 修改 `isaac_replay/build_arm_follow_replay.py`

这是核心改动。当前该文件生成固定base的23D动作序列，需要加入导航阶段。

### 1.1 新增CLI参数

在 `_make_parser()` 中新增：
```python
parser.add_argument("--nav-start-pos-w", default=None,
    help="Navigation start position x,y,z. If set, enables walk-to phase.")
parser.add_argument("--nav-target-pos-w", default=None,
    help="Navigation target position x,y,z (where robot stops near table).")
parser.add_argument("--nav-duration-sec", type=float, default=4.0,
    help="Duration of navigation phase in seconds.")
parser.add_argument("--nav-fps", type=float, default=50.0,
    help="Action FPS (must match Isaac sim rate).")
```

当 `--nav-start-pos-w` 被设置时，启用walk-to模式；否则保持现有固定base行为。

### 1.2 导航阶段nav_cmd计算

参考 `bridge/build_replay.py:374-390` 的nav_cmd计算逻辑。在 `main()` 中，当walk-to模式启用时：

```python
nav_frames = int(args.nav_duration_sec * args.nav_fps)
total_frames = nav_frames + n  # 导航帧 + 原有手臂跟随帧

# 导航阶段：从start直线走到target
nav_start = _parse_csv_floats(args.nav_start_pos_w, 3, "nav_start_pos_w")
nav_target = _parse_csv_floats(args.nav_target_pos_w, 3, "nav_target_pos_w")

# 线性插值base位置
t_nav = np.linspace(0, 1, nav_frames)
base_xy = np.outer(1-t_nav, nav_start[:2]) + np.outer(t_nav, nav_target[:2])

# base朝向：面朝目标
direction = nav_target[:2] - nav_start[:2]
base_yaw = np.full(nav_frames, math.atan2(direction[1], direction[0]))

# 计算速度指令（世界→base坐标系变换）
nav_cmd = np.zeros((nav_frames, 3))
for i in range(nav_frames - 1):
    dxy_w = (base_xy[i+1] - base_xy[i]) * args.nav_fps
    yaw = base_yaw[i]
    vx = math.cos(yaw)*dxy_w[0] + math.sin(yaw)*dxy_w[1]
    vy = -math.sin(yaw)*dxy_w[0] + math.cos(yaw)*dxy_w[1]
    nav_cmd[i] = np.clip([vx, vy, 0.0], -0.4, 0.4)
```

### 1.3 拼接导航+手臂跟随

```python
# 导航阶段动作：手臂固定，nav_cmd非零
nav_actions = np.zeros((nav_frames, ACTION_DIM), dtype=np.float32)
nav_actions[:, LEFT_WRIST_POS_START_IDX:LEFT_WRIST_POS_END_IDX] = left_wrist_pos
nav_actions[:, LEFT_WRIST_QUAT_START_IDX:LEFT_WRIST_QUAT_END_IDX] = left_wrist_quat
nav_actions[:, RIGHT_WRIST_POS_START_IDX:RIGHT_WRIST_POS_END_IDX] = left_wrist_pos  # 右手也固定
nav_actions[:, RIGHT_WRIST_QUAT_START_IDX:RIGHT_WRIST_QUAT_END_IDX] = left_wrist_quat
nav_actions[:, NAV_CMD_START_IDX:NAV_CMD_END_IDX] = nav_cmd
nav_actions[:, BASE_HEIGHT_IDX] = args.base_height

# 手臂跟随阶段动作：现有逻辑（nav_cmd=0）
arm_actions = actions  # 现有的build结果

# 拼接
all_actions = np.concatenate([nav_actions, arm_actions], axis=0)
```

### 1.4 注意事项

- 手臂跟随阶段的 `base_pos_w` 应该用 `nav_target`（导航终点），因为机器人走到那里后base就在那
- 所以现有的 `--base-pos-w` 参数在walk-to模式下应自动设为 `nav_target`
- `--base-yaw` 也应自动设为导航终点朝向

---

## Step 2: 新建 `scripts/13_walk_to_arm_follow_gui.sh`

复制 `scripts/08_debug_arm_follow_gui.sh`，修改为walk-to版本。

关键改动：

```bash
# 新增变量
NAV_START_POS_W=${NAV_START_POS_W:-"0.8,-3.0,0.0"}   # 远离桌子的出生点
NAV_TARGET_POS_W=${NAV_TARGET_POS_W:-"0.05,0.0,0.0"}  # 桌边停靠点(原BASE_POS_W)

# build_arm_follow_replay.py 调用新增参数
"$PYTHON" "$PACK_ROOT/isaac_replay/build_arm_follow_replay.py" \
  ... \
  --nav-start-pos-w "$NAV_START_POS_W" \
  --nav-target-pos-w "$NAV_TARGET_POS_W" \
  --nav-duration-sec 4.0

# policy_runner 的 --g1-init-pos-w 改为出生点
cmd_runner+=(--g1-init-pos-w "$NAV_START_POS_W" --g1-init-yaw-deg "$G1_INIT_YAW_DEG")

# --max-steps 需要增加导航帧数（4秒×50fps=200帧）
--max-steps 608  # 原408 + 200导航帧
```

kitchen场景坐标参考：
- 桌子位置: `(0.0, 0.55, 0.0)`
- cracker_box初始: `(0.4, 0.0, 0.1)`
- 原G1站位: `(0.05, 0.0, 0.0)` 或 `(0.8, -1.38, 0.78)`

---

## Step 3: 验证

运行新脚本，确认：
1. 机器人从远处出生
2. 导航阶段：机器人稳定走向桌子（WBC P-controller被nav_cmd触发）
3. 到达后停下：nav_cmd归零，机器人站稳
4. 手臂跟随阶段：右手跟随cracker_box的lift_place轨迹

```bash
cd /home/ubuntu/DATA2/workspace/xmh/Humanoid-gen-pack
DEVICE="cuda:0" NAV_START_POS_W="0.8,-3.0,0.0" NAV_TARGET_POS_W="0.05,0.0,0.0" \
bash scripts/13_walk_to_arm_follow_gui.sh \
  artifacts/walk_to_grasp_test \
  lift_place kitchen_pick_and_place cracker_box
```

---

## 文件改动总表

| 操作 | 文件 | 说明 |
|------|------|------|
| 修改 | `isaac_replay/build_arm_follow_replay.py` | 新增导航阶段nav_cmd计算+拼接 |
| 新建 | `scripts/13_walk_to_arm_follow_gui.sh` | walk-to版运行脚本 |

不需要改动的文件：
- `isaac_replay/generate_debug_object_traj.py` — 物体轨迹不变
- `isaac_replay/policy_runner_kinematic_object_replay.py` — 已支持replay
- WBC/P-controller — nav_cmd非零时自动触发导航

---

## 关键参考文件

| 文件 | 参考内容 |
|------|---------|
| `bridge/build_replay.py:374-390` | nav_cmd速度计算（世界→base坐标系变换） |
| `bridge/build_replay.py:340-372` | base轨迹规划（offset+yaw计算） |
| `scripts/08_debug_arm_follow_gui.sh` | 现有shell脚本结构 |
| `isaac_replay/build_arm_follow_replay.py` | 23D动作向量构建 |
