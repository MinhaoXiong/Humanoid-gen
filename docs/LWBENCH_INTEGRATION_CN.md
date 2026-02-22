# LW-BenchHub + Spider 场景适配集成

## 概要

将 LW-BenchHub 的 100 个厨房场景（10 layout × 10 style）接入 Humanoid-gen-pack 的 walk-to-grasp 管线，同时支持 Spider SMPL→G1 重定向。不影响原有 `scripts/13_todo_walk_to_grasp_gui.sh` 逻辑，通过独立的 `scripts/15_lwbench_walk_to_grasp.sh` 入口运行。

---

## 1. 数据流

```
SMPL motion ──→ Spider retarget ──→ smpl_to_g1_spider.py ──→ object_kinematic_traj.npz
                                                                      │
LW-BenchHub kitchen ──→ lwbench_scene_adapter.py ──→ scene_config.json │
                                                          │            │
                                                          ▼            ▼
                                                   run_walk_to_grasp_todo.py
                                                          │
                                                          ▼
                                              CuRobo IK + MotionGen
                                                          │
                                                          ▼
                                              replay_actions_arm_follow.hdf5
                                                          │
                                                          ▼
                                              Isaac Sim policy runner
```

---

## 2. 新增文件

### 2.1 `bridge/lwbench_scene_adapter.py`（580 行）

从 LW-BenchHub 厨房场景提取几何信息，生成统一的 scene config JSON。

三种运行模式：

| 模式 | 触发条件 | 说明 |
|------|---------|------|
| SDK 模式 | 默认（`--layout-id` + `--style-id`） | 调用 `KitchenArena` API 提取 USD 几何 |
| Standalone USD | `--local-usd /path/to/kitchen.usd` | 用 OpenUSD `pxr` 直接解析本地 USD |
| Mock 模式 | `--mock` | 生成硬编码厨房参数，用于无 SDK/USD 环境测试 |

输出 JSON 结构：
```json
{
  "scene_info": {
    "scene_name": "lwbench_kitchen_0_0",
    "counter_top_z": 0.85,
    "counter_center_xy": [0.5, 0.0],
    "counter_dims_xy": [0.8, 1.2],
    "robot_init_pos": [-0.55, 0.0, 0.0],
    "collision_cuboids": [...],
    "obstacles_2d": [...]
  },
  "scene_config": {
    "table_z": 0.85,
    "object_align_pos": [0.5, 0.0, 0.85],
    "base_height": 0.75,
    "replay_base_height": 0.78,
    "right_wrist_pos_obj": [-0.18, -0.04, 0.08],
    ...
  },
  "collision": {
    "collision_cuboids": [...],
    "obstacles_2d": [...]
  }
}
```

### 2.2 `bridge/smpl_to_g1_spider.py`（356 行）

将 Spider 输出的 `trajectory_mjwp.npz` 转换为管线所需的 `object_kinematic_traj.npz`。

核心逻辑：
- 从 MuJoCo qpos 中提取物体 free joint（`[x,y,z,qw,qx,qy,qz]`）
- 支持 XML 解析（精确）和启发式提取（取 qpos 末尾 7 个值）
- 对齐到场景台面高度，从源 FPS 重采样到 50 FPS

```bash
python bridge/smpl_to_g1_spider.py \
  --spider-npz trajectory_mjwp.npz \
  --spider-xml scene.xml \
  --output-npz object_kinematic_traj.npz \
  --scene lwbench_kitchen_0_0 \
  --object-name cracker_box
```

### 2.3 `scripts/15_lwbench_walk_to_grasp.sh`（328 行）

独立入口脚本，不修改 `13_todo_walk_to_grasp_gui.sh`。

流程：
1. Step 0：调用 `lwbench_scene_adapter.py` 提取场景配置
2. Step 0.5（可选）：调用 `smpl_to_g1_spider.py` 转换 Spider 轨迹
3. Step 1：调用 `run_walk_to_grasp_todo.py`（通过 `LWBENCH_SCENE_JSON` 环境变量注入场景）

```bash
# 基本用法
bash scripts/15_lwbench_walk_to_grasp.sh artifacts/lwbench_test cracker_box 0 0

# Mock 模式（无需 SDK）
MOCK_SCENE=1 bash scripts/15_lwbench_walk_to_grasp.sh artifacts/lwbench_test cracker_box 0 0

# 带 Spider 输入
SPIDER_NPZ=~/spider_output/trajectory_mjwp.npz \
SPIDER_XML=~/spider_output/scene.xml \
bash scripts/15_lwbench_walk_to_grasp.sh artifacts/lwbench_spider cracker_box 0 0

# 本地 USD
LOCAL_USD=/path/to/kitchen.usd \
bash scripts/15_lwbench_walk_to_grasp.sh artifacts/lwbench_usd cracker_box
```

环境变量：

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `MOCK_SCENE` | `0` | 设为 `1` 启用 mock 模式 |
| `LOCAL_USD` | 空 | 本地 USD 路径，跳过 SDK |
| `SPIDER_NPZ` | 空 | Spider 轨迹 npz 路径 |
| `SPIDER_XML` | 空 | Spider 场景 xml 路径 |
| `HEADLESS` | `0` | 设为 `1` 无头模式 |
| `DEVICE` | `cuda:0` | GPU 设备 |
| `REPLAY_BASE_HEIGHT` | `0.78` | 骨盆高度（米） |
| `MOMAGEN_START_DIST_MIN/MAX` | `0.5/0.8` | 起始位置采样距离范围 |
| `MOMAGEN_TARGET_DIST_MIN/MAX` | `0.25/0.45` | 目标位置采样距离范围 |

### 2.4 `todo/04_lwbenchhub_spider_adaptation.md`（208 行）

设计文档，记录适配方案的架构决策和数据流设计。

---

## 3. 修改的已有文件

### 3.1 `bridge/scene_config.py`（+38 行）

新增两个函数，支持运行时动态注册场景（不修改硬编码的 `SCENES` 字典）：

```python
def register_scene(name: str, config: SceneConfig) -> None:
    """运行时注册一个新场景到 SCENES 字典。"""
    if name not in SCENES:
        SCENES[name] = config

def register_scene_from_json(json_path: str) -> str:
    """从 lwbench_scene_adapter.py 输出的 JSON 加载并注册场景。"""
    ...
```

### 3.2 `isaac_replay/g1_curobo_planner.py`

三处修改：

**a) 动态碰撞注册（+47 行，commit cf49c97）**

```python
_DYNAMIC_OBSTACLES_2D: dict[str, list[...]] = {}
_DYNAMIC_COLLISION_CUBOIDS: dict[str, list[dict]] = {}

def register_scene_collision(scene, obstacles_2d=None, collision_cuboids=None): ...
def register_scene_collision_from_json(json_path): ...
```

`_scene_obstacles()` 和 `_scene_collision_cuboids()` 先查硬编码场景，再查动态注册表。

**b) IK Z-offset 修复（+16/-3 行，未提交）**

Bug：`_wrist_goal_in_base_frame` 收到 `base_pos_w.z = 0`（地面），但 CuRobo 的 `base_link` 是 `torso_link`（骨盆在 Z=0.78m）。导致 IK 求解器以为手腕在骨盆上方 0.93m（实际只有 0.15m）。

修复：
- `PlannerRequest` 新增 `base_height: float = 0.0` 字段
- `_wrist_goal_in_base_frame` 新增 `base_height` 参数，计算前将 `base_pos_w.z += base_height`
- 两个调用点（`_resolve_target_pose_momagen` 和 `plan_walk_to_grasp`）传入 `req.base_height`

**c) IK position_threshold 放宽（10mm → 20mm）**

原来 `position_threshold=0.01` 过于严格。G1 手臂只有 0.41m，14mm 误差对可行性预检查完全可接受。

### 3.3 `scripts/run_walk_to_grasp_todo.py`

两处修改：

**a) 自动注册 LW-BenchHub 场景（+12 行，commit cf49c97）**

在 `main()` 开头检查 `LWBENCH_SCENE_JSON` 环境变量：

```python
lwbench_json = os.environ.get("LWBENCH_SCENE_JSON")
if lwbench_json and os.path.isfile(lwbench_json):
    from bridge.scene_config import register_scene_from_json
    from isaac_replay.g1_curobo_planner import register_scene_collision_from_json
    register_scene_from_json(lwbench_json)
    register_scene_collision_from_json(lwbench_json)
```

**b) 传递 base_height 到 PlannerRequest（+1 行，未提交）**

```python
base_height=float(args.replay_base_height),
```

---

## 4. MoMaGen 采样参数说明

管线使用两阶段采样（类似 MoMaGen）：

| 阶段 | 参数 | 默认值 | 含义 |
|------|------|--------|------|
| Start（起始位置） | `start_dist_min/max` | 0.5~0.8m | 机器人初始站位距物体的距离 |
| Target（目标位置） | `target_dist_min/max` | 0.25~0.45m | 机器人走到的抓取位置距物体的距离 |
| 行走约束 | `min/max_travel_dist` | 0.25~0.90m | 起始到目标的行走距离范围 |

Target 距离设为 0.25~0.45m 是因为 G1 右臂从肩到末端执行器最大伸展只有 0.41m（URDF 实测），加上肩膀在骨盆侧面偏移 0.10m，从骨盆中心算有效臂展约 0.51m。

---

## 5. G1 右臂运动学参数（参考）

来源：`configs/curobo/g1_right_arm.urdf`

```
运动链：torso_link → shoulder_pitch → shoulder_roll → shoulder_yaw → elbow
        → wrist_roll → wrist_pitch → wrist_yaw (EE)

关节数：7 DOF
基座链接：torso_link
末端执行器：right_wrist_yaw_link

肩膀位置（相对骨盆）：x=0.004, y=-0.100, z=0.238
上臂长度（shoulder→elbow）：0.185m
前臂长度（elbow→EE）：      0.184m
总臂展：                     0.41m（理论最大）
```

---

## 6. 修改文件清单

| 文件 | 操作 | 说明 |
|------|------|------|
| `bridge/lwbench_scene_adapter.py` | 新建 | LW-BenchHub 场景提取适配器 |
| `bridge/smpl_to_g1_spider.py` | 新建 | Spider SMPL→物体轨迹转换 |
| `scripts/15_lwbench_walk_to_grasp.sh` | 新建 | LW-BenchHub 管线入口脚本 |
| `todo/04_lwbenchhub_spider_adaptation.md` | 新建 | 设计文档 |
| `bridge/scene_config.py` | 修改 | 动态场景注册 `register_scene()` / `register_scene_from_json()` |
| `isaac_replay/g1_curobo_planner.py` | 修改 | 动态碰撞注册 + IK Z-offset 修复 + threshold 放宽 |
| `scripts/run_walk_to_grasp_todo.py` | 修改 | `LWBENCH_SCENE_JSON` 自动注册 + `base_height` 传递 |
