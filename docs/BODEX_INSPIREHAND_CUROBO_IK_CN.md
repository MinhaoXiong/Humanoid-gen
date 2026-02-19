# BODex InspireHand 抓取 & CuRobo IK/MotionGen 使用手册

## 0. 新机器部署

另一台机器 clone Humanoid-gen-pack 后，需要：

### 0.1 克隆 BODex 并部署 InspireHand 资产

```bash
# 1. 克隆 BODex（和 Humanoid-gen-pack 同级目录）
cd /path/to/workspace
git clone https://github.com/JYChen18/BODex.git

# 2. 一键部署 InspireHand 资产到 BODex
cd Humanoid-gen-pack
bash scripts/setup_bodex_inspirehand.sh ../BODex
```

### 0.2 安装 CuRobo 依赖

```bash
# 在 isaaclab_arena conda 环境中安装
bash scripts/setup_curobo_deps.sh isaaclab_arena
```

### 0.3 环境变量

```bash
export PACK_ROOT=/path/to/Humanoid-gen-pack
export ISAAC_PYTHON=/path/to/miniconda3/envs/isaaclab_arena/bin/python
export DEVICE=cuda:0
```

## 1. BODex InspireHand 抓取生成

对指定物体 mesh 生成 InspireHand 抓取位姿（wrist 7D + 12 关节角），输出 `.pt` 文件：

```bash
cd "$PACK_ROOT"
python3 scripts/generate_bodex_inspirehand_grasps.py \
  --mesh-file /path/to/cracker_box.obj \
  --top-k 16
```

或使用 shell wrapper：

```bash
bash scripts/generate_bodex_inspirehand_grasps.sh /path/to/cracker_box.obj
```

输出位于 `artifacts/bodex_inspire_grasps/<object>_top16_<timestamp>.pt`。

## 2. TODO 管道（含 CuRobo IK 验证）

```bash
cd "$PACK_ROOT"
python3 scripts/run_walk_to_grasp_todo.py \
  --out-dir "$PACK_ROOT/artifacts/todo_kitchen_test" \
  --scene kitchen_pick_and_place \
  --object cracker_box \
  --pattern lift_place \
  --planner auto \
  --no-runner
```

去掉 `--no-runner` 可直接启动 Isaac Sim 回放。

查看 IK 验证结果：

```bash
python3 -c "
import json
r = json.load(open('$PACK_ROOT/artifacts/todo_kitchen_test/todo_run_report.json'))
print('IK result:', json.dumps(r['planner'].get('ik_result'), indent=2))
print('Notes:', r['planner']['notes'])
"
```

## 3. TODO 管道 vs 08_debug_arm_follow_gui.sh 的区别

`08_debug_arm_follow_gui.sh` 是早期的"方案 A-2"调试脚本，`run_walk_to_grasp_todo.py` 是新的 TODO 管道。两者都能完成"生成轨迹 → 构建 replay → Isaac 回放"的流程，但设计目标和能力不同。

### 对比命令

```bash
# 旧：08 脚本（shell 串联，无 IK 验证）
cd "$PACK_ROOT"
DEVICE="$DEVICE" BASE_POS_W="$BASE_POS_W" G1_INIT_YAW_DEG="$G1_INIT_YAW_DEG" \
bash scripts/08_debug_arm_follow_gui.sh \
  "$PACK_ROOT/artifacts/debug_schemeA2_gui" \
  lift_place \
  kitchen_pick_and_place \
  cracker_box

# 新：TODO 管道（Python 统一调度，含 IK 验证）
cd "$PACK_ROOT"
python3 scripts/run_walk_to_grasp_todo.py \
  --out-dir "$PACK_ROOT/artifacts/todo_kitchen_test" \
  --scene kitchen_pick_and_place \
  --object cracker_box \
  --pattern lift_place \
  --planner auto
```

### 核心差异

| 维度 | `08_debug_arm_follow_gui.sh` | `run_walk_to_grasp_todo.py` |
|------|------------------------------|----------------------------|
| 调度方式 | bash 脚本串联 3 个 Python 子进程 | 单个 Python 进程统一调度 8 步 |
| IK 验证 | 无 | Step 3 自动调用 CuRobo IKSolver 验证 wrist pose 可达性 |
| 路径规划 | 直接传参给 `build_arm_follow_replay.py`，由它内部处理 | 独立的 `g1_curobo_planner.py` 模块，支持 CuRobo/open_loop 切换 |
| 碰撞检测 | 无 | IK 求解时加载场景 3D 障碍物（kitchen 桌子等） |
| 错误处理 | `set -euo pipefail`，任一步失败即退出 | 每步 try/except，失败记录到 report 继续执行 |
| 输出报告 | 无统一报告，需手动查看各 debug json | 自动生成 `todo_run_report.json`，含每步状态、耗时、IK 结果 |
| 参数传递 | 环境变量（`DEVICE`, `BASE_POS_W` 等） | 命令行参数（`--planner`, `--scene` 等） |
| 可跳过仿真 | 不支持 | `--no-runner` 可只生成 replay 不启动 Isaac |

### 什么时候用哪个

- **快速 GUI 调试**：用 `08_debug_arm_follow_gui.sh`，环境变量配好一行命令跑完，直接看画面。
- **自动化验证 / CI**：用 `run_walk_to_grasp_todo.py`，有结构化 report，IK 可达性预检查，适合批量跑和自动判断结果。
- **需要 IK 预检查**：只有 TODO 管道支持。当 `--planner auto` 且 CuRobo 可用时，会在 Step 3 自动验证目标 wrist pose 是否在 G1 右臂工作空间内，report 中会输出 `ik_reachable: true/false`。

## 4. CuRobo MotionGen 轨迹规划

当 IK 检查通过后，管道自动调用 CuRobo MotionGen 规划 rest → target wrist pose 的无碰撞轨迹。

流程：`BODex 抓取位姿 → IK 可达性 → MotionGen 轨迹 → 平滑过渡插入 replay`

- MotionGen 成功：生成平滑关节轨迹，插入到 walk hold 和 arm-follow 之间
- MotionGen 失败：自动 fallback 到开环（和之前一样的瞬移），不阻塞管道

查看 MotionGen 结果：

```bash
python3 -c "
import json
r = json.load(open('artifacts/todo_kitchen_test/todo_run_report.json'))
p = r['planner']
print('Planner used:', p['planner_used'])
mg = p.get('motion_gen_result')
if mg:
    print('MotionGen success:', mg['success'])
    print('MotionGen steps:', mg.get('num_steps'))
"
```

## 5. 一键 GUI 运行

```bash
cd "$PACK_ROOT"
bash scripts/13_todo_walk_to_grasp_gui.sh
```

自定义参数：

```bash
DEVICE=cuda:0 PLANNER=auto \
bash scripts/13_todo_walk_to_grasp_gui.sh \
  artifacts/todo_curobo_gui \
  lift_place \
  kitchen_pick_and_place \
  cracker_box
```
