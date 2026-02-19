# Walk-to-Grasp TODO 实际落地说明（2026-02-19）

## 背景

原始 TODO 文档里写的是 CuRobo 8 步实现计划，但仓库中实际已可运行路径是 `build_arm_follow_replay.py --walk-to-grasp`（开环导航 + arm-follow），两者存在脱节。

本次已补齐一条**不影响原始 `08_debug_arm_follow_gui.sh` 调用方式**的新入口，专门用于把 TODO 的 8 步流程落成可执行脚本。

## 已落地文件

- `isaac_replay/g1_curobo_planner.py`
- `scripts/run_walk_to_grasp_todo.py`
- `scripts/13_run_todo_walk_to_grasp.sh`
- `scripts/convert_cedex_to_isaaclab.py`
- `scripts/generate_inspirehand_grasps.py`
- `scripts/generate_inspirehand_grasps.sh`

## 8步流水线（已在代码中实现）

入口：`scripts/run_walk_to_grasp_todo.py`

1. `generate_object_traj`
2. `load_planner_inputs`
3. `plan_walk_to_grasp`
4. `build_replay_actions`
5. `read_replay_metadata`
6. `run_isaac_policy_runner`（可 `--no-runner` 跳过）
7. `validate_artifacts`
8. `write_report`

每步都会写入 `todo_run_report.json`，包含状态、耗时、关键命令与细节。

## 规划器策略（CuRobo / fallback）

- `planner=auto|curobo|open_loop`
- 当前工作区缺少可直接使用的 G1 CuRobo robot/world 全链路配置，`auto` 下会探测 CuRobo import 能力，若无法完成完整规划则**显式回退**为确定性 open-loop 路径。
- `--strict-curobo` 可强制 CuRobo，不满足时直接报错。

## 新命令（不改原 08 流程）

```bash
cd "$PACK_ROOT"
HEADLESS=1 MAX_STEPS=8 \
bash scripts/13_run_todo_walk_to_grasp.sh \
  "$PACK_ROOT/artifacts/todo_walk_to_grasp" \
  kitchen_pick_and_place \
  cracker_box \
  lift_place
```

### 设备选择

- `13_run_todo_walk_to_grasp.sh` 默认 `DEVICE=auto`，会优先选择最空闲 GPU。
- 也可手工指定，例如：`DEVICE=cuda:11`。
- 已禁止 `cpu` 设备（会直接退出），避免误用 CPU。

## 与原命令关系

原命令保持不变（可继续使用）：

```bash
cd "$PACK_ROOT"
DEVICE="$DEVICE" BASE_POS_W="$BASE_POS_W" G1_INIT_YAW_DEG="$G1_INIT_YAW_DEG" \
bash scripts/08_debug_arm_follow_gui.sh \
  "$PACK_ROOT/artifacts/debug_schemeA2_gui" \
  lift_place \
  kitchen_pick_and_place \
  cracker_box
```

本次新增的是 `13` 新入口，不替换你现有 `08` 工作流。

## 验证记录

- `NO_RUNNER=1` 干跑通过：
  - `artifacts/todo_walk_to_grasp_dryrun/todo_run_report.json`
- `HEADLESS=1 MAX_STEPS=8 DEVICE=cuda:11` 实跑通过：
  - `artifacts/todo_walk_to_grasp_headless_gpu11/todo_run_report.json`
  - 报告中 Step 6 为 `ok`，`max_steps=8`。

注：若运行在繁忙卡（如 `cuda:0`）可能触发 PhysX 显存不足并导致 `no active physics scene found`；请使用空闲 GPU。
