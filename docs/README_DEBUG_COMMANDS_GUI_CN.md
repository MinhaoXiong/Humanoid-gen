# Dual Debug 工作流命令手册（有 GUI 机器）

本文是给“有图形界面”的机器使用的命令手册。

适用范围：
- 你在有显示器/X11/桌面会话的机器上跑 Isaac。
- 你希望直接看 GUI 回放，不走 `--headless`。

如果你在无 GUI 机器运行，请改用：`docs/README_DEBUG_COMMANDS_CN.md`。

## 0. 环境变量

```bash
export PACK_ROOT=/home/ubuntu/DATA2/workspace/xmh/Humanoid-gen-pack
export ISAAC_PYTHON=/home/ubuntu/miniconda3/envs/isaaclab_arena/bin/python
export DEVICE=cuda:0
export HOI_PKL=$PACK_ROOT/artifacts/hoifhli/human_object_results_compare_fine_01_p0_o3.pkl
export HOI_ROOT=$PACK_ROOT/assets/hoifhli_release_min
export BASE_POS_W=0.05,0.0,0.0
export G1_INIT_YAW_DEG=0.0

# 已经推送好的 HOIDiNi 结果（另一台机器无 HOIDiNi 环境时直接用）
export HOIDINI_FINAL_PKL=$PACK_ROOT/artifacts/hoidini_kitchen_pickplace_run1/cphoi__cphoi_05011024_c15p100_v0__model000120000__0000__s10_alarmclock_lift_Retake__alarmclock__The_person_is_lifting_a_alarmclock__final.pickle
```

## 1. 四步 Debug 主链路（GUI）

### Step 1: 从 HOI pkl 构建基础 replay

```bash
cd "$PACK_ROOT"
bash scripts/03_build_replay.sh \
  "$HOI_PKL" \
  "$PACK_ROOT/artifacts/debug4_gui"
```

### Step 2: 物体轨迹 GUI 预览（object-only）

```bash
cd "$PACK_ROOT/repos/IsaacLab-Arena"
"$ISAAC_PYTHON" "$PACK_ROOT/isaac_replay/policy_runner_kinematic_object_replay.py" \
  --device "$DEVICE" --enable_cameras \
  --object-only \
  --kin-traj-path "$PACK_ROOT/artifacts/debug4_gui/object_kinematic_traj.npz" \
  --kin-apply-timing pre_step \
  --use-hoi-object \
  --hoi-root "$HOI_ROOT" \
  --hoi-usd-cache-dir "$PACK_ROOT/artifacts/hoi_runtime_usd" \
  galileo_g1_locomanip_pick_and_place \
  --embodiment g1_wbc_pink
```

### Step 3: 机器人 + 物体 GUI 回放（policy replay）

```bash
cd "$PACK_ROOT/repos/IsaacLab-Arena"
"$ISAAC_PYTHON" "$PACK_ROOT/isaac_replay/policy_runner_kinematic_object_replay.py" \
  --device "$DEVICE" --enable_cameras \
  --policy_type replay \
  --replay_file_path "$PACK_ROOT/artifacts/debug4_gui/replay_actions.hdf5" \
  --episode_name demo_0 \
  --kin-traj-path "$PACK_ROOT/artifacts/debug4_gui/object_kinematic_traj.npz" \
  --kin-apply-timing pre_step \
  --use-hoi-object \
  --hoi-root "$HOI_ROOT" \
  --hoi-usd-cache-dir "$PACK_ROOT/artifacts/hoi_runtime_usd" \
  --max-steps 408 \
  galileo_g1_locomanip_pick_and_place \
  --embodiment g1_wbc_pink
```

### Step 4: 读 debug 元数据

```bash
cd "$PACK_ROOT"
python3 - <<'PY'
import json
p='artifacts/debug4_gui/bridge_debug.json'
with open(p,'r',encoding='utf-8') as f:
    d=json.load(f)
print('action_shape:', d.get('outputs', {}).get('action_shape'))
print('traj_constraints:', d.get('traj_constraints', {}).get('enabled'))
print('range_min:', d.get('traj_constraints', {}).get('result_min_w'))
print('range_max:', d.get('traj_constraints', {}).get('result_max_w'))
PY
```

## 2. 方案 A（A-2）GUI

### A-2.1 一键 synthetic 轨迹 + arm-follow + GUI 回放

```bash
cd "$PACK_ROOT"
DEVICE="$DEVICE" BASE_POS_W="$BASE_POS_W" G1_INIT_YAW_DEG="$G1_INIT_YAW_DEG" \
bash scripts/08_debug_arm_follow_gui.sh \
  "$PACK_ROOT/artifacts/debug_schemeA2_gui" \
  lift_place \
  kitchen_pick_and_place \
  cracker_box
```

### A-2.2 A-HOI（HOIFHI 轨迹）GUI 回放

```bash
cd "$PACK_ROOT"
ISAAC_PYTHON="$ISAAC_PYTHON" DEVICE="$DEVICE" HEADLESS=0 \
BASE_POS_W="$BASE_POS_W" G1_INIT_YAW_DEG="$G1_INIT_YAW_DEG" \
bash scripts/09_debug_arm_follow_from_hoi_headless.sh \
  "$HOI_PKL" \
  "$PACK_ROOT/artifacts/debug_schemeA2_hoi_gui" \
  kitchen_pick_and_place \
  cracker_box
```

## 3. 方案 B（约束 HOI 轨迹）GUI

### B.1 构建 constrained replay

```bash
cd "$PACK_ROOT"
ISAAC_PYTHON="$ISAAC_PYTHON" bash scripts/07_build_replay_constrained.sh \
  "$HOI_PKL" \
  "$PACK_ROOT/artifacts/acceptance_b_gui"
```

### B.2 GUI 回放（优先 HOI mesh）

```bash
cd "$PACK_ROOT/repos/IsaacLab-Arena"
"$ISAAC_PYTHON" "$PACK_ROOT/isaac_replay/policy_runner_kinematic_object_replay.py" \
  --device "$DEVICE" --enable_cameras \
  --policy_type replay \
  --replay_file_path "$PACK_ROOT/artifacts/acceptance_b_gui/replay_actions.hdf5" \
  --episode_name demo_0 \
  --kin-traj-path "$PACK_ROOT/artifacts/acceptance_b_gui/object_kinematic_traj.npz" \
  --kin-apply-timing pre_step \
  --use-hoi-object \
  --hoi-root "$HOI_ROOT" \
  --hoi-usd-cache-dir "$PACK_ROOT/artifacts/hoi_runtime_usd" \
  --max-steps 120 \
  galileo_g1_locomanip_pick_and_place \
  --embodiment g1_wbc_pink
```

## 4. HOIDiNi 结果接入（GUI 机器，无需本地 HOIDiNi 环境）

你这台 GUI 机器如果没有 `hoidini` conda 环境，可以直接用已推送的 `final.pickle`。

### 4.1 一键命令（推荐）

```bash
cd "$PACK_ROOT"
DEVICE="$DEVICE" HOI_FPS=20 MAX_STEPS=120 \
bash scripts/12_debug_arm_follow_from_hoidini_gui.sh \
  "$HOIDINI_FINAL_PKL" \
  "$PACK_ROOT/artifacts/debug_schemeA2_hoidini_gui" \
  kitchen_pick_and_place \
  cracker_box
```

### 4.2 分步命令（先转格式，再 GUI 回放）

```bash
cd "$PACK_ROOT"
python3 scripts/10_convert_hoidini_final_to_bridge_pkl.py \
  --hoidini-final-pickle "$HOIDINI_FINAL_PKL" \
  --output-pickle "$PACK_ROOT/artifacts/debug_schemeA2_hoidini_gui/hoidini_bridge_input.pkl" \
  --output-debug-json "$PACK_ROOT/artifacts/debug_schemeA2_hoidini_gui/hoidini_convert_debug.json" \
  --object-name-override cracker_box

ISAAC_PYTHON="$ISAAC_PYTHON" DEVICE="$DEVICE" HEADLESS=0 HOI_FPS=20 \
bash scripts/09_debug_arm_follow_from_hoi_headless.sh \
  "$PACK_ROOT/artifacts/debug_schemeA2_hoidini_gui/hoidini_bridge_input.pkl" \
  "$PACK_ROOT/artifacts/debug_schemeA2_hoidini_gui" \
  kitchen_pick_and_place \
  cracker_box
```

## 5. 常见问题

- 黑屏或无法创建窗口：检查当前会话是否有可用显示（`echo $DISPLAY`）。
- GPU 不可用：先试 `DEVICE=cpu` 验证链路，再切回 `cuda:0`。
- 物体飞天：优先看 `bridge_debug.json` 的 `result_min_w/result_max_w` 是否超出桌面范围。
- 机器人离桌子太远：调 `BASE_POS_W`（并同步 `G1_INIT_YAW_DEG`），推荐先试：
  - `BASE_POS_W=0.05,0.0,0.0`
  - `BASE_POS_W=0.0,-0.15,0.0`
