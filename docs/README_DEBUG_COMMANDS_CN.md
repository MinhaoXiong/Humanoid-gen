# Dual Debug 工作流命令手册（无 GUI 机器）

本文把你要的命令集中到一个地方：
- 四步 Debug 主链路
- 方案 A（A-2：桌面场景 + 固定 pelvis + 手臂跟随）
- 方案 B（HOI 轨迹约束）

适用前提：本机没有图形化界面，所以统一使用 `--headless`。

如果你在有图形化界面的机器上运行，请使用：
- `docs/README_DEBUG_COMMANDS_GUI_CN.md`

## 0. 环境变量

```bash
export PACK_ROOT=/home/ubuntu/DATA2/workspace/xmh/Humanoid-gen-pack
export ISAAC_PYTHON=/home/ubuntu/miniconda3/envs/isaaclab_arena/bin/python
export DEVICE=cpu
export HOI_PKL=$PACK_ROOT/artifacts/hoifhli/human_object_results_compare_fine_01_p0_o3.pkl
```

## 1. 四步 Debug 主链路（Headless）

### Step 1: 从 HOI pkl 构建基础 replay

```bash
cd "$PACK_ROOT"
bash scripts/03_build_replay.sh \
  "$HOI_PKL" \
  "$PACK_ROOT/artifacts/debug4_run1"
```

产物：
- `artifacts/debug4_run1/replay_actions.hdf5`
- `artifacts/debug4_run1/object_kinematic_traj.npz`
- `artifacts/debug4_run1/bridge_debug.json`

### Step 2: 只看物体轨迹（object-only）

```bash
cd "$PACK_ROOT"
ISAAC_PYTHON="$ISAAC_PYTHON" bash scripts/05_object_only_replay.sh \
  "$PACK_ROOT/artifacts/debug4_run1" \
  debug4_object_only \
  "$PACK_ROOT/artifacts/videos"
```

### Step 3: 机器人 + 物体回放（policy replay）

```bash
cd "$PACK_ROOT"
ISAAC_PYTHON="$ISAAC_PYTHON" bash scripts/04_isaac_replay_video.sh \
  "$PACK_ROOT/artifacts/debug4_run1" \
  debug4_full_replay \
  "$PACK_ROOT/artifacts/videos"
```

### Step 4: 读 debug 元数据做快速判定

```bash
cd "$PACK_ROOT"
python - <<'PY'
import json
p='artifacts/debug4_run1/bridge_debug.json'
with open(p,'r',encoding='utf-8') as f:
    d=json.load(f)
print('stage_indices:', d.get('stage_indices', {}))
print('action_shape:', d.get('outputs', {}).get('action_shape'))
print('object_range_min_w:', d.get('object_world', {}).get('min_w'))
print('object_range_max_w:', d.get('object_world', {}).get('max_w'))
PY
```

## 2. 方案 A（A-2）命令

目标：在桌面场景里固定 pelvis/base，只让手臂按物体轨迹跟随。

### A-2.1 生成 synthetic 物体轨迹

```bash
cd "$PACK_ROOT"
"$ISAAC_PYTHON" isaac_replay/generate_debug_object_traj.py \
  --output "$PACK_ROOT/artifacts/acceptance_a2/object_kinematic_traj.npz" \
  --output-debug-json "$PACK_ROOT/artifacts/acceptance_a2/debug_traj.json" \
  --object-name cracker_box \
  --pattern lift_place \
  --scene-preset kitchen_pick_and_place \
  --fps 50 --duration-sec 8.0
```

### A-2.2 生成 arm-follow replay（固定 pelvis）

```bash
cd "$PACK_ROOT"
"$ISAAC_PYTHON" isaac_replay/build_arm_follow_replay.py \
  --kin-traj-path "$PACK_ROOT/artifacts/acceptance_a2/object_kinematic_traj.npz" \
  --output-hdf5 "$PACK_ROOT/artifacts/acceptance_a2/replay_actions_arm_follow.hdf5" \
  --output-debug-json "$PACK_ROOT/artifacts/acceptance_a2/debug_replay.json" \
  --base-pos-w "0.0,0.0,0.0" \
  --base-yaw 0.0 \
  --right-wrist-pos-obj=-0.20,-0.03,0.10 \
  --right-wrist-quat-obj-wxyz=1.0,0.0,0.0,0.0
```

### A-2.3 Headless 验收回放

```bash
cd "$PACK_ROOT/repos/IsaacLab-Arena"
conda run -n isaaclab_arena python "$PACK_ROOT/isaac_replay/policy_runner_kinematic_object_replay.py" \
  --headless --device "$DEVICE" --enable_cameras \
  --policy_type replay \
  --replay_file_path "$PACK_ROOT/artifacts/acceptance_a2/replay_actions_arm_follow.hdf5" \
  --episode_name demo_0 \
  --kin-traj-path "$PACK_ROOT/artifacts/acceptance_a2/object_kinematic_traj.npz" \
  --kin-asset-name cracker_box \
  --kin-apply-timing pre_step \
  --max-steps 120 \
  kitchen_pick_and_place \
  --object cracker_box \
  --embodiment g1_wbc_pink
```

### A-HOI：把 A 的物体轨迹来源替换为 HOIFHI（带场景约束）

说明：
- 这是 A-2 的扩展链路：控制侧仍是“固定 pelvis + 手臂跟随”；
- 轨迹侧改为 HOIFHI `human_object_results.pkl`；
- 默认会按 `kitchen_pick_and_place` 桌面尺寸做缩放+对齐+裁剪。

```bash
cd "$PACK_ROOT"
ISAAC_PYTHON="$ISAAC_PYTHON" DEVICE="$DEVICE" \
bash scripts/09_debug_arm_follow_from_hoi_headless.sh \
  "$HOI_PKL" \
  "$PACK_ROOT/artifacts/debug_schemeA2_hoi" \
  kitchen_pick_and_place \
  cracker_box
```

关键默认参数（kitchen）：
- `TRJ_SCALE_XYZ="0.13,0.22,0.30"`
- `ALIGN_FIRST_POS_W="0.40,0.00,0.10"`
- `CLIP_Z_MIN=0.08`、`CLIP_Z_MAX=0.38`
- `CLIP_XY_MIN="0.05,-0.45"`、`CLIP_XY_MAX="0.65,0.45"`

可按场景覆盖（示例）：
```bash
cd "$PACK_ROOT"
TRJ_SCALE_XYZ="0.12,0.18,0.28" \
CLIP_XY_MAX="0.62,0.40" \
MAX_STEPS=150 \
bash scripts/09_debug_arm_follow_from_hoi_headless.sh \
  "$HOI_PKL" \
  "$PACK_ROOT/artifacts/debug_schemeA2_hoi_tuned" \
  kitchen_pick_and_place \
  cracker_box
```

## 3. 方案 B 命令（约束 HOI 轨迹）

### B.1 构建 constrained replay

```bash
cd "$PACK_ROOT"
ISAAC_PYTHON="$ISAAC_PYTHON" bash scripts/07_build_replay_constrained.sh \
  "$HOI_PKL" \
  "$PACK_ROOT/artifacts/acceptance_b"
```

产物：
- `artifacts/acceptance_b/replay_actions.hdf5`
- `artifacts/acceptance_b/object_kinematic_traj.npz`
- `artifacts/acceptance_b/bridge_debug.json`

### B.2 优先命令（使用 HOI mesh）

```bash
cd "$PACK_ROOT/repos/IsaacLab-Arena"
conda run -n isaaclab_arena python "$PACK_ROOT/isaac_replay/policy_runner_kinematic_object_replay.py" \
  --headless --device "$DEVICE" --enable_cameras \
  --policy_type replay \
  --replay_file_path "$PACK_ROOT/artifacts/acceptance_b/replay_actions.hdf5" \
  --episode_name demo_0 \
  --kin-traj-path "$PACK_ROOT/artifacts/acceptance_b/object_kinematic_traj.npz" \
  --kin-apply-timing pre_step \
  --use-hoi-object \
  --hoi-root "$PACK_ROOT/repos/hoifhli_release" \
  --hoi-usd-cache-dir "$PACK_ROOT/artifacts/hoi_runtime_usd" \
  --max-steps 120 \
  galileo_g1_locomanip_pick_and_place \
  --embodiment g1_wbc_pink
```

### B.3 如果 HOI mesh 缺失（例如 `smallbox`）的回退命令

```bash
cd "$PACK_ROOT/repos/IsaacLab-Arena"
conda run -n isaaclab_arena python "$PACK_ROOT/isaac_replay/policy_runner_kinematic_object_replay.py" \
  --headless --device "$DEVICE" --enable_cameras \
  --policy_type replay \
  --replay_file_path "$PACK_ROOT/artifacts/acceptance_b/replay_actions.hdf5" \
  --episode_name demo_0 \
  --kin-traj-path "$PACK_ROOT/artifacts/acceptance_b/object_kinematic_traj.npz" \
  --kin-asset-name brown_box \
  --kin-apply-timing pre_step \
  --max-steps 120 \
  galileo_g1_locomanip_pick_and_place \
  --object brown_box \
  --embodiment g1_wbc_pink
```

## 3.5 HOIDiNi -> A-2 厨房链路（新增）

目标：
- 直接使用 HOIDiNi 生成的 `final.pickle`；
- 经过适配器转换成 bridge 输入；
- 在 `kitchen_pick_and_place` 执行 A-2（固定 pelvis + 手臂跟随）。

### 3.5.1 先跑 HOIDiNi 生成（示例：本机 no-DNO 厨房配置）

```bash
cd /home/ubuntu/DATA2/workspace/xmh/HOIDiNi
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
export PYTHONPATH=/home/ubuntu/DATA2/workspace/xmh/HOIDiNi/hoidini:$PYTHONPATH
export TMP_DIR=/home/ubuntu/DATA2/workspace/xmh/Humanoid-gen-pack/artifacts/hoidini_tmp
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

conda run -n hoidini python hoidini/cphoi/cphoi_inference.py \
  --config-path /tmp \
  --config-name sampling_cphoi_kitchen_local_nodno
```

### 3.5.2 一键转换 + A-2 厨房 headless 回放

```bash
cd "$PACK_ROOT"
ISAAC_PYTHON="$ISAAC_PYTHON" DEVICE=cpu HEADLESS=1 MAX_STEPS=60 \
bash scripts/11_debug_arm_follow_from_hoidini_headless.sh \
  "$PACK_ROOT/artifacts/hoidini_kitchen_pickplace_run1/cphoi__cphoi_05011024_c15p100_v0__model000120000__0000__s10_alarmclock_lift_Retake__alarmclock__The_person_is_lifting_a_alarmclock__final.pickle" \
  "$PACK_ROOT/artifacts/acceptance_a2_hoidini" \
  kitchen_pick_and_place \
  cracker_box
```

### 3.5.3 仅做转换（不启动 Isaac）

```bash
cd "$PACK_ROOT"
python3 scripts/10_convert_hoidini_final_to_bridge_pkl.py \
  --hoidini-final-pickle "$PACK_ROOT/artifacts/hoidini_kitchen_pickplace_run1/cphoi__cphoi_05011024_c15p100_v0__model000120000__0000__s10_alarmclock_lift_Retake__alarmclock__The_person_is_lifting_a_alarmclock__final.pickle" \
  --output-pickle "$PACK_ROOT/artifacts/hoidini_kitchen_pickplace_run1/bridge_input_alarmclock.pkl" \
  --output-debug-json "$PACK_ROOT/artifacts/hoidini_kitchen_pickplace_run1/bridge_input_alarmclock_debug.json" \
  --object-name-override cracker_box
```

## 4. 这次已完成的验收记录

- A-2 headless 验收：通过（`max-steps=120`，命令正常结束）。
- B constrained 构建：通过（3 个核心产物已写出）。
- B headless（`--use-hoi-object`）首次失败：`smallbox` mesh 缺失。
- B headless 回退（场景内置 `brown_box`）：通过（`max-steps=120`）。

## 5. 本次顺手修复的两个问题

- 修复 `scripts/08_debug_arm_follow_gui.sh` 里负数参数传递问题：
  - `--right-wrist-pos-obj=-0.20,-0.03,0.10`
- 修复 `ReplayActionPolicy` 调用参数错位：
  - 文件：`repos/IsaacLab-Arena/isaaclab_arena/examples/policy_runner_cli.py`
  - 由位置参数改为关键字参数，避免把 `episode_name` 误传成 `device`。
