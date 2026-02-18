# HOI + BODex -> G1 (Path A Bridge)

This folder contains a minimal bridge for your selected **Path A**:

- Robot: `G1`
- Controller: `g1_wbc_pink`
- Mode: offline replay
- Grasp: single right hand first, left hand fixed
- Priority: object trajectory fidelity first, then contact realism

## 1. What is implemented

### `build_replay.py`
Input:
- `hoifhli_release` output: `human_object_results.pkl` (`obj_pos`, `obj_rot_mat`)
- optional `BODex` output: `grasp.npy`

Output:
- replay actions HDF5 for IsaacLab-Arena (`data/demo_0/actions`, shape `[T,23]`)
- object kinematic trajectory (`object_kinematic_traj.npz`)
- debug metadata (`bridge_debug.json`)

### `policy_runner_kinematic_object_replay.py`
An IsaacLab-Arena policy runner variant that:
- replays 23D actions
- writes object root pose each step from `object_kinematic_traj.npz`
- keeps object motion kinematic (not physically pushed/pulled)

This is exactly the stage you asked for: first guarantee object trajectory precision.

## 2. Why "kinematic object replay first"

In this phase, object pose is treated as ground-truth reference and forced every step:
- you isolate retargeting errors (frame mapping, wrist target, timing)
- you avoid physics confounds (friction, solver instability, missed contact)
- once this is stable, you can move to dynamic manipulation and contact-consistent control

## 3. Coordinate assumptions

1. `hoifhli_release` object trajectory is in one world frame.
2. `g1_wbc_pink` action wrist targets are in **pelvis frame**.
3. `build_replay.py` computes a simple base plan, then converts world wrist -> pelvis wrist.
4. If `BODex grasp.npy` is provided:
   - script reads BODex hand root pose (`robot_pose[..., :7]`, assumed `[x,y,z,qw,qx,qy,qz]`)
   - script reads object world pose from `world_cfg["mesh"][...]["pose"]`
   - computes object->hand relative transform and applies it along HOI object motion

If object canonical frames differ between HOI and BODex assets, you must align assets first.

## 4. Command templates

## 4.1 Build replay actions (with BODex grasp)

```bash
cd /home/ubuntu/DATA2/workspace/xmh/IsaacLab-Arena

python tools/hoi_bodex_g1_bridge/build_replay.py \
  --hoi-pickle /home/ubuntu/DATA2/workspace/xmh/hoifhli_release/results/interaction/compare_fine_01/10_long_seq_w_waypoints_pidx_0_oidx_3_interaction_guidance/objs_step_10_bs_idx_0_vis_no_scene/human_object_results.pkl \
  --bodex-grasp-npy /path/to/BODex/grasp.npy \
  --output-hdf5 /tmp/g1_bridge/replay_actions.hdf5 \
  --output-object-traj /tmp/g1_bridge/object_kinematic_traj.npz \
  --output-debug-json /tmp/g1_bridge/bridge_debug.json \
  --hoi-fps 30 \
  --target-fps 50 \
  --yaw-face-object \
  --base-offset-obj-xy "-0.55,0.00"
```

## 4.2 Build replay actions (without BODex yet)

```bash
cd /home/ubuntu/DATA2/workspace/xmh/IsaacLab-Arena

python tools/hoi_bodex_g1_bridge/build_replay.py \
  --hoi-pickle /home/ubuntu/DATA2/workspace/xmh/hoifhli_release/results/interaction/compare_fine_01/10_long_seq_w_waypoints_pidx_0_oidx_3_interaction_guidance/objs_step_10_bs_idx_0_vis_no_scene/human_object_results.pkl \
  --output-hdf5 /tmp/g1_bridge/replay_actions.hdf5 \
  --output-object-traj /tmp/g1_bridge/object_kinematic_traj.npz \
  --output-debug-json /tmp/g1_bridge/bridge_debug.json \
  --pregrasp-pos-obj "-0.35,-0.08,0.10" \
  --grasp-pos-obj "-0.28,-0.05,0.06" \
  --yaw-face-object
```

## 4.3 Replay in Isaac with kinematic object

```bash
cd /home/ubuntu/DATA2/workspace/xmh/IsaacLab-Arena

python isaaclab_arena/examples/policy_runner_kinematic_object_replay.py \
  --policy_type replay \
  --replay_file_path /tmp/g1_bridge/replay_actions.hdf5 \
  --episode_name demo_0 \
  --kin-traj-path /tmp/g1_bridge/object_kinematic_traj.npz \
  --kin-asset-name pick_up_object \
  --kin-apply-timing pre_step \
  --enable_cameras \
  galileo_g1_locomanip_pick_and_place \
  --object brown_box \
  --embodiment g1_wbc_pink
```

## 5. MoMaGen-style segmentation used here

`build_replay.py` writes stage indices in `bridge_debug.json`:
- `navigation`
- `pregrasp`
- `approach`
- `grasp_close`
- `grasp_hold`

This mirrors your preferred workflow pattern: navigation and manipulation are separated.

## Full upstream pipeline doc

For the full chain starting from `hoifhli_release` and `BODex` commands (plus env setup, outputs, and upgrade ideas), see:

- `tools/hoi_bodex_g1_bridge/END_TO_END_FROM_HOI_BODEX_CN.md`

## 6. Current limitations

1. Single env only (`g1_wbc_pink` itself asserts `num_envs == 1`).
2. Single right-hand grasp only in this bridge.
3. Base trajectory retargeting is heuristic (object-relative offset), not full-body optimal control.
4. Kinematic object replay intentionally ignores contact dynamics.

## 7. Upgrade path to your final target

1. Replace heuristic base planner with optimization (object+grasp geometric constraints + joint limits).
2. Add `G1 + inspire hand` mapping and finger trajectory transfer.
3. Remove kinematic forcing and switch to dynamic object interaction with contact-consistent control.
4. Add closed-loop correction (state feedback) on top of replay initialization.

## 8. 调试方案 A（不使用 HOI 轨迹）

目标：
- 先把 Isaac 侧的问题（场景、物体资产、坐标系、尺度）和 HOI 生成问题解耦。
- 用可控的“人工轨迹”验证 `policy_runner_kinematic_object_replay.py` 的回放链路是否稳定。

已实现内容：
- `isaac_replay/generate_debug_object_traj.py`
  - 生成三种轨迹模式：`line`、`circle`、`lift_place`
  - 可指定/覆盖场景起终点、圆轨迹参数、yaw 旋转
  - 产出标准 `object_kinematic_traj.npz`，可直接被 replay runner 使用
- `scripts/06_debug_scene_object_gui.sh`
  - 一条命令完成“轨迹生成 + GUI 回放（`--object-only`）”
  - 自动根据环境名映射 `scene preset`

推荐判定方式：
1. 如果 synthetic 轨迹也“飞天/穿模”，优先排查 Isaac 侧：资产尺寸、初始位姿、场景碰撞、物体参考系。
2. 如果 synthetic 轨迹稳定，而 HOI 轨迹不稳定，问题大概率在 HOI 轨迹的世界系对齐/尺度映射。

示例（GUI，非 headless）：

```bash
cd /home/ubuntu/DATA2/workspace/xmh/Humanoid-gen-pack
./scripts/06_debug_scene_object_gui.sh \
  /home/ubuntu/DATA2/workspace/xmh/Humanoid-gen-pack/artifacts/debug_scheme1 \
  lift_place brown_box galileo_g1_locomanip_pick_and_place
```

## 9. 调试方案 B（对 HOI 轨迹加场景约束）

目标：
- 保留 HOI 生成轨迹，但在 bridge 阶段加入“场景一致性约束”，降低飞天/漂移。

已实现内容：
- `bridge/build_replay.py` 新增轨迹约束参数：
  - 尺度：`--traj-scale-xyz`
  - 平移：`--traj-offset-w`
  - 起点对齐：`--align-first-pos-w`
  - 终点对齐：`--align-last-pos-w` + `--align-last-ramp-sec`
  - 范围裁剪：`--clip-z-min`、`--clip-z-max`、`--clip-xy-min`、`--clip-xy-max`
- `scripts/07_build_replay_constrained.sh`
  - 一条命令构建带约束的 replay 产物
  - 默认参数针对 `galileo_g1_locomanip_pick_and_place`
- `bridge_debug.json` 写入 `traj_constraints`
  - 包含操作列表、源轨迹范围、约束后范围，便于定量排查

约束执行顺序（代码真实顺序）：
1. `scale_xyz`（绕首帧锚点缩放）
2. `offset_w`（整体平移）
3. `align_first_pos_w`（对齐首帧到场景目标点）
4. `align_last_pos_w`（可选，支持尾段 ramp 对齐）
5. `clip_z/clip_xy`（硬边界裁剪）

示例：

```bash
cd /home/ubuntu/DATA2/workspace/xmh/Humanoid-gen-pack
./scripts/07_build_replay_constrained.sh \
  /home/ubuntu/DATA2/workspace/xmh/hoifhli_release/results/interaction/compare_fine_01/10_long_seq_w_waypoints_pidx_0_oidx_3_interaction_guidance/objs_step_10_bs_idx_0_vis_no_scene/human_object_results.pkl \
  /home/ubuntu/DATA2/workspace/xmh/Humanoid-gen-pack/artifacts/g1_bridge_constrained
```

## 10. 四条命令跑完整个双工作流（GUI）

1. 生成 synthetic 基线轨迹：

```bash
cd /home/ubuntu/DATA2/workspace/xmh/Humanoid-gen-pack
/home/ubuntu/miniconda3/envs/isaaclab_arena/bin/python isaac_replay/generate_debug_object_traj.py \
  --output artifacts/debug_scheme1/object_kinematic_traj.npz \
  --output-debug-json artifacts/debug_scheme1/debug_traj.json \
  --object-name brown_box \
  --pattern lift_place \
  --scene-preset galileo_locomanip
```

2. 在 Isaac GUI 回放 synthetic（`object-only`）：

```bash
cd /home/ubuntu/DATA2/workspace/xmh/Humanoid-gen-pack/repos/IsaacLab-Arena
/home/ubuntu/miniconda3/envs/isaaclab_arena/bin/python /home/ubuntu/DATA2/workspace/xmh/Humanoid-gen-pack/isaac_replay/policy_runner_kinematic_object_replay.py \
  --device cuda:0 --enable_cameras \
  --object-only \
  --kin-traj-path /home/ubuntu/DATA2/workspace/xmh/Humanoid-gen-pack/artifacts/debug_scheme1/object_kinematic_traj.npz \
  --kin-asset-name brown_box \
  --kin-apply-timing pre_step \
  galileo_g1_locomanip_pick_and_place \
  --object brown_box \
  --embodiment g1_wbc_pink
```

3. 从 HOI 构建带约束 replay：

```bash
cd /home/ubuntu/DATA2/workspace/xmh/Humanoid-gen-pack
/home/ubuntu/miniconda3/envs/isaaclab_arena/bin/python bridge/build_replay.py \
  --hoi-pickle /home/ubuntu/DATA2/workspace/xmh/hoifhli_release/results/interaction/compare_fine_01/10_long_seq_w_waypoints_pidx_0_oidx_3_interaction_guidance/objs_step_10_bs_idx_0_vis_no_scene/human_object_results.pkl \
  --output-hdf5 artifacts/g1_bridge_constrained/replay_actions.hdf5 \
  --output-object-traj artifacts/g1_bridge_constrained/object_kinematic_traj.npz \
  --output-debug-json artifacts/g1_bridge_constrained/bridge_debug.json \
  --hoi-fps 30 --target-fps 50 --yaw-face-object \
  --align-first-pos-w "0.5785,0.18,0.0707" \
  --clip-z-min 0.06 --clip-z-max 0.40
```

4. 在 Isaac GUI 回放带约束 HOI 结果：

```bash
cd /home/ubuntu/DATA2/workspace/xmh/Humanoid-gen-pack/repos/IsaacLab-Arena
/home/ubuntu/miniconda3/envs/isaaclab_arena/bin/python /home/ubuntu/DATA2/workspace/xmh/Humanoid-gen-pack/isaac_replay/policy_runner_kinematic_object_replay.py \
  --device cuda:0 --enable_cameras \
  --policy_type replay \
  --replay_file_path /home/ubuntu/DATA2/workspace/xmh/Humanoid-gen-pack/artifacts/g1_bridge_constrained/replay_actions.hdf5 \
  --episode_name demo_0 \
  --kin-traj-path /home/ubuntu/DATA2/workspace/xmh/Humanoid-gen-pack/artifacts/g1_bridge_constrained/object_kinematic_traj.npz \
  --kin-apply-timing pre_step \
  --use-hoi-object \
  --hoi-root /home/ubuntu/DATA2/workspace/xmh/Humanoid-gen-pack/repos/hoifhli_release \
  --hoi-usd-cache-dir /home/ubuntu/DATA2/workspace/xmh/Humanoid-gen-pack/artifacts/hoi_runtime_usd \
  --max-steps 408 \
  galileo_g1_locomanip_pick_and_place \
  --embodiment g1_wbc_pink
```

## 11. 双工作流代码级详细流程（函数/文件粒度）

### 11.1 方案 A：Synthetic 轨迹链路

入口：
- 脚本入口：`scripts/06_debug_scene_object_gui.sh`
- 原子入口：
  - `isaac_replay/generate_debug_object_traj.py`
  - `isaac_replay/policy_runner_kinematic_object_replay.py --object-only`

流程：
1. `06_debug_scene_object_gui.sh`
   - 解析参数：`OUT_DIR`、`PATTERN`、`OBJECT`、`SCENE`
   - 根据 `SCENE` 推导 `SCENE_PRESET`
   - 调 `generate_debug_object_traj.py` 生成 `object_kinematic_traj.npz`
   - 调 `policy_runner_kinematic_object_replay.py` 进行 GUI 回放
2. `generate_debug_object_traj.py`
   - 解析模式和 scene preset
   - 生成 `object_pos_w` + `object_quat_wxyz` + `object_rot_mat_w`
   - 写入 npz（与 bridge 输出格式一致）
   - 可选写 `debug_traj.json`
3. `policy_runner_kinematic_object_replay.py`
   - 创建 `ObjectKinematicReplayer`
   - 每 step 直接写物体根位姿与零速度
   - `--object-only` 时，机器人 action 为全零，机器人不参与动作控制

输入输出：
- 输入：无 HOI 依赖（仅场景 preset + 脚本参数）
- 输出：
  - `artifacts/debug_scheme1/object_kinematic_traj.npz`
  - `artifacts/debug_scheme1/debug_traj.json`

### 11.2 方案 B：带约束 HOI 轨迹链路

入口：
- 脚本入口：`scripts/07_build_replay_constrained.sh`
- 原子入口：
  - `bridge/build_replay.py`
  - `isaac_replay/policy_runner_kinematic_object_replay.py --policy_type replay`

流程：
1. `07_build_replay_constrained.sh`
   - 读取 HOI pkl 和输出目录
   - 拼装约束参数（含默认值）
   - 调 `bridge/build_replay.py`
2. `build_replay.py`
   - 加载 HOI `obj_pos/obj_rot_mat`
   - 重采样到目标 fps
   - 在 `_apply_object_traj_constraints` 中按顺序做约束
   - 生成：
     - `replay_actions.hdf5`
     - `object_kinematic_traj.npz`
     - `bridge_debug.json`（含 `traj_constraints`）
3. `policy_runner_kinematic_object_replay.py`
   - 回放 action policy（replay）
   - 每 step 覆写物体根位姿（来自约束后的 npz）
   - 可选 `--use-hoi-object` 将 HOI mesh 转成运行时 USD

输入输出：
- 输入：HOI `human_object_results.pkl`
- 输出：
  - `artifacts/g1_bridge_constrained/replay_actions.hdf5`
  - `artifacts/g1_bridge_constrained/object_kinematic_traj.npz`
  - `artifacts/g1_bridge_constrained/bridge_debug.json`

## 12. 命令矩阵（脚本版 + 原始命令版）

### 12.1 脚本一键版

方案 A（GUI）：
```bash
cd /home/ubuntu/DATA2/workspace/xmh/Humanoid-gen-pack
./scripts/06_debug_scene_object_gui.sh \
  /home/ubuntu/DATA2/workspace/xmh/Humanoid-gen-pack/artifacts/debug_scheme1 \
  lift_place brown_box galileo_g1_locomanip_pick_and_place
```

方案 B（构建约束 replay）：
```bash
cd /home/ubuntu/DATA2/workspace/xmh/Humanoid-gen-pack
./scripts/07_build_replay_constrained.sh \
  /home/ubuntu/DATA2/workspace/xmh/hoifhli_release/results/interaction/compare_fine_01/10_long_seq_w_waypoints_pidx_0_oidx_3_interaction_guidance/objs_step_10_bs_idx_0_vis_no_scene/human_object_results.pkl \
  /home/ubuntu/DATA2/workspace/xmh/Humanoid-gen-pack/artifacts/g1_bridge_constrained
```

### 12.2 原始命令版（完全显式）

生成 synthetic 轨迹：
```bash
cd /home/ubuntu/DATA2/workspace/xmh/Humanoid-gen-pack
/home/ubuntu/miniconda3/envs/isaaclab_arena/bin/python isaac_replay/generate_debug_object_traj.py \
  --output artifacts/debug_scheme1/object_kinematic_traj.npz \
  --output-debug-json artifacts/debug_scheme1/debug_traj.json \
  --object-name brown_box \
  --pattern lift_place \
  --scene-preset galileo_locomanip \
  --fps 50 --duration-sec 8.0
```

GUI 回放 synthetic：
```bash
cd /home/ubuntu/DATA2/workspace/xmh/Humanoid-gen-pack/repos/IsaacLab-Arena
/home/ubuntu/miniconda3/envs/isaaclab_arena/bin/python /home/ubuntu/DATA2/workspace/xmh/Humanoid-gen-pack/isaac_replay/policy_runner_kinematic_object_replay.py \
  --device cuda:0 --enable_cameras \
  --object-only \
  --kin-traj-path /home/ubuntu/DATA2/workspace/xmh/Humanoid-gen-pack/artifacts/debug_scheme1/object_kinematic_traj.npz \
  --kin-asset-name brown_box \
  --kin-apply-timing pre_step \
  galileo_g1_locomanip_pick_and_place \
  --object brown_box \
  --embodiment g1_wbc_pink
```

构建带约束 replay：
```bash
cd /home/ubuntu/DATA2/workspace/xmh/Humanoid-gen-pack
/home/ubuntu/miniconda3/envs/isaaclab_arena/bin/python bridge/build_replay.py \
  --hoi-pickle /home/ubuntu/DATA2/workspace/xmh/hoifhli_release/results/interaction/compare_fine_01/10_long_seq_w_waypoints_pidx_0_oidx_3_interaction_guidance/objs_step_10_bs_idx_0_vis_no_scene/human_object_results.pkl \
  --output-hdf5 artifacts/g1_bridge_constrained/replay_actions.hdf5 \
  --output-object-traj artifacts/g1_bridge_constrained/object_kinematic_traj.npz \
  --output-debug-json artifacts/g1_bridge_constrained/bridge_debug.json \
  --hoi-fps 30 --target-fps 50 --yaw-face-object \
  --traj-scale-xyz "1.0,1.0,1.0" \
  --traj-offset-w "0.0,0.0,0.0" \
  --align-first-pos-w "0.5785,0.18,0.0707" \
  --align-last-ramp-sec 1.0 \
  --clip-z-min 0.06 --clip-z-max 0.40
```

GUI 回放带约束 HOI：
```bash
cd /home/ubuntu/DATA2/workspace/xmh/Humanoid-gen-pack/repos/IsaacLab-Arena
/home/ubuntu/miniconda3/envs/isaaclab_arena/bin/python /home/ubuntu/DATA2/workspace/xmh/Humanoid-gen-pack/isaac_replay/policy_runner_kinematic_object_replay.py \
  --device cuda:0 --enable_cameras \
  --policy_type replay \
  --replay_file_path /home/ubuntu/DATA2/workspace/xmh/Humanoid-gen-pack/artifacts/g1_bridge_constrained/replay_actions.hdf5 \
  --episode_name demo_0 \
  --kin-traj-path /home/ubuntu/DATA2/workspace/xmh/Humanoid-gen-pack/artifacts/g1_bridge_constrained/object_kinematic_traj.npz \
  --kin-apply-timing pre_step \
  --use-hoi-object \
  --hoi-root /home/ubuntu/DATA2/workspace/xmh/Humanoid-gen-pack/repos/hoifhli_release \
  --hoi-usd-cache-dir /home/ubuntu/DATA2/workspace/xmh/Humanoid-gen-pack/artifacts/hoi_runtime_usd \
  --max-steps 408 \
  galileo_g1_locomanip_pick_and_place \
  --embodiment g1_wbc_pink
```

### 12.3 快速核验命令

检查约束是否生效（读取 `traj_constraints`）：
```bash
python - <<'PY'
import json
p='artifacts/g1_bridge_constrained/bridge_debug.json'
with open(p) as f:
    d=json.load(f)
print('enabled:', d['traj_constraints']['enabled'])
print('ops:', [op['type'] for op in d['traj_constraints']['ops']])
print('first:', d['traj_constraints']['result_first_w'])
print('z_max:', d['traj_constraints']['result_max_w'][2])
PY
```
