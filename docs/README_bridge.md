# HOI + BODex -> G1 (Path A Bridge)

This folder contains a minimal bridge for your selected **Path A**:

- Robot: `G1`
- Controller: `g1_wbc_pink`
- Mode: offline replay
- Grasp: single right hand first, left hand fixed
- Priority: object trajectory fidelity first, then contact realism

Chinese deep-dive docs:
- `docs/README_DEBUG_COMMANDS_CN.md` (headless command handbook)
- `docs/A2_HOIDINI_INTEGRATION_CN.md` (A-2/A-HOI details + HOIDiNi feasibility analysis)

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

## 8. 调试方案 A-2（桌面场景 + 固定 pelvis + 手臂跟随物体）

目标：
- 在非 HOI 轨迹下做“可控基线”，并贴近你要的调试形态：
  - 使用桌面场景
  - 固定 pelvis（base/nav 命令为 0）
  - 只让手臂按给定规则跟随物体运动

已实现内容：
- `isaac_replay/generate_debug_object_traj.py`
  - 生成三种轨迹模式：`line`、`circle`、`lift_place`
  - 可指定/覆盖场景起终点、圆轨迹参数、yaw 旋转
  - 产出标准 `object_kinematic_traj.npz`，可直接被 replay runner 使用
- `isaac_replay/build_arm_follow_replay.py`
  - 从 `object_kinematic_traj.npz` 生成 `replay_actions.hdf5`
  - base/nav 固定为 0（pelvis 不走导航）
  - 右手腕目标由“物体位姿 + object frame 下相对偏移”计算
- `scripts/08_debug_arm_follow_gui.sh`
  - 一条命令完成：轨迹生成 + A-2 replay 生成 + GUI 回放
  - 默认场景是 `kitchen_pick_and_place`（桌面厨房场景）

推荐判定方式：
1. 如果 A-2 下仍然“飞天/穿模”，优先排查 Isaac 资产/场景/坐标系和手腕偏移配置。
2. 如果 A-2 稳定而 HOI 不稳定，优先排查 HOI 世界系对齐与尺度映射。
3. A-2 默认场景环境定义文件：
   - `repos/IsaacLab-Arena/isaaclab_arena/examples/example_environments/kitchen_pick_and_place_environment.py`
   - 已为 G1 显式设置初始位姿 `position_xyz=(0.0, 0.0, 0.0)`，并将物体初始位姿设为 `(0.4, 0.0, 0.1)`，确保机器人位于桌边可操作区域

示例（GUI，非 headless）：

```bash
cd /home/ubuntu/DATA2/workspace/xmh/Humanoid-gen-pack
./scripts/08_debug_arm_follow_gui.sh \
  /home/ubuntu/DATA2/workspace/xmh/Humanoid-gen-pack/artifacts/debug_schemeA2 \
  lift_place kitchen_pick_and_place cracker_box
```

### 8.1 A-HOI 扩展：A 方案直接接 HOIFHI 物体轨迹

目标：
- 保留 A-2 的“固定 pelvis + 手臂跟随物体”控制链路；
- 仅替换物体轨迹来源：从 synthetic 改为 HOIFHI `human_object_results.pkl`；
- 在导入 Isaac 场景前，对 HOI 轨迹做场景尺寸约束重映射。

已实现：
- `scripts/09_debug_arm_follow_from_hoi_headless.sh`
  - `HOI pkl -> 约束轨迹(npz) -> arm-follow replay(hdf5) -> headless replay`
  - 默认场景 `kitchen_pick_and_place`
  - 默认会覆写输出 `object_name` 为你选定的 Isaac 资产名（如 `cracker_box`）

为什么必须约束（以当前 HOI 样例为例）：
- `object_name=smallbox`
- 帧数：`245`
- 原始范围（world）：
  - `x: [-1.464, 0.351]`（跨度约 `1.816m`）
  - `y: [-5.059, -3.320]`（跨度约 `1.739m`）
  - `z: [0.080, 0.933]`（跨度约 `0.853m`）
- 这个尺度直接放进厨房桌面场景会越界，所以 A-HOI 默认对 kitchen 采用：
  - `--traj-scale-xyz "0.13,0.22,0.30"`
  - `--align-first-pos-w "0.40,0.00,0.10"`
  - `--clip-z-min 0.08 --clip-z-max 0.38`
  - `--clip-xy-min "0.05,-0.45" --clip-xy-max "0.65,0.45"`

一键命令（无 GUI，推荐）：

```bash
cd /home/ubuntu/DATA2/workspace/xmh/Humanoid-gen-pack
ISAAC_PYTHON=/home/ubuntu/miniconda3/envs/isaaclab_arena/bin/python \
DEVICE=cpu \
bash scripts/09_debug_arm_follow_from_hoi_headless.sh \
  /home/ubuntu/DATA2/workspace/xmh/Humanoid-gen-pack/artifacts/hoifhli/human_object_results_compare_fine_01_p0_o3.pkl \
  /home/ubuntu/DATA2/workspace/xmh/Humanoid-gen-pack/artifacts/debug_schemeA2_hoi \
  kitchen_pick_and_place \
  cracker_box
```

可调参数（通过环境变量覆盖）：
- `TRJ_SCALE_XYZ`、`TRJ_OFFSET_W`
- `ALIGN_FIRST_POS_W`、`ALIGN_LAST_POS_W`、`ALIGN_LAST_RAMP_SEC`
- `CLIP_Z_MIN`、`CLIP_Z_MAX`、`CLIP_XY_MIN`、`CLIP_XY_MAX`
- `BASE_POS_W`、`MAX_STEPS`、`DEVICE`、`HEADLESS`

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

## 10. 五条命令跑完整个双工作流（GUI）

1. 生成 A-2 的 synthetic 物体轨迹（桌面场景）：

```bash
cd /home/ubuntu/DATA2/workspace/xmh/Humanoid-gen-pack
/home/ubuntu/miniconda3/envs/isaaclab_arena/bin/python isaac_replay/generate_debug_object_traj.py \
  --output artifacts/debug_schemeA2/object_kinematic_traj.npz \
  --output-debug-json artifacts/debug_schemeA2/debug_traj.json \
  --object-name cracker_box \
  --pattern lift_place \
  --scene-preset kitchen_pick_and_place
```

2. 生成 A-2 手臂跟随 replay（固定 pelvis，仅手臂跟随）：

```bash
cd /home/ubuntu/DATA2/workspace/xmh/Humanoid-gen-pack
/home/ubuntu/miniconda3/envs/isaaclab_arena/bin/python isaac_replay/build_arm_follow_replay.py \
  --kin-traj-path artifacts/debug_schemeA2/object_kinematic_traj.npz \
  --output-hdf5 artifacts/debug_schemeA2/replay_actions_arm_follow.hdf5 \
  --output-debug-json artifacts/debug_schemeA2/debug_replay.json \
  --base-pos-w "0.0,0.0,0.0" \
  --base-yaw 0.0 \
  --right-wrist-pos-obj=-0.20,-0.03,0.10 \
  --right-wrist-quat-obj-wxyz=1.0,0.0,0.0,0.0
```

3. 在 Isaac GUI 回放 A-2（桌面场景）：

```bash
cd /home/ubuntu/DATA2/workspace/xmh/Humanoid-gen-pack/repos/IsaacLab-Arena
/home/ubuntu/miniconda3/envs/isaaclab_arena/bin/python /home/ubuntu/DATA2/workspace/xmh/Humanoid-gen-pack/isaac_replay/policy_runner_kinematic_object_replay.py \
  --device cuda:0 --enable_cameras \
  --policy_type replay \
  --replay_file_path /home/ubuntu/DATA2/workspace/xmh/Humanoid-gen-pack/artifacts/debug_schemeA2/replay_actions_arm_follow.hdf5 \
  --episode_name demo_0 \
  --kin-traj-path /home/ubuntu/DATA2/workspace/xmh/Humanoid-gen-pack/artifacts/debug_schemeA2/object_kinematic_traj.npz \
  --kin-asset-name cracker_box \
  --kin-apply-timing pre_step \
  kitchen_pick_and_place \
  --object cracker_box \
  --embodiment g1_wbc_pink
```

4. 从 HOI 构建带约束 replay：

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

5. 在 Isaac GUI 回放带约束 HOI 结果：

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

### 11.1 方案 A-2：桌面场景 + 固定 pelvis + 手臂跟随

入口：
- 脚本入口：`scripts/08_debug_arm_follow_gui.sh`
- 原子入口：
  - `isaac_replay/generate_debug_object_traj.py`
  - `isaac_replay/build_arm_follow_replay.py`
  - `isaac_replay/policy_runner_kinematic_object_replay.py --policy_type replay`

流程：
1. `08_debug_arm_follow_gui.sh`
   - 解析参数：`OUT_DIR`、`PATTERN`、`OBJECT`、`SCENE`
   - 根据 `SCENE` 推导 `SCENE_PRESET`
   - 调 `generate_debug_object_traj.py` 生成 `object_kinematic_traj.npz`
   - 调 `build_arm_follow_replay.py` 生成 `replay_actions_arm_follow.hdf5`
   - 调 `policy_runner_kinematic_object_replay.py` 进行 GUI 回放
2. `generate_debug_object_traj.py`
   - 解析模式和 scene preset
   - 生成 `object_pos_w` + `object_quat_wxyz` + `object_rot_mat_w`
   - 写入 npz（与 bridge 输出格式一致）
   - 可选写 `debug_traj.json`
3. `build_arm_follow_replay.py`
   - 读取 `object_kinematic_traj.npz`
   - 固定 `NAV_CMD=[0,0,0]`，实现 pelvis/base 不走导航
   - 将“右手腕在物体坐标系的相对位姿”映射为 pelvis frame 目标
   - 输出 `replay_actions_arm_follow.hdf5`
4. `policy_runner_kinematic_object_replay.py`
   - 按 replay action 驱动机器人（A-2 中只需要上肢跟随）
   - 每 step 覆写物体根位姿（来自 `object_kinematic_traj.npz`）
   - 场景建议用 `kitchen_pick_and_place`

输入输出：
- 输入：无 HOI 依赖（仅场景 preset + 脚本参数）
- 输出：
  - `artifacts/debug_schemeA2/object_kinematic_traj.npz`
  - `artifacts/debug_schemeA2/replay_actions_arm_follow.hdf5`
  - `artifacts/debug_schemeA2/debug_traj.json`
  - `artifacts/debug_schemeA2/debug_replay.json`

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

方案 A-2（GUI）：
```bash
cd /home/ubuntu/DATA2/workspace/xmh/Humanoid-gen-pack
./scripts/08_debug_arm_follow_gui.sh \
  /home/ubuntu/DATA2/workspace/xmh/Humanoid-gen-pack/artifacts/debug_schemeA2 \
  lift_place kitchen_pick_and_place cracker_box
```

方案 A-HOI（Headless，HOIFHI 轨迹 + 场景约束 + arm-follow）：
```bash
cd /home/ubuntu/DATA2/workspace/xmh/Humanoid-gen-pack
ISAAC_PYTHON=/home/ubuntu/miniconda3/envs/isaaclab_arena/bin/python \
DEVICE=cpu \
./scripts/09_debug_arm_follow_from_hoi_headless.sh \
  /home/ubuntu/DATA2/workspace/xmh/Humanoid-gen-pack/artifacts/hoifhli/human_object_results_compare_fine_01_p0_o3.pkl \
  /home/ubuntu/DATA2/workspace/xmh/Humanoid-gen-pack/artifacts/debug_schemeA2_hoi \
  kitchen_pick_and_place \
  cracker_box
```

方案 B（构建约束 replay）：
```bash
cd /home/ubuntu/DATA2/workspace/xmh/Humanoid-gen-pack
./scripts/07_build_replay_constrained.sh \
  /home/ubuntu/DATA2/workspace/xmh/hoifhli_release/results/interaction/compare_fine_01/10_long_seq_w_waypoints_pidx_0_oidx_3_interaction_guidance/objs_step_10_bs_idx_0_vis_no_scene/human_object_results.pkl \
  /home/ubuntu/DATA2/workspace/xmh/Humanoid-gen-pack/artifacts/g1_bridge_constrained
```

### 12.2 原始命令版（完全显式）

生成 A-2 synthetic 轨迹：
```bash
cd /home/ubuntu/DATA2/workspace/xmh/Humanoid-gen-pack
/home/ubuntu/miniconda3/envs/isaaclab_arena/bin/python isaac_replay/generate_debug_object_traj.py \
  --output artifacts/debug_schemeA2/object_kinematic_traj.npz \
  --output-debug-json artifacts/debug_schemeA2/debug_traj.json \
  --object-name cracker_box \
  --pattern lift_place \
  --scene-preset kitchen_pick_and_place \
  --fps 50 --duration-sec 8.0
```

A-2 replay 生成（固定 pelvis + 手臂跟随）：
```bash
cd /home/ubuntu/DATA2/workspace/xmh/Humanoid-gen-pack
/home/ubuntu/miniconda3/envs/isaaclab_arena/bin/python isaac_replay/build_arm_follow_replay.py \
  --kin-traj-path artifacts/debug_schemeA2/object_kinematic_traj.npz \
  --output-hdf5 artifacts/debug_schemeA2/replay_actions_arm_follow.hdf5 \
  --output-debug-json artifacts/debug_schemeA2/debug_replay.json \
  --base-pos-w "0.0,0.0,0.0" \
  --base-yaw 0.0 \
  --right-wrist-pos-obj=-0.20,-0.03,0.10 \
  --right-wrist-quat-obj-wxyz=1.0,0.0,0.0,0.0
```

A-2 GUI 回放：
```bash
cd /home/ubuntu/DATA2/workspace/xmh/Humanoid-gen-pack/repos/IsaacLab-Arena
/home/ubuntu/miniconda3/envs/isaaclab_arena/bin/python /home/ubuntu/DATA2/workspace/xmh/Humanoid-gen-pack/isaac_replay/policy_runner_kinematic_object_replay.py \
  --device cuda:0 --enable_cameras \
  --policy_type replay \
  --replay_file_path /home/ubuntu/DATA2/workspace/xmh/Humanoid-gen-pack/artifacts/debug_schemeA2/replay_actions_arm_follow.hdf5 \
  --episode_name demo_0 \
  --kin-traj-path /home/ubuntu/DATA2/workspace/xmh/Humanoid-gen-pack/artifacts/debug_schemeA2/object_kinematic_traj.npz \
  --kin-asset-name cracker_box \
  --kin-apply-timing pre_step \
  kitchen_pick_and_place \
  --object cracker_box \
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
