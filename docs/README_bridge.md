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

## 8. Debug Plan A (No HOI Trajectory)

Goal:
- isolate Isaac scene/object/asset scale issues from HOI generation errors.

What is implemented:
- `isaac_replay/generate_debug_object_traj.py`: generate synthetic object trajectory (`line`, `circle`, `lift_place`).
- `scripts/06_debug_scene_object_gui.sh`: one-command GUI replay with `--object-only`.

Why this is useful:
- if object still flies with synthetic trajectory, issue is in scene/object asset/or frame usage.
- if synthetic trajectory is stable, issue is likely in HOI trajectory frame/scale mismatch.

Example (GUI, non-headless):

```bash
cd /home/ubuntu/DATA2/workspace/xmh/Humanoid-gen-pack
./scripts/06_debug_scene_object_gui.sh \
  /home/ubuntu/DATA2/workspace/xmh/Humanoid-gen-pack/artifacts/debug_scheme1 \
  lift_place brown_box galileo_g1_locomanip_pick_and_place
```

## 9. Debug Plan B (Constrain HOI Trajectory To Isaac Scene)

Goal:
- keep using HOI motion, but enforce scene-compatible start/end and spatial bounds.

What is implemented:
- `bridge/build_replay.py` adds trajectory constraints:
  - `--traj-scale-xyz`
  - `--traj-offset-w`
  - `--align-first-pos-w`
  - `--align-last-pos-w`
  - `--align-last-ramp-sec`
  - `--clip-z-min`, `--clip-z-max`
  - `--clip-xy-min`, `--clip-xy-max`
- `scripts/07_build_replay_constrained.sh`: one-command constrained replay build.
- constraint ops and before/after range are recorded under `traj_constraints` in `bridge_debug.json`.

Why this is useful:
- directly addresses object flying caused by world-frame offset, over-high z trajectory, or scene boundary mismatch.
- lower cost than retraining HOI; preserves your current bridge/policy stack.

Example:

```bash
cd /home/ubuntu/DATA2/workspace/xmh/Humanoid-gen-pack
./scripts/07_build_replay_constrained.sh \
  /home/ubuntu/DATA2/workspace/xmh/hoifhli_release/results/interaction/compare_fine_01/10_long_seq_w_waypoints_pidx_0_oidx_3_interaction_guidance/objs_step_10_bs_idx_0_vis_no_scene/human_object_results.pkl \
  /home/ubuntu/DATA2/workspace/xmh/Humanoid-gen-pack/artifacts/g1_bridge_constrained
```

## 10. Four-Command Debug Workflow (GUI)

1) Generate synthetic baseline trajectory:

```bash
cd /home/ubuntu/DATA2/workspace/xmh/Humanoid-gen-pack
/home/ubuntu/miniconda3/envs/isaaclab_arena/bin/python isaac_replay/generate_debug_object_traj.py \
  --output artifacts/debug_scheme1/object_kinematic_traj.npz \
  --output-debug-json artifacts/debug_scheme1/debug_traj.json \
  --object-name brown_box \
  --pattern lift_place \
  --scene-preset galileo_locomanip
```

2) Replay synthetic trajectory in Isaac GUI (`object-only`):

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

3) Build constrained replay from HOI trajectory:

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

4) Replay constrained HOI trajectory in Isaac GUI:

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

## 11. Dual Debug Workflow: Detailed Code Flow

This section explains the code-level execution path for both debug workflows.

### 11.1 Workflow A: Synthetic Trajectory (No HOI)

Entry points:
- Script mode: `scripts/06_debug_scene_object_gui.sh`
- Raw mode:
  - `isaac_replay/generate_debug_object_traj.py`
  - `isaac_replay/policy_runner_kinematic_object_replay.py --object-only`

Execution flow:
1. `scripts/06_debug_scene_object_gui.sh`
   - resolves scene preset from environment name (`galileo_locomanip`, `kitchen_pick_and_place`, or `none`)
   - generates trajectory npz into `OUT_DIR/object_kinematic_traj.npz`
   - launches Isaac replay runner in GUI with `--object-only`
2. `generate_debug_object_traj.py`
   - parses pattern (`line`, `circle`, `lift_place`)
   - resolves start/end from scene preset or overrides
   - generates:
     - `object_pos_w` (T,3)
     - `object_quat_wxyz` (T,4)
     - `object_rot_mat_w` (T,3,3)
   - writes:
     - trajectory npz
     - optional summary json
3. `policy_runner_kinematic_object_replay.py`
   - builds env via IsaacLab-Arena example environment
   - constructs `ObjectKinematicReplayer`
   - each sim step:
     - writes root pose from `object_pos_w/object_quat_wxyz`
     - writes zero velocity
   - because `--object-only`, robot actions are zeros and robot remains still

Output artifacts:
- `artifacts/debug_scheme1/object_kinematic_traj.npz`
- `artifacts/debug_scheme1/debug_traj.json` (if requested)

Use this to answer:
- does object still fly in a controlled trajectory?
- if yes: issue is likely scene/asset/frame side
- if no: HOI trajectory alignment is the likely source

### 11.2 Workflow B: Constrained HOI Trajectory

Entry points:
- Script mode: `scripts/07_build_replay_constrained.sh`
- Raw mode:
  - `bridge/build_replay.py` (with trajectory constraints)
  - `isaac_replay/policy_runner_kinematic_object_replay.py --policy_type replay`

Execution flow:
1. `scripts/07_build_replay_constrained.sh`
   - reads HOI pickle path and output dir
   - applies default constraint set tuned for `galileo_g1_locomanip_pick_and_place`
   - invokes `bridge/build_replay.py`
2. `bridge/build_replay.py`
   - loads HOI trajectory (`obj_pos`, `obj_rot_mat`)
   - resamples to target fps
   - applies trajectory constraints in `_apply_object_traj_constraints`:
     - scaling: `--traj-scale-xyz`
     - translation: `--traj-offset-w`
     - first-frame alignment: `--align-first-pos-w`
     - end-frame alignment (optional): `--align-last-pos-w` + `--align-last-ramp-sec`
     - spatial clipping:
       - `--clip-z-min`, `--clip-z-max`
       - `--clip-xy-min`, `--clip-xy-max`
   - builds replay actions (`[T,23]`) and object trajectory npz
   - records before/after constraint summary in `bridge_debug.json` under `traj_constraints`
3. `policy_runner_kinematic_object_replay.py`
   - replays robot action hdf5
   - overwrites object root pose each step from constrained `object_kinematic_traj.npz`
   - optional runtime HOI mesh conversion via `--use-hoi-object`

Output artifacts:
- `artifacts/g1_bridge_constrained/replay_actions.hdf5`
- `artifacts/g1_bridge_constrained/object_kinematic_traj.npz`
- `artifacts/g1_bridge_constrained/bridge_debug.json`

Use this to answer:
- whether world-frame drift / over-high z / out-of-range xy caused the flying behavior
- which exact constraints were applied (`traj_constraints.ops`)

## 12. Dual Debug Workflow: Command Matrix

### 12.1 One-command wrappers

Workflow A (GUI, synthetic trajectory):
```bash
cd /home/ubuntu/DATA2/workspace/xmh/Humanoid-gen-pack
./scripts/06_debug_scene_object_gui.sh \
  /home/ubuntu/DATA2/workspace/xmh/Humanoid-gen-pack/artifacts/debug_scheme1 \
  lift_place brown_box galileo_g1_locomanip_pick_and_place
```

Workflow B (build constrained replay):
```bash
cd /home/ubuntu/DATA2/workspace/xmh/Humanoid-gen-pack
./scripts/07_build_replay_constrained.sh \
  /home/ubuntu/DATA2/workspace/xmh/hoifhli_release/results/interaction/compare_fine_01/10_long_seq_w_waypoints_pidx_0_oidx_3_interaction_guidance/objs_step_10_bs_idx_0_vis_no_scene/human_object_results.pkl \
  /home/ubuntu/DATA2/workspace/xmh/Humanoid-gen-pack/artifacts/g1_bridge_constrained
```

### 12.2 Raw commands (fully explicit)

Synthetic trajectory generation:
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

Object-only GUI replay:
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

Constrained replay build:
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

GUI replay with constrained HOI trajectory:
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

### 12.3 Quick verification commands

Check constraint summary in debug json:
```bash
python - <<'PY'
import json
p='artifacts/g1_bridge_constrained/bridge_debug.json'
with open(p) as f: d=json.load(f)
print('enabled:', d['traj_constraints']['enabled'])
print('ops:', [op['type'] for op in d['traj_constraints']['ops']])
print('first:', d['traj_constraints']['result_first_w'])
print('z_max:', d['traj_constraints']['result_max_w'][2])
PY
```
