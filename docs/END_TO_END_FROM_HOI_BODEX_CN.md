# HOIFHLI + BODex + IsaacLab-Arena 端到端流程（G1 Path A）

更新时间：2026-02-16

本文是你当前目标的可执行流程文档：
- 机器人：`G1`
- 控制：`g1_wbc_pink`（Path A 快速落地）
- 抓取：先单手（右手）
- 执行：离线回放
- 优先级：先保证物体轨迹精度（运动学回放），再升级动力学交互

---

## 0. 环境与仓库映射

- `hoifhli_release`：HOI / text-to-motion 生成（物体轨迹来源）
- `BODex`：灵巧手抓取位姿（`grasp.npy`）
- `IsaacLab-Arena`：G1 回放与验证

推荐环境：
- HOIFHLI：`hoifhli_env`
- BODex：`objdex`
- Isaac：`isaaclab_arena`

---

## 1. 一次性安装（已在 objdex 验证）

在 `objdex` 中安装 BODex 依赖：

```bash
/home/ubuntu/miniconda3/envs/objdex/bin/pip install setuptools_scm
/home/ubuntu/miniconda3/envs/objdex/bin/pip install usd-core
/home/ubuntu/miniconda3/envs/objdex/bin/pip install --no-cache-dir torch-scatter -f https://data.pyg.org/whl/torch-2.4.0+cu121.html
/home/ubuntu/miniconda3/envs/objdex/bin/pip install --no-cache-dir warp-lang
/home/ubuntu/miniconda3/envs/objdex/bin/pip install --no-cache-dir coal==3.0.1
/home/ubuntu/miniconda3/envs/objdex/bin/pip install --no-cache-dir cmeel-eigen

cd /home/ubuntu/DATA2/workspace/xmh/BODex
/home/ubuntu/miniconda3/envs/objdex/bin/pip install -e . --no-build-isolation
```

编译 `coal_openmp_wrapper`（本机需要 `cmeel` 路径）：

```bash
cd /home/ubuntu/DATA2/workspace/xmh/BODex/src/curobo/geom/cpp
env CONDA_PREFIX=/home/ubuntu/miniconda3/envs/objdex/lib/python3.8/site-packages/cmeel.prefix \
  /home/ubuntu/miniconda3/envs/objdex/bin/python setup.py install
```

说明：为兼容当前 `coal/boost` 版本，`BODex/src/curobo/geom/cpp/setup.py` 中编译标准需为 `-std=c++14`。

---

## 2. HOIFHLI 生成 human-object 轨迹

### 2.1 直接运行已有示例（推荐先跑通）

```bash
cd /home/ubuntu/DATA2/workspace/xmh/hoifhli_release
/home/ubuntu/miniconda3/envs/hoifhli_env/bin/bash scripts/sample.sh
```

### 2.2 指定对象（例如 `smallbox`）

```bash
cd /home/ubuntu/DATA2/workspace/xmh/hoifhli_release
/home/ubuntu/miniconda3/envs/hoifhli_env/bin/python sample.py \
  --window=120 \
  --data_root_folder="./data/processed_data" \
  --project="./experiments" \
  --test_sample_res \
  --use_long_planned_path \
  --add_interaction_root_xy_ori \
  --add_interaction_feet_contact \
  --add_finger_motion \
  --use_guidance_in_denoising \
  --vis_wdir="moving_box" \
  --test_object_names smallbox
```

### 2.3 找到 HOI 输出文件

```bash
find /home/ubuntu/DATA2/workspace/xmh/hoifhli_release/results/interaction -name human_object_results.pkl
```

你当前机器已有可用样例（可直接用）：
- `hoifhli_release/results/interaction/compare_fine_01/.../human_object_results.pkl`

---

## 3. BODex 生成 grasp.npy

## 3.1 启动前环境变量（必须）

```bash
export BODEX_ROOT=/home/ubuntu/DATA2/workspace/xmh/BODex
export PYTHONPATH=$BODEX_ROOT/src:$PYTHONPATH
export LD_LIBRARY_PATH=/home/ubuntu/miniconda3/envs/objdex/lib/python3.8/site-packages/cmeel.prefix/lib:$LD_LIBRARY_PATH
```

## 3.2 生成 grasp（前提：已准备 DGN_2k 资产）

```bash
cd $BODEX_ROOT
/home/ubuntu/miniconda3/envs/objdex/bin/python example_grasp/plan_batch_env.py \
  -c sim_shadow/fc.yml \
  -w 1 \
  -m npy \
  -f run_hoi_bridge
```

查找输出：

```bash
find $BODEX_ROOT/src/curobo/content/assets/output/run_hoi_bridge -name "*grasp.npy"
```

注意：
- 你当前仓库里还没有 `DGN_2k` 物体资产目录（`src/curobo/content/assets/object/DGN_2k/...`）。
- 所以 BODex 这一步在未补资产前无法产出真实 `grasp.npy`。

---

## 4. 由 HOI + BODex 构建 G1 回放动作

```bash
cd /home/ubuntu/DATA2/workspace/xmh/IsaacLab-Arena

/home/ubuntu/miniconda3/envs/isaaclab_arena/bin/python tools/hoi_bodex_g1_bridge/build_replay.py \
  --hoi-pickle /home/ubuntu/DATA2/workspace/xmh/hoifhli_release/results/interaction/compare_fine_01/10_long_seq_w_waypoints_pidx_0_oidx_3_interaction_guidance/objs_step_10_bs_idx_0_vis_no_scene/human_object_results.pkl \
  --bodex-grasp-npy /abs/path/to/your_grasp.npy \
  --output-hdf5 /tmp/g1_bridge_run1/replay_actions.hdf5 \
  --output-object-traj /tmp/g1_bridge_run1/object_kinematic_traj.npz \
  --output-debug-json /tmp/g1_bridge_run1/bridge_debug.json \
  --hoi-fps 30 \
  --target-fps 50 \
  --yaw-face-object
```

如果先不接 BODex（只验证链路）：

```bash
cd /home/ubuntu/DATA2/workspace/xmh/IsaacLab-Arena

/home/ubuntu/miniconda3/envs/isaaclab_arena/bin/python tools/hoi_bodex_g1_bridge/build_replay.py \
  --hoi-pickle /home/ubuntu/DATA2/workspace/xmh/hoifhli_release/results/interaction/compare_fine_01/10_long_seq_w_waypoints_pidx_0_oidx_3_interaction_guidance/objs_step_10_bs_idx_0_vis_no_scene/human_object_results.pkl \
  --output-hdf5 /tmp/g1_bridge_run1/replay_actions.hdf5 \
  --output-object-traj /tmp/g1_bridge_run1/object_kinematic_traj.npz \
  --output-debug-json /tmp/g1_bridge_run1/bridge_debug.json \
  --hoi-fps 30 \
  --target-fps 50 \
  --yaw-face-object
```

---

## 5. Isaac 中回放（含物体运动学轨迹）

```bash
cd /home/ubuntu/DATA2/workspace/xmh/IsaacLab-Arena

/home/ubuntu/miniconda3/envs/isaaclab_arena/bin/python isaaclab_arena/examples/policy_runner_kinematic_object_replay.py \
  --headless \
  --device cpu \
  --policy_type replay \
  --replay_file_path /tmp/g1_bridge_run1/replay_actions.hdf5 \
  --episode_name demo_0 \
  --kin-traj-path /tmp/g1_bridge_run1/object_kinematic_traj.npz \
  --kin-asset-name brown_box \
  --kin-apply-timing pre_step \
  --max-steps 408 \
  galileo_g1_locomanip_pick_and_place \
  --object brown_box \
  --embodiment g1_wbc_pink
```

---

## 6. 这个流程每一步在做什么

1. HOIFHLI：给出人-物交互序列，核心拿 `obj_pos + obj_rot_mat`。
2. BODex：给出手相对物体的抓取位姿（pregrasp / grasp）。
3. `build_replay.py`：
   - 把 HOI 轨迹从 30Hz 重采样到 50Hz。
   - 用 BODex 的 object->hand 相对位姿，沿物体轨迹生成手腕目标。
   - 生成 `g1_wbc_pink` 的 23D action（HDF5）。
   - 分段写入 `navigation/pregrasp/approach/grasp_close/grasp_hold`。
4. `policy_runner_kinematic_object_replay.py`：
   - 机器人按 23D action 回放。
   - 物体每步强制写入目标位姿（运动学回放）。
   - 先把“轨迹精度”做对，避免动力学接触噪声干扰调试。

---

## 7. 已知问题与当前状态

1. `objdex` 中 BODex 依赖已安装并验证可导入。
2. `coal_openmp_wrapper` 需要 `LD_LIBRARY_PATH` 包含 `cmeel.prefix/lib`。
3. 当前还缺 `DGN_2k` 资产，因此 BODex 真实 grasp 仍需你补齐资产或做对象预处理导入。
4. G1 路线当前是 Path A 快速版本：腕目标 + 手开合状态，不含 Inspire 精细指型。
5. BODex 抓取求解默认走 CUDA；无可用 GPU 时会在初始化时报错（`cudaGetDeviceCount`）。

---

## 8. 可改进方向（按优先级）

1. 资产打通：把 HOIFHLI 的目标物体自动转成 BODex scene_cfg + info + mesh（避免手工准备 DGN 资产）。
2. 抓型升级：从“手开合”升级到 `G1 + InspireHand` 手指关节级映射。
3. 反推优化：把 base/wrist 的启发式规划升级为几何约束优化（object+grasp+关节限位+可达性）。
4. 动力学升级：从 kinematic object replay 过渡到真实接触控制（闭环修正 + 接触一致性）。
5. 分段策略：延续 MoMaGen 范式，把导航段/操作段策略与参数完全解耦。

---

## 9. 最短可跑命令链（你现在就能执行）

1. HOI 结果直接复用已有 `human_object_results.pkl`。
2. 用 `build_replay.py` 先生成 `replay_actions.hdf5 + object_kinematic_traj.npz`。
3. 用 `policy_runner_kinematic_object_replay.py` 回放验证 G1 与物体时序。
4. 等 BODex 资产补齐后，把 `--bodex-grasp-npy` 接入同一流程。
