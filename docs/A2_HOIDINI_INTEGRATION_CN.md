# A-2 / A-HOI 调试链路与 HOIDiNi 接入分析（中文详细版）

本文目标：
- 把当前已经实现的 A-2 / A-HOI 工作流按“代码文件 + 数据格式 + 命令链路”写清楚；
- 评估 HOIDiNi 是否可用于同一任务（答案：可以），并说明它与 HOIFHI 的关键差异；
- 给出一条可执行的 HOIDiNi 接入规划，重点考虑 Isaac 场景空间约束。

---

## 1. 当前 A-2 / A-HOI 已实现内容（落地现状）

### 1.1 A-2 目标与控制策略
A-2 是“可控基线”调试方案：
- 场景：桌面任务场景（默认 `kitchen_pick_and_place`）；
- 机器人：`pelvis/base` 不导航（`NAV_CMD = 0`）；
- 行为：只让手臂跟随物体轨迹（arm-follow），用于隔离问题来源。

这套方案用于回答一个核心问题：
- 如果 A-2 稳定，而 HOI 回放不稳定，则问题主要在“HOI 轨迹坐标/尺度/场景对齐”；
- 如果 A-2 也不稳定，则优先查场景资产、机器人初始位姿、腕部偏移配置。

### 1.2 关键代码文件
- `isaac_replay/generate_debug_object_traj.py`
  - 生成 synthetic 物体轨迹（`line/circle/lift_place`）。
- `isaac_replay/build_arm_follow_replay.py`
  - 把物体轨迹转换成 replay action（固定 `NAV_CMD`，仅上肢跟随）。
- `isaac_replay/policy_runner_kinematic_object_replay.py`
  - 在 Isaac 中按 hdf5 action + npz 物体轨迹回放。
- `scripts/08_debug_arm_follow_gui.sh`
  - A-2 GUI 一键脚本。
- `scripts/09_debug_arm_follow_from_hoi_headless.sh`
  - A-HOI headless 一键脚本（HOI 轨迹版本）。

### 1.3 A-HOI（A 的 HOI 轨迹版）
A-HOI 保留 A-2 控制链路，仅替换物体轨迹来源：
- `HOI pkl -> 轨迹约束重映射 -> arm-follow replay -> headless replay`

已验证产物（示例目录）：
- `artifacts/acceptance_a2_hoi/object_kinematic_traj.npz`
- `artifacts/acceptance_a2_hoi/replay_actions_arm_follow.hdf5`
- `artifacts/acceptance_a2_hoi/bridge_debug.json`
- `artifacts/acceptance_a2_hoi/debug_replay.json`

关键验收点：
- `debug_replay.json` 中 `nav_cmd_abs_max = [0,0,0]`；
- `bridge_debug.json` 中存在 `traj_constraints` 且范围进入场景可操作区。

---

## 2. A-HOI 为什么需要“场景约束重映射”

`bridge/build_replay.py` 本身支持轨迹约束，顺序为：
1. `scale_xyz`
2. `offset_w`
3. `align_first_pos_w`
4. `align_last_pos_w`（可选，支持末段 ramp）
5. `clip_z / clip_xy`

针对 `kitchen_pick_and_place` 的默认约束（在 `09` 脚本中）：
- `TRJ_SCALE_XYZ="0.13,0.22,0.30"`
- `ALIGN_FIRST_POS_W="0.40,0.00,0.10"`
- `CLIP_Z_MIN=0.08`, `CLIP_Z_MAX=0.38`
- `CLIP_XY_MIN="0.05,-0.45"`, `CLIP_XY_MAX="0.65,0.45"`

作用：
- 把 HOI 世界系下的轨迹压缩并搬运到厨房台面局部可操作区域；
- 避免“物体飞天/穿模/跑出桌面边界”。

---

## 3. HOIDiNi 能不能用于同一任务？结论与前提

结论：**可以用于这个任务**，但不能“原样直连”当前 HOIFHI 输入接口，必须做一层格式适配与场景重映射。

原因：
- 当前 bridge 主入口 `_load_hoi_human_object_results` 读取的是 HOIFHI 样式字段：
  - `obj_pos [T,3]`
  - `obj_rot_mat [T,3,3]`
  - `object_name`
- HOIDiNi 推理输出（`*final.pickle`）里可用字段是：
  - `smpldata.trans_obj [T,3]`
  - `smpldata.poses_obj [T,3]`（axis-angle，不是旋转矩阵）
  - `object_name`

所以要做：
- `poses_obj(axis-angle) -> obj_rot_mat(3x3)` 转换；
- 导出成 bridge 可读的最小 pkl 结构（`obj_pos/obj_rot_mat/object_name`）；
- 再走现有 A-HOI 约束链路。

---

## 4. HOIDiNi 与 HOIFHI 的差异（对接层面）

| 维度 | HOIFHI（当前已接） | HOIDiNi（可接入） | 对 A-HOI 的影响 |
|---|---|---|---|
| 轨迹文件 | `human_object_results.pkl` | `cphoi__...__final.pickle` | 需要适配器 |
| 平移字段 | `obj_pos` | `smpldata.trans_obj` | 字段映射 |
| 旋转字段 | `obj_rot_mat` | `smpldata.poses_obj`(axis-angle) | 需转 rotmat |
| 典型 fps | 常见 30 | 训练配置常见 20（`train.fps=20`） | 需重采样到 50 |
| 场景先验 | 与你当前链路已验证 | DNO 有桌面约束，但不是 Isaac 厨房坐标系 | 仍需场景重映射 |
| 物体命名 | 可能与 Isaac 资产不一致 | 也可能不一致（如 teapot/flute） | 需 `object_name -> Isaac asset` 映射 |

补充：HOIDiNi 的 `final` 合成策略是“body 来自 phase2，object 6DoF 来自 phase1”，这一点在 `cphoi_inference.py` 的 `merge_samples` 中已明确。

---

## 5. HOIDiNi 样例数据观察（本机已有结果）

在 `HOIDiNi/hoidini_results_test/` 的 `*final.pickle`，可直接读到：
- `smpldata.trans_obj`、`smpldata.poses_obj`
- `sampling_cphoi.yaml` 中 `train.fps=20`、`n_frames=115`

样例（简化）：
- `teapot`：`trans_obj` 跨度约 `[0.138, 0.334, 0.181]`
- `flute`：`trans_obj` 跨度约 `[1.335, 0.619, 0.383]`

这说明：
- HOIDiNi 的轨迹尺度/起点位置在不同样本间差异可较大；
- 直接塞进 Isaac 厨房场景风险高，仍然需要 `scale/align/clip` 这类硬约束。

---

## 6. HOIDiNi 接入 A-HOI 的推荐落地流程

### Step A: 运行 HOIDiNi 得到 `final.pickle`
示例入口：
- `HOIDiNi/scripts/inference.sh`
- 或 `python hoidini/cphoi/cphoi_inference.py ...`

### Step B: 把 `final.pickle` 适配成 bridge 输入 pkl
适配目标字段：
- `obj_pos = smpldata.trans_obj`
- `obj_rot_mat = axis_angle_to_rotmat(smpldata.poses_obj)`
- `object_name = final.pickle 中对象名`

### Step C: 走 A-HOI 现成链路
用 `scripts/09_debug_arm_follow_from_hoi_headless.sh`，并保留场景约束参数。

### Step D: 检查三类指标
- 轨迹几何：`bridge_debug.json` 的 `result_min_w/result_max_w`
- 控制冻结：`debug_replay.json` 的 `nav_cmd_abs_max`
- 回放稳定性：是否出现飞天、穿台面、越界。

---

## 7. 为什么 HOIDiNi 不能“直接替换”而要先适配

1. 字段不兼容：`poses_obj` 不是 rotmat。  
2. 坐标/场景不一致：HOIDiNi 的桌面先验不是 Isaac 厨房场景真值边界。  
3. 物体资产不一致：名称与 mesh 来源不统一。  
4. 时序不一致：常见 20fps，Isaac 回放常用 50fps。  

所以正确路径不是“直接替换”，而是“先格式适配，再做场景约束映射”。

---

## 8. 实操建议（优先级）

1. 先用 HOIDiNi 的一个短序列（100~150 帧）做 A-HOI 接入，快速调约束窗口。  
2. 先固定 `object_name` 到 Isaac 内置资产（如 `cracker_box`），绕过 mesh 缺失问题。  
3. 约束先紧后松：先保证不飞天/不越界，再放宽动作幅度。  
4. 每次迭代都保存 `bridge_debug.json`，只改一组参数，避免混淆变量。

---

## 9. 与当前仓库的关系

- 你现在已有 A-2/A-HOI 可跑链路；
- HOIDiNi 的接入工作本质是“新增一个输入适配层”；
- 一旦适配层做好，后续调参可完全复用 `09` 脚本与现有验收流程。

