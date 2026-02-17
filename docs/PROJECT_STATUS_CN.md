# 项目阶段性状态说明（中文）

更新时间：2026-02-16

## 1. 你最初让我做的事情（需求复述）

你希望做一个大而复杂的跨仓库项目，并先进行“非常详细分析 + 代码落地”，目标流程是：

1. 用 `hoifhli_release` 的 text-to-human-motion / HOI 结果，拿到“人-物交互下的物体运动轨迹”。
2. 把该物体交给 `BODex`，生成灵巧手抓取位姿（你先接受单手）。
3. 根据“物体轨迹 + 抓取位姿”反推人形机器人轨迹。
4. 在 `IsaacLab-Arena` 上以 `G1` 执行，先做离线回放。
5. 参考 `MoMaGen` 的流程范式：导航段与操作段拆分（不是强依赖直接复用代码）。

你明确给了方案约束：

- 机器人固定：`G1`
- 路线选择：路径 A（快速可跑）`g1_wbc_pink`
- 抓取方式：先单手
- 执行方式：离线轨迹回放
- 目标优先级：先把“物体轨迹精度”做对，再谈动力学真实抓取
- 允许简化：先用 BODex 提供腕目标（pregrasp/grasp），G1 手部先开/合状态

---

## 2. 我完成了什么（分析 + 实现）

## 2.1 已完成的技术分析（跨仓库）

我先做了静态分析并确认链路可行：

1. `hoifhli_release` 可导出 `human_object_results.pkl`，包含：
   - `obj_pos`（物体质心轨迹）
   - `obj_rot_mat`（物体旋转矩阵序列）
   - `object_name`
2. `BODex` 抓取结果可提供 `robot_pose` 与 `world_cfg`，可从中恢复物体与手的相对位姿（object->hand）。
3. `IsaacLab-Arena` 的 `g1_wbc_pink` 使用 23D action，`ReplayActionPolicy` 可直接从 HDF5 `data/demo_x/actions` 回放。
4. `MoMaGen` 在你当前阶段最有价值的是“分段流程范式”（导航/操作拆段），不是直接复制代码。

关键约束也已确认：

- `g1_wbc_pink` 当前只支持单环境（`num_envs == 1`）
- `g1_wbc_pink` 的手腕 action 目标是 pelvis frame，不是 world frame
- HOI 轨迹通常需从 30Hz 重采样到 Isaac 的 50Hz 节奏

## 2.2 已新增实现（不改你已有修改文件，只新增）

### A. 桥接主脚本

文件：
- `tools/hoi_bodex_g1_bridge/build_replay.py`

能力：

1. 读取 HOI 物体轨迹（支持 CPU 环境加载含 CUDA tensor 的 pickle）。
2. 重采样：
   - 位置：线性插值
   - 姿态：SLERP（四元数球面插值）
3. 读取 BODex（可选）：
   - 从 `grasp.npy` 解析 `robot_pose` 和 `world_cfg`
   - 自动反算 object->hand 的 pregrasp / grasp 相对位姿
4. 若暂时没有 BODex 文件，支持手动给定 object frame 下的 pregrasp/grasp 位姿（兜底可跑）。
5. 按 MoMaGen 式范式生成分段：
   - navigation
   - pregrasp
   - approach
   - grasp_close
   - grasp_hold
6. 生成 `g1_wbc_pink` 23D action HDF5（可直接 replay）。
7. 额外导出：
   - `object_kinematic_traj.npz`（物体世界系轨迹）
   - `bridge_debug.json`（分段索引与调试元信息）

### B. Isaac 运动学物体回放 runner

文件：
- `isaaclab_arena/examples/policy_runner_kinematic_object_replay.py`

能力：

1. 继承原 policy_runner 思路，继续使用 replay action。
2. 每步把 `pick_up_object`（可参数化 asset name）强制写到目标轨迹位姿。
3. 支持 pre_step / post_step 写入策略。
4. 实现你要求的“先做运动学物体回放（非动力学被推拉）”。

### C. 使用说明文档

文件：
- `tools/hoi_bodex_g1_bridge/README.md`

内容：

1. 为什么先做运动学物体回放。
2. 坐标系与接口约束说明。
3. 端到端命令模板（含有/无 BODex 两种模式）。
4. 当前限制与后续升级路径。

---

## 3. 最终结果（当前可交付）

截至当前，已经得到“可运行的最小闭环”：

1. 输入 HOI 轨迹，能自动生成 G1 replay 所需 HDF5（23D）。
2. 可在 Isaac 中同步回放机器人动作 + 物体运动学轨迹。
3. 产物分离清晰：
   - 机器人控制输入（HDF5 actions）
   - 物体真值轨迹（NPZ）
   - 调试元数据（JSON）

我已做过离线验证：

1. 脚本语法编译通过（`py_compile`）。
2. 使用你现有 HOI 样例成功生成：
   - HDF5 action：`(408, 23)`
   - object traj：位置/四元数长度一致
   - debug 分段索引正确写出

示例输出（本地测试路径）：

- `/tmp/g1_bridge/replay_actions.hdf5`
- `/tmp/g1_bridge/object_kinematic_traj.npz`
- `/tmp/g1_bridge/bridge_debug.json`

---

## 4. 当前仍未完成（边界说明）

当前版本是你要求的“路径A快速可跑版”，尚未覆盖最终完整版目标：

1. 尚未做 `G1 + inspirehand` 细手指抓型映射。
2. 尚未切换到动力学真实抓取（目前是运动学强制物体轨迹）。
3. 尚未加入全身最优化反推（当前 base / wrist 轨迹是工程化启发式）。
4. 尚未对所有资产做 HOI 与 BODex canonical frame 自动对齐。

---

## 5. 你现在可以怎么用（最短路径）

1. 先用 `build_replay.py` 生成你指定 case 的 `replay_actions.hdf5 + object_kinematic_traj.npz`。
2. 用 `policy_runner_kinematic_object_replay.py` 在 `galileo_g1_locomanip_pick_and_place + g1_wbc_pink` 回放验证。
3. 当你给出真实 BODex `grasp.npy` 后，把“手动兜底 grasp 参数”切换为真实 BODex 相对位姿，进入下一阶段。

---

## 6. 本次新增文件清单

1. `tools/hoi_bodex_g1_bridge/build_replay.py`
2. `isaaclab_arena/examples/policy_runner_kinematic_object_replay.py`
3. `tools/hoi_bodex_g1_bridge/README.md`
4. `tools/hoi_bodex_g1_bridge/PROJECT_STATUS_CN.md`（本文件）

---

## 7. 新增进展：BODex 依赖在 objdex 已安装

你刚才要求“把 BODex 装在 objdex 并继续步骤”，目前状态如下：

1. `objdex` 内已可导入：
   - `curobo`
   - `torch_scatter`
   - `coal`
   - `warp`
   - `pxr`
2. `coal_openmp_wrapper` 已编译安装成功。
3. 运行 BODex 脚本前需要设置：
   - `PYTHONPATH=/home/ubuntu/DATA2/workspace/xmh/BODex/src:$PYTHONPATH`
   - `LD_LIBRARY_PATH=/home/ubuntu/miniconda3/envs/objdex/lib/python3.8/site-packages/cmeel.prefix/lib:$LD_LIBRARY_PATH`
4. 当前剩余阻塞：
   - `BODex/src/curobo/content/assets/object/DGN_2k/...` 资产尚未准备，因此还不能产出真实 `grasp.npy`。
5. 端到端命令文档（从 HOI + BODex 起步）已新增：
   - `tools/hoi_bodex_g1_bridge/END_TO_END_FROM_HOI_BODEX_CN.md`
