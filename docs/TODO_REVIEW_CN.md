# Todo 功能实现状态审查报告

更新时间：2026-02-19

本文档对照 `todo/` 中规划的两大功能，逐项审查代码库实际实现状态。

---

## 一、总结

| 功能 | todo规划 | 实际实现 | 差距 |
|------|---------|---------|------|
| 功能1: Walk-to-Grasp | CuRobo混合架构(8步) | 开环nav_cmd速度指令 | todo与实际方案完全不同 |
| 功能2: InspireHand+CEDex | 8步计划 | 脚本层完成，embodiment未改 | 差G1 USD切换 |

---

## 二、功能1: Walk-to-Grasp 逐项审查

### 2.1 Todo规划 vs 实际方案

Todo规划了CuRobo混合架构（8步），但实际代码走了**开环速度指令方案**——直接在 `build_arm_follow_replay.py` 中生成三阶段nav_cmd（转向→直行→转向），无CuRobo依赖。

### 2.2 Todo各Step实现状态

| Step | 规划内容 | 状态 | 说明 |
|------|---------|------|------|
| 1 | CuRobo Robot Config YAML | ❌ 未实现 | `planning/` 目录不存在，实际方案不需要 |
| 2 | CuRobo规划Wrapper | ❌ 未实现 | 同上 |
| 3 | G1+InspireHand Embodiment | ❌ 未实现 | g1.py仍用 `g1_29dof_with_hand_rev_1_0.usd`(Dex3-1) |
| 4 | 三阶段策略类 | ⚠️ 替代实现 | 无独立policy类，逻辑在build_arm_follow_replay.py中 |
| 5 | CEDex生成脚本 | ✅ 已实现 | `scripts/generate_inspirehand_grasps.py` + `.sh` |
| 6 | 抓取格式转换 | ✅ 已实现 | `scripts/convert_cedex_to_isaaclab.py` |
| 7 | 运行入口 | ⚠️ 替代实现 | 无独立run脚本，功能集成在 `08_debug_arm_follow_gui.sh` |
| 8 | 验证 | ⚠️ 部分完成 | 有artifacts测试记录，但无InspireHand端到端验证 |

### 2.3 实际已实现的Walk-to-Grasp（todo中未记录）

代码中已有完整的walk-to-grasp实现，但todo文档完全没有反映：

**`isaac_replay/build_arm_follow_replay.py`**:
- `_build_walk_to_grasp_nav_cmds()` (L230-286): 三阶段导航指令生成（turn1→move→turn2）
- `_resolve_walk_target_base_pose()` (L289-322): 根据物体位置自动计算目标base位姿
- `--walk-to-grasp` flag启用导航阶段，`--walk-target-offset-obj-w` 控制停靠偏移
- 导航阶段手臂固定在 `--right-wrist-nav-pos` 姿态
- `--walk-pregrasp-hold-steps` 导航结束后稳定等待

**`scripts/08_debug_arm_follow_gui.sh`**:
- kitchen场景默认 `WALK_TO_GRASP=1`（L58）
- 完整参数暴露：速度、角速度、dt、偏移、朝向模式
- 自动从debug json读取 `recommended_kin_start_step` 和 `total_steps`

**与todo规划的关键差异**:
- 无CuRobo：直线走向目标，无碰撞检测，无IK可达验证
- 无独立planning模块：逻辑嵌入build脚本
- 无独立policy类：通过hdf5 replay实现
- 优点：简单可靠，无额外GPU依赖，已能跑通

---

## 三、功能2: InspireHand + CEDex 逐项审查

### 3.1 Todo各Step实现状态

| Step | 规划内容 | 状态 | 说明 |
|------|---------|------|------|
| 1 | CuRobo config (同功能1) | ❌ 未实现 | 见功能1 |
| 2 | CuRobo planner (同功能1) | ❌ 未实现 | 见功能1 |
| 3 | G1+InspireHand Embodiment | ❌ 未实现 | g1.py未修改，仍用Dex3-1 USD |
| 4 | 三阶段策略 (同功能1) | ⚠️ 替代实现 | 见功能1 |
| 5 | CEDex生成InspireHand抓取 | ✅ 已实现 | 见下文 |
| 6 | 抓取格式转换 | ✅ 已实现 | 见下文 |
| 7 | 运行入口 | ⚠️ 替代实现 | 集成在08脚本 |
| 8 | 验证 | ❌ 未完成 | InspireHand端到端未验证 |

### 3.2 已完成的CEDex集成（todo中部分记录）

**CEDex-Grasp侧**（`CEDex-Grasp/`）:
- ✅ `urdf_assets_meta.json` 注册了inspirehand
- ✅ `HandModel.py` 16-part语义映射
- ✅ `CMapAdam.py` 优化器适配
- ✅ `controller.py` 开合策略
- ✅ `generate_data.py` 支持 `--robot_name inspirehand`
- ⚠️ 状态：代码适配完成，端到端验证未做（`INSPIREHAND_ADAPTATION.md` 明确说明）

**Humanoid-gen-pack侧**:
- ✅ `scripts/generate_inspirehand_grasps.py`: 调用CEDex AdamGrasp，保存top-k抓取到.pt
- ✅ `scripts/generate_inspirehand_grasps.sh`: wrapper脚本
- ✅ `scripts/convert_cedex_to_isaaclab.py`: CEDex .pt → wrist pose参数（rot6d→quat转换）
- ✅ `build_arm_follow_replay.py` 内置CEDex导入: `--cedex-grasp-pt` 直接加载.pt覆盖wrist pose
- ✅ `08_debug_arm_follow_gui.sh` 暴露 `CEDEX_GRASP_PT` 等环境变量

### 3.3 未完成：G1 Embodiment切换

**当前状态** (`repos/IsaacLab-Arena/isaaclab_arena/embodiments/g1/g1.py`):
- L124: `usd_path=...g1_29dof_with_hand_rev_1_0.usd` — 仍是Dex3-1
- 无 `G1InspireHandSceneCfg` 类
- 无 `G1InspireHandWBCPinkEmbodiment` 类
- hand actuator用通配符 `.*_hand_.*`，理论上兼容InspireHand关节名

**已有的InspireHand配置**（可直接参考）:
- `IsaacLab-Arena/submodules/IsaacLab/source/isaaclab_assets/robots/unitree.py` 中有 `G1_INSPIRE_FTP_CFG`
- USD: `g1_29dof_inspire_hand.usd`
- 手指关节: `.*_index_.*`, `.*_middle_.*`, `.*_thumb_.*`, `.*_ring_.*`, `.*_pinky_.*`（24个）
- `pickplace_unitree_g1_inspire_hand_env_cfg.py` 有完整环境配置

---

## 四、Todo文档本身的问题

### 4.1 与实际代码严重脱节

Todo (`03_implementation_steps.md`) 规划的CuRobo 8步方案，实际代码一步都没按这个走。实际实现是更务实的开环方案，但todo没有更新。

### 4.2 已实现功能未记录

以下已实现的功能在todo中完全没有体现：
- `build_arm_follow_replay.py` 的 `--walk-to-grasp` 三阶段导航
- `build_arm_follow_replay.py` 的 `--cedex-grasp-pt` CEDex直接导入
- `08_debug_arm_follow_gui.sh` 的完整walk-to-grasp + CEDex参数化
- `convert_cedex_to_isaaclab.py` 格式转换工具
- `generate_inspirehand_grasps.py` 抓取生成脚本

---

## 五、剩余待完成工作

### 5.1 必须完成

| 优先级 | 任务 | 文件 | 说明 |
|--------|------|------|------|
| P0 | G1 embodiment切换到InspireHand | `g1.py` | 新增G1InspireHandSceneCfg，USD改为g1_29dof_inspire_hand.usd |
| P0 | CEDex端到端验证 | `CEDex-Grasp/` | 运行generate_data.py --robot_name inspirehand验证输出 |
| P0 | InspireHand抓取→Isaac回放端到端 | 全链路 | CEDex生成→convert→build_replay→policy_runner |

### 5.2 建议完成

| 优先级 | 任务 | 说明 |
|--------|------|------|
| P1 | 更新todo文档 | 让todo反映实际实现方案而非CuRobo空想 |
| P1 | InspireHand手指关节映射 | 当前hand_state只有开/合(1D)，InspireHand有24个关节 |
| P2 | 碰撞检测 | 当前直线导航无避障，复杂场景会撞障碍物 |
| P2 | IK可达性验证 | 当前不检查停靠位置手臂能否够到物体 |

### 5.3 CuRobo方案是否还需要

`docs/CUROBO_G1_MOTION_PLANNING_CN.md` 已分析了当前方案的问题（手臂穿模、无碰撞检测、无IK验证）。CuRobo方案是长期正确方向，但当前开环方案已能跑通基本场景。建议：
- 短期：先用开环方案完成InspireHand端到端验证
- 中期：加入CuRobo做手臂MP（解决穿模问题）
- 长期：完整CuRobo混合架构（导航避障+手臂MP+IK验证）
