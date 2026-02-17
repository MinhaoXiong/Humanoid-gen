# Humanoid-gen-pack

你当前项目的最小可运行链路（G1 Path A）代码整理包。

目标：
1. `hoifhli_release` 产出 human-object motion（取 object 轨迹）
2. `BODex` 产出单手 grasp（pregrasp/grasp）
3. `build_replay.py` 融合为 G1 `g1_wbc_pink` 23D 回放动作
4. Isaac 回放并保存视频

## 目录说明

- `bridge/build_replay.py`：HOI + BODex -> G1 replay 动作/物体轨迹
- `isaac_replay/policy_runner_kinematic_object_replay.py`：Isaac 回放（含视频导出）
- `scripts/01_hoi_sample.sh`：HOI 采样和结果定位
- `scripts/02_bodex_grasp.sh`：BODex grasp 生成
- `scripts/03_build_replay.sh`：构建回放文件
- `scripts/04_isaac_replay_video.sh`：回放并导出视频
- `docs/`：桥接流程与状态说明（中文）
- `notes/BODex_cpp14_fix.md`：BODex 编译修复记录

## 环境

- HOI：`hoifhli_env`
- BODex：`objdex`
- Isaac：`isaaclab_arena`

## 一次完整执行

### 1) HOI 生成

```bash
cd /home/ubuntu/DATA2/workspace/xmh/Humanoid-gen-pack
bash scripts/01_hoi_sample.sh
```

得到 `human_object_results.pkl` 后，记住它的绝对路径。

### 2) BODex 生成 grasp

```bash
cd /home/ubuntu/DATA2/workspace/xmh/Humanoid-gen-pack
bash scripts/02_bodex_grasp.sh
```

得到 `grasp.npy` 后，记住它的绝对路径。

### 3) 构建回放

```bash
cd /home/ubuntu/DATA2/workspace/xmh/Humanoid-gen-pack
bash scripts/03_build_replay.sh \
  /abs/path/to/human_object_results.pkl \
  /abs/path/to/grasp.npy \
  /tmp/g1_bridge_run1
```

输出：
- `/tmp/g1_bridge_run1/replay_actions.hdf5`
- `/tmp/g1_bridge_run1/object_kinematic_traj.npz`
- `/tmp/g1_bridge_run1/bridge_debug.json`

### 4) Isaac 回放 + 视频

```bash
cd /home/ubuntu/DATA2/workspace/xmh/Humanoid-gen-pack
bash scripts/04_isaac_replay_video.sh /tmp/g1_bridge_run1 g1_bridge_run1
```

默认视频目录：
- `/home/ubuntu/DATA2/workspace/xmh/IsaacLab-Arena/.workflow_data/videos`

## 你当前反馈的问题（右手飞出视角）

当前 Path A 是“腕目标 + 手开合”，容易在某些帧产生腕部不可达目标。建议按顺序处理：

1. 在 `bridge/build_replay.py` 增加腕目标相对躯干的可达域裁剪（半径/高度限幅）
2. 增大 pregrasp 偏置、减小 approach 位移
3. 放慢接近和闭合时序（减小速度尖峰）
4. 再升级到 `G1 + InspireHand` 做手指级抓型约束
