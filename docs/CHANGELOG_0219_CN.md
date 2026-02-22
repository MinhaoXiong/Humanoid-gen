# 2026-02-19 工作记录

## 概要

本次完成三项任务：验证 Spider InspireHand 重定向管线、实现 BODex seed 注入（模块 B）、实现人手距离排序（模块 C）。

---

## 1. Spider InspireHand 重定向管线验证

### 问题

之前 `spider/preprocess/generate_xml.py` 报 "Added 0 objects"，无法生成正确的 `scene.xml`。

### 根因

凸分解 mesh 文件命名不符合 spider 预期：
- 错误：`convex/convex_0.obj`
- 正确：`convex/0.obj`

`generate_xml.py` 第 336 行用 `suffix.isdigit()` 判断是否为碰撞几何体，`convex_0` 不是纯数字所以被跳过。

### 修复

```bash
cd spider/example_datasets/processed/synthetic/assets/objects/box/convex
mv convex_0.obj 0.obj
```

### 验证结果

```bash
# Step 1: 生成 scene.xml（成功）
conda run -n spider python spider/preprocess/generate_xml.py \
  --dataset-dir=example_datasets --dataset-name=synthetic \
  --robot-type=inspire --embodiment-type=right \
  --task=grasp_test --data-id=0 --no-show-viewer
# → Added 1 objects, Added 27 contact pairs

# Step 2: 运行 IK（成功）
MUJOCO_GL=egl conda run -n spider python spider/preprocess/ik.py \
  --dataset-dir=example_datasets --dataset-name=synthetic \
  --robot-type=inspire --embodiment-type=right \
  --task=grasp_test --data-id=0 --open-hand --no-show-viewer
# → best_qpos_diff_sum: 0.727
# → 输出: trajectory_ikrollout.npz, shape=(46, 25)
```

输出 qpos 布局（25 DOF）：
```
[0:6]   wrist: xyz + euler_xyz
[6:18]  finger: thumb_yaw, thumb_pitch, thumb_inter, thumb_distal,
                index_prox, index_inter, middle_prox, middle_inter,
                ring_prox, ring_inter, pinky_prox, pinky_inter
[18:25] object: xyz + quat_wxyz
```

---

## 2. 模块 B：Spider Seed 注入 BODex

### 修改文件

`scripts/generate_bodex_inspirehand_grasps.py`

### 新增函数

`load_spider_seed(npz_path, frame=-1)` — 将 spider IK 输出转换为 BODex seed_config：

| Spider 格式 | BODex 格式 |
|-------------|-----------|
| wrist xyz (3) | xyz (3) |
| wrist euler_xyz (3) | quat wxyz (4)（通过 scipy Rotation 转换） |
| finger 12 DOF | 6 独立 DOF（索引 `[0,1,4,6,8,10]`） |

输出 shape: `[1, 1, 13]`，直接传给 `GraspSolver.solve_batch_env(seed_config=...)`。

### 使用方式

```bash
python scripts/generate_bodex_inspirehand_grasps.py \
  --mesh-file object.obj \
  --seed-from-spider path/to/trajectory_ikrollout.npz \
  --seed-frame -1
```

---

## 3. 模块 C：人手距离排序

### 修改文件

`scripts/generate_bodex_inspirehand_grasps.py`

### 新增函数

`rank_by_human_distance(solutions, human_wrist_pose_7d, human_finger_6)` — 按与人手姿态的距离排序 BODex 结果：

```
d = ||t_robot - t_human||₂ + λ_rot · arccos(|q_robot · q_human|) + λ_finger · ||q_finger_diff||₂
```

参数：
- `--rank-by-human`：启用人手距离排序（替代默认的 grasp_error 排序）
- `--lambda-rot`：旋转距离权重（默认 1.0）
- `--lambda-finger`：手指距离权重（默认 0.5）

### 使用方式

```bash
# 同时使用 seed 注入 + 人手距离排序
python scripts/generate_bodex_inspirehand_grasps.py \
  --mesh-file object.obj \
  --seed-from-spider trajectory_ikrollout.npz \
  --rank-by-human --lambda-rot 1.0 --lambda-finger 0.5

# Shell 快捷方式（自动启用两者）
bash scripts/generate_bodex_inspirehand_grasps.sh object.obj spider_ik.npz output.pt
```

输出 `.pt` 额外包含 `human_distance` 和 `human_seed_13` 字段。

---

## 4. 修改文件清单

| 文件 | 操作 | 说明 |
|------|------|------|
| `scripts/generate_bodex_inspirehand_grasps.py` | 修改 | 新增 seed 注入 + 距离排序 |
| `scripts/generate_bodex_inspirehand_grasps.sh` | 修改 | 支持 spider npz 参数 |
| `docs/INSPIREHAND_RETARGET_CN.md` | 修改 | 更新 4.2/4.3 节文档 |
| `docs/CHANGELOG_0219_CN.md` | 新建 | 本文件 |
