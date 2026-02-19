# Cracker Box 场景功能开发总览

## 开发基础

基于现有 `scripts/08_debug_arm_follow_gui.sh` pipeline 扩展，该pipeline的三步链路：
```
generate_debug_object_traj.py → build_arm_follow_replay.py → policy_runner_kinematic_object_replay.py
```

当前运行命令：
```bash
cd "$PACK_ROOT"
DEVICE="$DEVICE" BASE_POS_W="$BASE_POS_W" G1_INIT_YAW_DEG="$G1_INIT_YAW_DEG" \
bash scripts/08_debug_arm_follow_gui.sh \
  "$PACK_ROOT/artifacts/debug_schemeA2_gui" \
  lift_place \
  kitchen_pick_and_place \
  cracker_box
```

## 两大功能需求

### 功能1: Walk-to-Grasp（导航+手臂跟随）
当前：机器人固定在桌边（`BASE_POS_W=0.05,0.0,0.0`），`nav_cmd`全为0，只有手臂跟随物体。
目标：机器人从远处出生 → 走向桌子 → 停下后手臂跟随物体replay。

核心改动：`build_arm_follow_replay.py` 中 `nav_cmd` 从全0改为先有导航阶段。

### 功能2: 替换Dex3-1为InspireHand + CEDex-Grasp抓取
当前：G1使用Dex3-1灵巧手，抓取位姿手动指定。
目标：换用InspireHand，通过CEDex-Grasp生成抓取位姿。

## 文件索引

- [01_feature1_walk_to_grasp.md](01_feature1_walk_to_grasp.md) — 功能1: 导航+手臂跟随
- [02_feature2_inspirehand_grasp.md](02_feature2_inspirehand_grasp.md) — 功能2: InspireHand抓取
- [03_codex_prompt.md](03_codex_prompt.md) — 给Codex的完整prompt
