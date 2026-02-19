# Cracker Box 场景功能开发总览

## 两大功能需求

### 功能1: MoMaGen式移动操作轨迹规划
当前状态：机器人生成在桌子边缘，只有手臂操作。
目标：机器人生成在离桌子较远处 → 走向物体 → 停在物体附近切换到预抓取位姿 → 手臂跟随物体replay。

### 功能2: 替换Dex3-1灵巧手为InspireHand + 基于CeDexGrasp生成抓取位姿
当前状态：机器人使用Unitree Dex3-1灵巧手，抓取位姿基于BODex生成。
目标：换用InspireHand，通过CEDex-Grasp为物体obj生成InspireHand的抓取位姿。

## 涉及的核心项目

| 项目 | 路径 | 作用 |
|------|------|------|
| IsaacLab-Arena | `IsaacLab-Arena/` | 场景管理、embodiment定义、任务执行 |
| MoMaGen | `MoMaGen/` | 移动操作数据生成（导航→MP→Replay三阶段） |
| BODex | `BODex/` | 基于CuRobo的灵巧手抓取合成（当前不支持InspireHand） |
| CEDex-Grasp | `CEDex-Grasp/` | 跨embodiment灵巧手抓取生成（已支持InspireHand） |
| ManipTrans | `ManipTrans/` | InspireHand URDF/mesh资源 |

## 文件索引

- [01_feature1_walk_to_grasp.md](01_feature1_walk_to_grasp.md) — 功能1详细分析与代码规划
- [02_feature2_inspirehand_grasp.md](02_feature2_inspirehand_grasp.md) — 功能2详细分析与代码规划
- [03_implementation_steps.md](03_implementation_steps.md) — 给Codex的分步代码实现指令
