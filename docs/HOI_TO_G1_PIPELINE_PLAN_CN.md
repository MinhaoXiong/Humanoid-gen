# Text-to-HOI → G1 机器人训练 Pipeline 规划与进度

## 总体目标

输入一段文本（如 "pick up the cracker box"），输出：
1. 物体运动轨迹 — 由 text-to-HOI 模型生成（hoifhli 或 HOIDiNi）
2. 适配 G1 机器人的物体轨迹 — 经过坐标变换、缩放、IK retarget，适合机器人操作
3. Isaac Sim 可导入的场景 — 包含物体、桌面、机器人的完整仿真环境
4. 机器人 walk-to-grasp 回放 — 完整的行走→抓取→操作 pipeline

## 4 模块 Pipeline 架构

```
┌─────────────────────────────────────────────────────────┐
│  Module A: Text-to-HOI Generation                       │
│  (hoifhli / HOIDiNi)                                    │
│  输入: text + object mesh                               │
│  输出: human_object_results.pkl                         │
│        (obj_pos [T,3], obj_rot_mat [T,3,3])             │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│  Module B: HOI-to-Robot Trajectory Adapter (新模块)      │
│  核心: 将人类操作轨迹适配到 G1 机器人工作空间             │
│  输入: HOI pkl + 场景配置                                │
│  输出: object_kinematic_traj.npz (标准格式)              │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│  Module C: Scene Generation (新模块)                     │
│  核心: 生成/组装 Isaac Sim 场景                          │
│  输入: object name + layout config                      │
│  输出: USD 场景 + solve_state.json                       │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│  Module D: Robot Replay & Training (已有，需整合)         │
│  核心: 脚本 13 的 walk→grasp→replay pipeline             │
│  输入: NPZ 轨迹 + USD 场景                              │
│  输出: HDF5 replay + Isaac Sim 仿真                     │
└─────────────────────────────────────────────────────────┘
```

---

## Module A: Text-to-HOI Generation

### 设计

两个 HOI 模型的输出格式不同：

| 字段 | hoifhli | HOIDiNi |
|------|---------|---------|
| 人体关节 | `human_jnts_pos [T,24,3]` | 无（bridge pkl 中不含） |
| 人体根位置 | `human_root_pos [T,3]` | 无 |
| 手指关节 | `finger_jnts_local_rot_aa [T,30,3]` | 无 |
| 物体位置 | `obj_pos [T,3]` | `obj_pos [T,3]` |
| 物体旋转 | `obj_rot_mat [T,3,3]` | `obj_rot_mat [T,3,3]` |
| 物体名称 | `object_name` | `object_name` |
| 坐标系 | Y-up (SMPL 惯例) | Y-up |

### 状态: ⬜ 未实现

Module A 本身是已有的外部模型（hoifhli_release / HOIDiNi），不需要我们实现。
用户手动运行 HOI 生成，产出 pkl 文件作为 Module B 的输入。

### 待做

- [ ] 编写 hoifhli 一键生成脚本（封装推理命令）
- [ ] 编写 HOIDiNi 一键生成脚本（封装推理命令）
- [ ] 统一输出 pkl 格式文档

---

## Module B: HOI-to-Robot Trajectory Adapter

### 设计

这是连接 HOI 生成和机器人操作的桥梁，核心子任务：

**B1. 统一 HOI 输入解析**
- 统一 loader 处理 hoifhli（有人体数据）和 HOIDiNi（仅物体数据）两种 pkl
- 自动检测 Y-up 坐标系并转换为 Z-up

**B2. 人类→机器人工作空间映射**
1. 坐标系对齐：Y-up → Z-up `(x,y,z) → (x,-z,y)`
2. 轨迹重心化：起始位置对齐到场景桌面（如 kitchen `(0.40, 0.00, 0.10)`）
3. 缩放：人类操作幅度缩放到 G1 可达范围（人类手臂 ~0.6m → G1 ~0.3m）
4. MuJoCo IK retarget：用 weld constraint 求解 G1 关节配置
5. 物体轨迹重建：保持手-物相对位姿，从 G1 palm 位置重建物体轨迹
6. 可达性裁剪：clip 到场景工作空间边界
7. 时间重采样：HOI 30fps → Isaac Sim 50fps

**B3. 两条路径**
- 有人体数据（hoifhli）：完整 IK retarget → 物体轨迹重建
- 无人体数据（HOIDiNi）：scale-only fallback，启发式缩放到 G1 工作空间

### 状态: ✅ 已实现

**实现文件：**
- `bridge/hoi_to_g1_retarget.py` — 核心 retarget 模块（382 行）
  - `load_hoi_pkl()` — 统一 HOI loader (B1)
  - `yup_to_zup_pos/rot()` — 坐标系转换
  - `G1Retargeter` class — MuJoCo mocap IK (B2)
  - `reconstruct_obj_traj_simple()` — 物体轨迹重建
  - `align_and_clip()` — 场景对齐与裁剪
  - `resample_pos/rot()` — 时间重采样
  - `adapt_obj_only()` — HOIDiNi scale-only fallback (B3)
  - `save_npz()` — 标准 NPZ 输出

**测试结果：**
- hoifhli pkl（有人体数据）→ 408 帧 @ 50fps，IK retarget 路径，位置裁剪到 kitchen 工作空间 (X: 0.4~0.65, Z: 0.1~0.4)
- HOIDiNi pkl（无人体数据）→ 191 帧 @ 50fps，scale fallback 路径

**已解决的问题：**
- MjSpec equality API `body1/body2` → `name1/name2`
- Symlink mesh 路径解析 → 使用绝对路径指向 Spider G1 XML
- Weld constraint 需要显式设置 `objtype = mjOBJ_BODY`
- CUDA tensor → numpy 需要先 `.cpu()`

---

## Module C: Scene Generation

### 设计

两种路线：

**路线 1：TabletopGen（桌面操作场景）**
```
文本描述 → text2img 生成参考图 → pipeline.py 生成桌面场景 GLB
→ isaac_final_scene.py 转换为 USD + 物理属性 → 注入物体运动学轨迹
```

**路线 2：SceneWeaver（完整房间场景）**
```
文本描述 → SceneWeaver agent 生成室内场景 → infinigen export USDC
→ solve_state.json 配置物理属性 → 注入物体运动学轨迹 + G1 机器人
```

**路线 3：复用 IsaacLab-Arena 现有场景（当前方案）**
- 直接使用 `kitchen_pick_and_place`、`galileo_g1_locomanip_pick_and_place` 等已有场景
- 通过参数调整物体位置、机器人初始位姿

### 状态: ⬜ 未实现（使用路线 3 复用现有场景作为临时方案）

当前通过 `SCENE_DEFAULTS` 配置复用 IsaacLab-Arena 场景，未集成 TabletopGen/SceneWeaver。

### 待做

- [ ] TabletopGen USD 导出 → IsaacLab-Arena 场景注册
- [ ] SceneWeaver USDC 导出 → IsaacLab-Arena 场景注册
- [ ] 场景-物体匹配逻辑（根据 HOI 物体自动选择/生成场景）

---

## Module D: Robot Replay & Training

### 设计

基于现有脚本 13 的 walk→grasp→replay pipeline：
1. 物体轨迹 NPZ → CuRobo IK 验证 → walk planning
2. build_arm_follow_replay → HDF5 replay actions
3. Isaac Sim policy runner → 运动学回放 + 视频录制

### 状态: ✅ 已整合

**实现文件：**
- `scripts/run_walk_to_grasp_todo.py` — 修改 Step 1，添加 `--hoi-pickle` 分支
  - 当设置 `--hoi-pickle` 时，调用 `bridge/hoi_to_g1_retarget.py` 替代合成 pattern
  - 当未设置时，保持原有 `generate_debug_object_traj.py` 合成轨迹逻辑
- `scripts/14_hoi_to_g1_walk_grasp.sh` — 端到端 shell 脚本
  - Step 1: HOI pkl → retarget → object_kinematic_traj.npz
  - Step 2: NPZ → build_arm_follow_replay → HDF5
  - Step 3: 读取轨迹帧数
  - Step 4: Isaac Sim replay

**未端到端测试：** Isaac Sim replay（Step 4）需要 Isaac Sim 运行环境，尚未实际运行验证。

---

## 总体进度汇总

| 模块 | 描述 | 状态 | 说明 |
|------|------|------|------|
| Module A | Text-to-HOI 生成 | ⬜ 未实现 | 外部模型，需封装一键脚本 |
| Module B | HOI→G1 轨迹适配 | ✅ 已实现 | `bridge/hoi_to_g1_retarget.py`，已测试 |
| Module C | 场景生成 | ⬜ 未实现 | 当前复用 IsaacLab-Arena 现有场景 |
| Module D | 机器人回放整合 | ✅ 已整合 | `run_walk_to_grasp_todo.py` + `14_hoi_to_g1_walk_grasp.sh` |

**已完成核心链路：** HOI pkl → retarget → NPZ → arm-follow replay → Isaac Sim（B + D）

**待完成：** HOI 生成封装（A）、场景生成集成（C）、Isaac Sim 端到端验证
