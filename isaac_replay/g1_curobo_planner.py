#!/usr/bin/env python3
"""Planner utilities for TODO walk-to-grasp pipeline.

This module exposes a single planning API that prefers CuRobo when available
and falls back to a deterministic open-loop planner otherwise.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import os
import sys
import types
from typing import Any

import numpy as np


def _parse_csv_floats(text: str, expected_len: int, name: str) -> np.ndarray:
    values = [float(x.strip()) for x in text.split(",")]
    if len(values) != expected_len:
        raise ValueError(f"{name} expects {expected_len} values, got {len(values)}: {text}")
    return np.asarray(values, dtype=np.float64)


def _normalize_quat_wxyz(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    norm = np.linalg.norm(q)
    if norm < 1e-12:
        raise ValueError("Quaternion norm too small.")
    q = q / norm
    if q[0] < 0:
        q = -q
    return q


def _quat_to_rotmat_wxyz(q: np.ndarray) -> np.ndarray:
    w, x, y, z = _normalize_quat_wxyz(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _wrap_angle_rad(a: float) -> float:
    return float((a + math.pi) % (2.0 * math.pi) - math.pi)


def _line_intersects_aabb_2d(
    p0: tuple[float, float],
    p1: tuple[float, float],
    aabb_min: tuple[float, float],
    aabb_max: tuple[float, float],
) -> bool:
    # Liang-Barsky clipping in 2D
    x0, y0 = p0
    x1, y1 = p1
    dx = x1 - x0
    dy = y1 - y0
    p = (-dx, dx, -dy, dy)
    q = (x0 - aabb_min[0], aabb_max[0] - x0, y0 - aabb_min[1], aabb_max[1] - y0)
    u1, u2 = 0.0, 1.0
    for pi, qi in zip(p, q):
        if abs(pi) < 1e-12:
            if qi < 0:
                return False
            continue
        t = qi / pi
        if pi < 0:
            u1 = max(u1, t)
        else:
            u2 = min(u2, t)
        if u1 > u2:
            return False
    return True


def _scene_obstacles(scene: str) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    # coarse 2D obstacles for walk routing
    if scene == "kitchen_pick_and_place":
        return [((0.10, -0.62), (0.85, 0.62))]
    if scene == "galileo_g1_locomanip_pick_and_place":
        return [((0.20, -0.05), (1.10, 0.90))]
    return []


@dataclass
class PlannerRequest:
    planner: str
    strict_curobo: bool
    scene: str
    start_base_pos_w: tuple[float, float, float]
    start_base_yaw_rad: float
    object_pos_w: tuple[float, float, float]
    object_quat_wxyz: tuple[float, float, float, float]
    target_base_pos_w: tuple[float, float, float] | None
    target_offset_obj_w: tuple[float, float, float]
    target_offset_frame: str
    target_yaw_mode: str
    target_yaw_deg: float
    wrist_pos_w: tuple[float, float, float] | None = None
    wrist_quat_wxyz_ik: tuple[float, float, float, float] | None = None


@dataclass
class IKResult:
    reachable: bool
    joint_solution: list[float] | None
    position_error: float
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class MotionGenResult:
    success: bool
    joint_trajectory: list[list[float]] | None  # [T, 7] joint angles
    ee_pos_trajectory: list[list[float]] | None  # [T, 3] EE positions (pelvis frame)
    ee_quat_trajectory: list[list[float]] | None  # [T, 4] EE quats wxyz (pelvis frame)
    num_steps: int
    motion_time: float
    reason: str

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # Omit large trajectories from serialization, keep summary only
        if d.get("joint_trajectory"):
            d["joint_trajectory_len"] = len(d["joint_trajectory"])
            del d["joint_trajectory"]
        if d.get("ee_pos_trajectory"):
            del d["ee_pos_trajectory"]
        if d.get("ee_quat_trajectory"):
            del d["ee_quat_trajectory"]
        return d


@dataclass
class PlannerResult:
    planner_requested: str
    planner_used: str
    curobo_available: bool
    curobo_reason: str
    target_base_pos_w: tuple[float, float, float]
    target_base_yaw_rad: float
    path_waypoints_xy: list[tuple[float, float]]
    navigation_subgoals: list[tuple[list[float], bool]]
    notes: str
    ik_result: IKResult | None = None
    motion_gen_result: MotionGenResult | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # Use custom serialization for motion_gen_result
        if self.motion_gen_result is not None:
            d["motion_gen_result"] = self.motion_gen_result.to_dict()
        return d


def _probe_curobo() -> tuple[bool, str]:
    try:
        import importlib.util

        if importlib.util.find_spec("curobo") is not None:
            return True, "imported from environment"

        bodex_src = os.environ.get("BODEX_CUROBO_SRC", "/home/ubuntu/DATA2/workspace/xmh/BODex/src")
        if os.path.isdir(bodex_src) and bodex_src not in sys.path:
            sys.path.insert(0, bodex_src)

        # curobo __init__ may require setuptools_scm in a git checkout.
        if "setuptools_scm" not in sys.modules:
            stub = types.ModuleType("setuptools_scm")
            stub.get_version = lambda **_: "v0"
            sys.modules["setuptools_scm"] = stub

        import curobo  # noqa: F401

        return True, "imported from BODex source"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


_G1_ARM_CONFIG_PATH = os.path.join(
    os.path.dirname(__file__), "..", "configs", "curobo", "g1_right_arm.yml"
)


def _scene_collision_cuboids(scene: str) -> list[dict]:
    """Return 3-D cuboid obstacles for CuRobo world collision."""
    if scene == "kitchen_pick_and_place":
        return [
            {"name": "table", "pose": [0.475, 0.0, 0.4, 1, 0, 0, 0],
             "dims": [0.75, 1.24, 0.8]},
        ]
    if scene == "galileo_g1_locomanip_pick_and_place":
        return [
            {"name": "table", "pose": [0.65, 0.425, 0.4, 1, 0, 0, 0],
             "dims": [0.90, 0.95, 0.8]},
        ]
    return []


def _ik_check_reachability(
    wrist_pos: tuple[float, float, float],
    wrist_quat_wxyz: tuple[float, float, float, float],
    scene: str = "",
) -> IKResult:
    """Use CuRobo IKSolver to check if a wrist pose is reachable by G1 right arm."""
    try:
        import torch
        from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
        from curobo.types.math import Pose
        from curobo.types.robot import RobotConfig

        cfg_data, _ = _load_robot_cfg_data()
        robot_cfg = RobotConfig.from_dict(cfg_data["robot_cfg"])

        world_cfg = None
        try:
            world_cfg = _build_world_cfg(scene)
        except Exception:
            pass

        # Try with scene collision; fall back to no collision if deps missing
        try:
            ik_config = IKSolverConfig.load_from_robot_config(
                robot_cfg,
                world_model=world_cfg,
                num_seeds=32,
                position_threshold=0.01,
                rotation_threshold=0.05,
            )
        except Exception:
            robot_cfg = RobotConfig.from_dict(cfg_data["robot_cfg"])
            ik_config = IKSolverConfig.load_from_robot_config(
                robot_cfg,
                world_model=None,
                num_seeds=32,
                position_threshold=0.01,
                rotation_threshold=0.05,
                collision_checker_type=None,
                self_collision_check=False,
            )
        ik_solver = IKSolver(ik_config)

        # Convert wxyz -> xyzw for CuRobo Pose
        w, x, y, z = wrist_quat_wxyz
        goal = Pose.from_list(
            [wrist_pos[0], wrist_pos[1], wrist_pos[2], w, x, y, z],
            tensor_args=ik_solver.tensor_args,
        )

        result = ik_solver.solve_single(goal)
        success = bool(result.success.item())
        pos_err = float(result.position_error.min().item()) if hasattr(result, "position_error") else -1.0
        q_sol = result.solution.squeeze().tolist() if success else None

        return IKResult(success, q_sol, pos_err, "ok" if success else "no feasible IK solution")

    except Exception as exc:
        return IKResult(False, None, -1.0, f"{type(exc).__name__}: {exc}")


def _load_robot_cfg_data() -> tuple[dict, str]:
    """Load and patch g1_right_arm.yml, return (cfg_data, pack_root)."""
    import yaml
    cfg_path = os.path.abspath(_G1_ARM_CONFIG_PATH)
    with open(cfg_path, "r") as f:
        cfg_data = yaml.safe_load(f)
    pack_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    urdf_path = cfg_data["robot_cfg"]["kinematics"]["urdf_path"]
    cfg_data["robot_cfg"]["kinematics"]["urdf_path"] = urdf_path.replace(
        "{PACK_ROOT}", pack_root
    )
    return cfg_data, pack_root


def _build_world_cfg(scene: str):
    """Build CuRobo WorldConfig from scene obstacles. Returns None if no obstacles."""
    from curobo.geom.sdf.world import WorldConfig
    cuboids = _scene_collision_cuboids(scene)
    if not cuboids:
        return None
    world_dict: dict = {"cuboid": {}}
    for cub in cuboids:
        world_dict["cuboid"][cub["name"]] = {"pose": cub["pose"], "dims": cub["dims"]}
    return WorldConfig.from_dict(world_dict)


def plan_arm_trajectory(
    wrist_pos: tuple[float, float, float],
    wrist_quat_wxyz: tuple[float, float, float, float],
    rest_joint_angles: list[float] | None = None,
    scene: str = "",
    interpolation_dt: float = 0.02,
) -> MotionGenResult:
    """Use CuRobo MotionGen to plan rest -> target wrist pose trajectory."""
    try:
        import torch
        from curobo.wrap.reacher.motion_gen import (
            MotionGen, MotionGenConfig, MotionGenPlanConfig,
        )
        from curobo.types.robot import JointState as CuJointState
        from curobo.types.math import Pose

        cfg_data, _ = _load_robot_cfg_data()
        retract_cfg = cfg_data["robot_cfg"]["kinematics"]["cspace"]["retract_config"]
        if rest_joint_angles is None:
            rest_joint_angles = retract_cfg
        n_dof = len(rest_joint_angles)

        from curobo.types.robot import RobotConfig
        robot_cfg = RobotConfig.from_dict(cfg_data["robot_cfg"])

        # Try with collision world; fallback without
        world_cfg = None
        try:
            world_cfg = _build_world_cfg(scene)
        except Exception:
            pass

        try:
            mg_config = MotionGenConfig.load_from_robot_config(
                robot_cfg,
                world_model=world_cfg,
                num_ik_seeds=32,
                num_trajopt_seeds=4,
                interpolation_dt=interpolation_dt,
                use_cuda_graph=False,
                position_threshold=0.02,
                rotation_threshold=0.1,
            )
        except Exception:
            robot_cfg = RobotConfig.from_dict(cfg_data["robot_cfg"])
            mg_config = MotionGenConfig.load_from_robot_config(
                robot_cfg,
                world_model=None,
                num_ik_seeds=32,
                num_trajopt_seeds=4,
                interpolation_dt=interpolation_dt,
                use_cuda_graph=False,
                position_threshold=0.02,
                rotation_threshold=0.1,
            )

        motion_gen = MotionGen(mg_config)
        motion_gen.warmup(enable_graph=False, batch=1)

        # Start state
        device = motion_gen.tensor_args.device
        dtype = motion_gen.tensor_args.dtype
        start_q = torch.tensor(
            [rest_joint_angles], device=device, dtype=dtype,
        )
        start_state = CuJointState.from_position(start_q)

        # Goal pose
        w, x, y, z = wrist_quat_wxyz
        goal = Pose.from_list(
            [wrist_pos[0], wrist_pos[1], wrist_pos[2], w, x, y, z],
            tensor_args=motion_gen.tensor_args,
        )

        plan_cfg = MotionGenPlanConfig(
            enable_graph=False,
            enable_opt=True,
            max_attempts=30,
            timeout=10.0,
        )
        result = motion_gen.plan_single(start_state, goal, plan_cfg)

        if not result.success.item():
            return MotionGenResult(
                success=False, joint_trajectory=None,
                ee_pos_trajectory=None, ee_quat_trajectory=None,
                num_steps=0, motion_time=0.0,
                reason=f"MotionGen failed: {result.status}",
            )

        # Extract interpolated trajectory
        joint_traj = result.get_interpolated_plan()  # JointState [T, n_dof]
        joint_np = joint_traj.position.cpu().numpy()  # [T, 7]
        T = joint_np.shape[0]
        motion_time = float(result.motion_time) if hasattr(result, "motion_time") else T * interpolation_dt

        # FK to get EE trajectory
        ee_pos_list = None
        ee_quat_list = None
        try:
            kin_state = motion_gen.compute_kinematics(joint_traj)
            ee_pos_np = kin_state.ee_pos_seq.cpu().numpy()    # [T, 3]
            ee_quat_np = kin_state.ee_quat_seq.cpu().numpy()  # [T, 4]
            ee_pos_list = ee_pos_np.tolist()
            ee_quat_list = ee_quat_np.tolist()
        except Exception:
            pass

        return MotionGenResult(
            success=True,
            joint_trajectory=joint_np.tolist(),
            ee_pos_trajectory=ee_pos_list,
            ee_quat_trajectory=ee_quat_list,
            num_steps=T,
            motion_time=motion_time,
            reason="ok",
        )

    except Exception as exc:
        return MotionGenResult(
            success=False, joint_trajectory=None,
            ee_pos_trajectory=None, ee_quat_trajectory=None,
            num_steps=0, motion_time=0.0,
            reason=f"{type(exc).__name__}: {exc}",
        )


def _resolve_target_pose(req: PlannerRequest) -> tuple[np.ndarray, float]:
    start = np.asarray(req.start_base_pos_w, dtype=np.float64)
    obj_pos = np.asarray(req.object_pos_w, dtype=np.float64)
    obj_quat = np.asarray(req.object_quat_wxyz, dtype=np.float64)

    if req.target_base_pos_w is not None:
        target = np.asarray(req.target_base_pos_w, dtype=np.float64)
    else:
        offset = np.asarray(req.target_offset_obj_w, dtype=np.float64)
        if req.target_offset_frame == "object":
            delta = _quat_to_rotmat_wxyz(obj_quat) @ offset
        else:
            delta = offset
        target = obj_pos + delta
    target[2] = start[2]

    if req.target_yaw_mode == "face_object":
        yaw = math.atan2(obj_pos[1] - target[1], obj_pos[0] - target[0])
    elif req.target_yaw_mode == "base_yaw":
        yaw = float(req.start_base_yaw_rad)
    else:
        yaw = math.radians(float(req.target_yaw_deg))
    return target, float(yaw)


def _plan_open_loop_path(req: PlannerRequest, target_xy: tuple[float, float]) -> list[tuple[float, float]]:
    start_xy = (float(req.start_base_pos_w[0]), float(req.start_base_pos_w[1]))
    if req.scene not in {"kitchen_pick_and_place", "galileo_g1_locomanip_pick_and_place"}:
        return [start_xy, target_xy]

    obstacles = _scene_obstacles(req.scene)
    if not obstacles:
        return [start_xy, target_xy]

    blocked = any(_line_intersects_aabb_2d(start_xy, target_xy, bmin, bmax) for bmin, bmax in obstacles)
    if not blocked:
        return [start_xy, target_xy]

    # route around obstacle in Y with a conservative clearance.
    bmin, bmax = obstacles[0]
    clearance = 0.45
    if start_xy[1] >= 0.0:
        y_route = bmax[1] + clearance
    else:
        y_route = bmin[1] - clearance
    mid1 = (start_xy[0], y_route)
    mid2 = (target_xy[0], y_route)
    return [start_xy, mid1, mid2, target_xy]


def _waypoints_to_subgoals(
    waypoints_xy: list[tuple[float, float]],
    final_yaw_rad: float,
) -> list[tuple[list[float], bool]]:
    if len(waypoints_xy) < 2:
        return [([waypoints_xy[0][0], waypoints_xy[0][1], final_yaw_rad], True)]

    subgoals: list[tuple[list[float], bool]] = []
    for i in range(1, len(waypoints_xy)):
        prev_xy = waypoints_xy[i - 1]
        cur_xy = waypoints_xy[i]
        heading = math.atan2(cur_xy[1] - prev_xy[1], cur_xy[0] - prev_xy[0])
        subgoals.append(([float(cur_xy[0]), float(cur_xy[1]), float(heading)], False))
    # final in-place yaw align
    end_xy = waypoints_xy[-1]
    if abs(_wrap_angle_rad(final_yaw_rad - subgoals[-1][0][2])) > 1e-3:
        subgoals.append(([float(end_xy[0]), float(end_xy[1]), float(final_yaw_rad)], True))
    return subgoals


def plan_walk_to_grasp(req: PlannerRequest) -> PlannerResult:
    if req.planner not in {"auto", "curobo", "open_loop"}:
        raise ValueError(f"Unknown planner: {req.planner}")

    target_pos_w, target_yaw = _resolve_target_pose(req)
    target_xy = (float(target_pos_w[0]), float(target_pos_w[1]))

    curobo_available = False
    curobo_reason = "not requested"
    planner_used = "open_loop"
    notes = ""

    if req.planner in {"auto", "curobo"}:
        curobo_available, curobo_reason = _probe_curobo()
        if req.planner == "curobo" and req.strict_curobo and not curobo_available:
            raise RuntimeError(
                "Strict CuRobo mode requested, but CuRobo is not available in this workspace. "
                f"Probe result: {curobo_reason}"
            )
        if curobo_available:
            notes = "CuRobo available. IK reachability check enabled."
        else:
            notes = f"CuRobo unavailable, fallback to open-loop path: {curobo_reason}"

    # IK reachability check when CuRobo is available and wrist pose is provided
    ik_result = None
    if curobo_available and hasattr(req, "wrist_pos_w") and req.wrist_pos_w is not None:
        ik_result = _ik_check_reachability(
            req.wrist_pos_w, req.wrist_quat_wxyz_ik, scene=req.scene,
        )
        if ik_result.reachable:
            notes += " IK check: REACHABLE."
        else:
            notes += f" IK check: UNREACHABLE ({ik_result.reason})."

    # MotionGen trajectory planning when IK is reachable
    mg_result = None
    if curobo_available and ik_result is not None and ik_result.reachable and req.wrist_pos_w is not None:
        mg_result = plan_arm_trajectory(
            req.wrist_pos_w, req.wrist_quat_wxyz_ik, scene=req.scene,
        )
        if mg_result.success:
            planner_used = "curobo"
            notes += f" MotionGen: OK ({mg_result.num_steps} steps, {mg_result.motion_time:.3f}s)."
        else:
            notes += f" MotionGen: FAILED ({mg_result.reason}), fallback to open-loop."

    waypoints_xy = _plan_open_loop_path(req, target_xy)
    subgoals = _waypoints_to_subgoals(waypoints_xy, target_yaw)

    return PlannerResult(
        planner_requested=req.planner,
        planner_used=planner_used,
        curobo_available=curobo_available,
        curobo_reason=curobo_reason,
        target_base_pos_w=(float(target_pos_w[0]), float(target_pos_w[1]), float(target_pos_w[2])),
        target_base_yaw_rad=float(target_yaw),
        path_waypoints_xy=[(float(x), float(y)) for x, y in waypoints_xy],
        navigation_subgoals=subgoals,
        notes=notes,
        ik_result=ik_result,
        motion_gen_result=mg_result,
    )


def request_from_args(args: Any) -> PlannerRequest:
    explicit_target = None
    if getattr(args, "walk_target_base_pos_w", None):
        explicit_target_v = _parse_csv_floats(args.walk_target_base_pos_w, 3, "walk_target_base_pos_w")
        explicit_target = (float(explicit_target_v[0]), float(explicit_target_v[1]), float(explicit_target_v[2]))
    start_pos = _parse_csv_floats(args.base_pos_w, 3, "base_pos_w")
    obj_pos = _parse_csv_floats(args.object_pos_w, 3, "object_pos_w")
    obj_quat = _parse_csv_floats(args.object_quat_wxyz, 4, "object_quat_wxyz")
    target_offset = _parse_csv_floats(args.walk_target_offset_obj_w, 3, "walk_target_offset_obj_w")
    return PlannerRequest(
        planner=str(args.planner),
        strict_curobo=bool(args.strict_curobo),
        scene=str(args.scene),
        start_base_pos_w=(float(start_pos[0]), float(start_pos[1]), float(start_pos[2])),
        start_base_yaw_rad=math.radians(float(args.base_yaw_deg)),
        object_pos_w=(float(obj_pos[0]), float(obj_pos[1]), float(obj_pos[2])),
        object_quat_wxyz=(float(obj_quat[0]), float(obj_quat[1]), float(obj_quat[2]), float(obj_quat[3])),
        target_base_pos_w=explicit_target,
        target_offset_obj_w=(float(target_offset[0]), float(target_offset[1]), float(target_offset[2])),
        target_offset_frame=str(args.walk_target_offset_frame),
        target_yaw_mode=str(args.walk_target_yaw_mode),
        target_yaw_deg=float(args.walk_target_yaw_deg),
    )
