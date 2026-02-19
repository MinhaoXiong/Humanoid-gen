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

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


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
            # Workspace currently lacks a validated G1 CuRobo robot config + world export bridge.
            # We keep route planning deterministic and report fallback explicitly.
            notes = (
                "CuRobo import is available, but no validated G1 robot/world planning config was found. "
                "Used deterministic open-loop path fallback."
            )
        else:
            notes = f"CuRobo unavailable, fallback to open-loop path: {curobo_reason}"

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
