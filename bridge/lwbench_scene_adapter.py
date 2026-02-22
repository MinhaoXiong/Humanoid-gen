#!/usr/bin/env python3
"""LW-BenchHub scene adapter: extract scene geometry from LW-BenchHub kitchen
scenes and generate SceneConfig + CuRobo collision cuboids for the
walk-to-grasp pipeline.

Usage:
    # Extract scene config from a specific layout/style
    python bridge/lwbench_scene_adapter.py \
        --layout-id 0 --style-id 0 \
        --output-json artifacts/lwbench_scene_0_0.json

    # Extract from a local USD file
    python bridge/lwbench_scene_adapter.py \
        --local-usd /path/to/kitchen.usd \
        --output-json artifacts/lwbench_scene_custom.json

    # List available layouts
    python bridge/lwbench_scene_adapter.py --list-layouts
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

_SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
_PACK_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
if _PACK_ROOT not in sys.path:
    sys.path.insert(0, _PACK_ROOT)

# LW-BenchHub paths
_LWBENCH_ROOT = os.path.abspath(os.path.join(_PACK_ROOT, "..", "LW-BenchHub"))
if _LWBENCH_ROOT not in sys.path:
    sys.path.insert(0, _LWBENCH_ROOT)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class CollisionCuboid:
    """CuRobo-compatible 3D collision cuboid."""
    name: str
    pose: list[float]   # [x, y, z, qw, qx, qy, qz]
    dims: list[float]   # [dx, dy, dz]


@dataclass
class Obstacle2D:
    """2D AABB obstacle for walk path planning."""
    aabb_min: list[float]  # [x_min, y_min]
    aabb_max: list[float]  # [x_max, y_max]


@dataclass
class LwBenchSceneInfo:
    """Complete scene information extracted from LW-BenchHub."""
    scene_name: str
    layout_id: int | None
    style_id: int | None
    usd_path: str
    # Scene geometry
    scene_range_min: list[float]  # [x, y, z]
    scene_range_max: list[float]  # [x, y, z]
    # Counter/table info
    counter_top_z: float
    counter_center_xy: list[float]  # [x, y]
    counter_dims_xy: list[float]    # [dx, dy]
    # Robot placement
    robot_init_pos: list[float]     # [x, y, z]
    robot_init_yaw_deg: float
    # Walk target
    walk_target_offset: list[float]  # [dx, dy, dz] relative to object
    # Object placement
    object_align_pos: list[float]   # [x, y, z]
    # Collision
    collision_cuboids: list[CollisionCuboid] = field(default_factory=list)
    obstacles_2d: list[Obstacle2D] = field(default_factory=list)
    # Fixtures found
    fixture_names: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d


# ---------------------------------------------------------------------------
# USD geometry extraction (works without Isaac Sim running)
# ---------------------------------------------------------------------------
def _extract_aabb_from_usd(stage, prim_path: str) -> tuple[np.ndarray, np.ndarray] | None:
    """Extract axis-aligned bounding box from a USD prim.

    Returns (aabb_min, aabb_max) as numpy arrays, or None if not computable.
    """
    try:
        from pxr import UsdGeom, Gf
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            return None
        bbox_cache = UsdGeom.BBoxCache(0.0, [UsdGeom.Tokens.default_])
        bbox = bbox_cache.ComputeWorldBound(prim)
        box_range = bbox.ComputeAlignedRange()
        if box_range.IsEmpty():
            return None
        bmin = np.array(box_range.GetMin(), dtype=np.float64)
        bmax = np.array(box_range.GetMax(), dtype=np.float64)
        return bmin, bmax
    except Exception:
        return None


def _extract_all_fixture_aabbs(stage) -> list[dict]:
    """Walk the USD stage and extract AABBs for all Xform prims at depth 1-2."""
    from pxr import UsdGeom, Usd
    results = []
    root = stage.GetPseudoRoot()
    for child in root.GetChildren():
        if not child.IsA(UsdGeom.Xform):
            continue
        name = child.GetName()
        aabb = _extract_aabb_from_usd(stage, str(child.GetPath()))
        if aabb is not None:
            bmin, bmax = aabb
            results.append({
                "name": name,
                "prim_path": str(child.GetPath()),
                "aabb_min": bmin.tolist(),
                "aabb_max": bmax.tolist(),
                "center": ((bmin + bmax) / 2).tolist(),
                "dims": (bmax - bmin).tolist(),
            })
        # Also check direct children for nested fixtures
        for grandchild in child.GetChildren():
            if not grandchild.IsA(UsdGeom.Xform):
                continue
            gc_name = grandchild.GetName()
            gc_aabb = _extract_aabb_from_usd(stage, str(grandchild.GetPath()))
            if gc_aabb is not None:
                gc_bmin, gc_bmax = gc_aabb
                results.append({
                    "name": f"{name}/{gc_name}",
                    "prim_path": str(grandchild.GetPath()),
                    "aabb_min": gc_bmin.tolist(),
                    "aabb_max": gc_bmax.tolist(),
                    "center": ((gc_bmin + gc_bmax) / 2).tolist(),
                    "dims": (gc_bmax - gc_bmin).tolist(),
                })
    return results


def _is_counter_fixture(name: str) -> bool:
    """Heuristic: check if a fixture name looks like a counter/table."""
    lower = name.lower()
    keywords = ["counter", "table", "desk", "countertop", "island", "workspace"]
    return any(kw in lower for kw in keywords)


def _is_large_obstacle(name: str) -> bool:
    """Heuristic: check if a fixture is a large obstacle for walking."""
    lower = name.lower()
    keywords = [
        "counter", "table", "cabinet", "fridge", "refrigerator",
        "oven", "stove", "dishwasher", "island", "sink",
    ]
    return any(kw in lower for kw in keywords)


# ---------------------------------------------------------------------------
# Scene extraction via LW-BenchHub SDK
# ---------------------------------------------------------------------------
def extract_scene_via_sdk(
    layout_id: int | None = None,
    style_id: int | None = None,
    local_usd: str | None = None,
    robot_standoff: float = 0.65,
) -> LwBenchSceneInfo:
    """Extract scene info using LW-BenchHub's KitchenArena.

    Args:
        layout_id: Kitchen layout ID (0-9).
        style_id: Kitchen style ID (0-9).
        local_usd: Path to a local USD file (overrides layout/style).
        robot_standoff: Distance from counter edge to robot base (meters).

    Returns:
        LwBenchSceneInfo with all extracted geometry.
    """
    from lw_benchhub.core.scenes.kitchen.kitchen_arena import KitchenArena

    arena = KitchenArena(
        layout_id=layout_id,
        style_id=style_id,
        local_scene_path=local_usd,
    )

    scene_range = arena.scene_range  # [[xmin,ymin,zmin], [xmax,ymax,zmax]]
    usd_path = arena.usd_path
    actual_layout = arena.layout_id
    actual_style = arena.style_id

    scene_name = f"lwbench_kitchen_{actual_layout}_{actual_style}"
    if local_usd:
        base = os.path.splitext(os.path.basename(local_usd))[0]
        scene_name = f"lwbench_custom_{base}"

    # Extract fixture AABBs from USD
    fixture_aabbs = _extract_all_fixture_aabbs(arena.stage)

    # Find counter/table fixtures
    counters = [f for f in fixture_aabbs if _is_counter_fixture(f["name"])]

    # If no counter found, use the largest horizontal surface
    if not counters:
        # Heuristic: find fixtures with large XY extent and moderate Z
        candidates = [
            f for f in fixture_aabbs
            if f["dims"][0] > 0.3 and f["dims"][1] > 0.3 and f["dims"][2] < 1.5
        ]
        if candidates:
            candidates.sort(key=lambda f: f["dims"][0] * f["dims"][1], reverse=True)
            counters = [candidates[0]]

    # Compute counter geometry
    if counters:
        # Use the largest counter
        counters.sort(key=lambda f: f["dims"][0] * f["dims"][1], reverse=True)
        main_counter = counters[0]
        counter_top_z = float(main_counter["aabb_max"][2])
        counter_cx = float(main_counter["center"][0])
        counter_cy = float(main_counter["center"][1])
        counter_dx = float(main_counter["dims"][0])
        counter_dy = float(main_counter["dims"][1])
    else:
        # Fallback: use scene center at a reasonable height
        counter_top_z = 0.85
        counter_cx = float((scene_range[0][0] + scene_range[1][0]) / 2)
        counter_cy = float((scene_range[0][1] + scene_range[1][1]) / 2)
        counter_dx = 0.8
        counter_dy = 1.0

    # Robot placement: stand in front of counter (negative X direction from counter)
    # Determine which side of the counter has more free space
    scene_cx = float((scene_range[0][0] + scene_range[1][0]) / 2)
    if counter_cx > scene_cx:
        # Counter is on the +X side, robot approaches from -X
        robot_x = counter_cx - counter_dx / 2 - robot_standoff
        robot_yaw = 0.0  # Face +X toward counter
    else:
        # Counter is on the -X side, robot approaches from +X
        robot_x = counter_cx + counter_dx / 2 + robot_standoff
        robot_yaw = 180.0  # Face -X toward counter

    robot_y = counter_cy
    robot_z = 0.0

    # Object placement: center of counter top
    obj_x = counter_cx
    obj_y = counter_cy
    obj_z = counter_top_z

    # Walk target offset: from object toward robot
    walk_dx = -0.35 if robot_yaw == 0.0 else 0.35
    walk_dy = 0.0
    walk_dz = 0.0

    # Build collision cuboids
    collision_cuboids = []
    obstacles_2d = []

    for fxtr in fixture_aabbs:
        if not _is_large_obstacle(fxtr["name"]):
            continue
        center = fxtr["center"]
        dims = fxtr["dims"]
        # CuRobo cuboid: pose = [x, y, z, qw, qx, qy, qz]
        cuboid = CollisionCuboid(
            name=fxtr["name"],
            pose=[center[0], center[1], center[2], 1.0, 0.0, 0.0, 0.0],
            dims=[dims[0], dims[1], dims[2]],
        )
        collision_cuboids.append(cuboid)

        # 2D obstacle for walk planning
        obstacle = Obstacle2D(
            aabb_min=[fxtr["aabb_min"][0], fxtr["aabb_min"][1]],
            aabb_max=[fxtr["aabb_max"][0], fxtr["aabb_max"][1]],
        )
        obstacles_2d.append(obstacle)

    fixture_names = [f["name"] for f in fixture_aabbs]

    return LwBenchSceneInfo(
        scene_name=scene_name,
        layout_id=actual_layout,
        style_id=actual_style,
        usd_path=usd_path,
        scene_range_min=scene_range[0].tolist(),
        scene_range_max=scene_range[1].tolist(),
        counter_top_z=counter_top_z,
        counter_center_xy=[counter_cx, counter_cy],
        counter_dims_xy=[counter_dx, counter_dy],
        robot_init_pos=[robot_x, robot_y, robot_z],
        robot_init_yaw_deg=robot_yaw,
        walk_target_offset=[walk_dx, walk_dy, walk_dz],
        object_align_pos=[obj_x, obj_y, obj_z],
        collision_cuboids=collision_cuboids,
        obstacles_2d=obstacles_2d,
        fixture_names=fixture_names,
    )


# ---------------------------------------------------------------------------
# Standalone USD extraction (no LW-BenchHub SDK needed)
# ---------------------------------------------------------------------------
def extract_scene_from_usd_standalone(
    usd_path: str,
    counter_top_z: float | None = None,
    robot_standoff: float = 0.65,
) -> LwBenchSceneInfo:
    """Extract scene info directly from a USD file without LW-BenchHub SDK.

    This is a fallback for when the Lightwheel SDK is not available.
    Requires pxr (OpenUSD) to be importable.
    """
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.Open(usd_path)
    fixture_aabbs = _extract_all_fixture_aabbs(stage)

    # Compute scene range
    all_mins = []
    all_maxs = []
    for f in fixture_aabbs:
        all_mins.append(f["aabb_min"])
        all_maxs.append(f["aabb_max"])

    if all_mins:
        scene_min = np.min(all_mins, axis=0).tolist()
        scene_max = np.max(all_maxs, axis=0).tolist()
    else:
        scene_min = [0.0, 0.0, 0.0]
        scene_max = [2.0, 2.0, 2.0]

    # Find counters
    counters = [f for f in fixture_aabbs if _is_counter_fixture(f["name"])]
    if not counters:
        candidates = [
            f for f in fixture_aabbs
            if f["dims"][0] > 0.3 and f["dims"][1] > 0.3 and f["dims"][2] < 1.5
        ]
        if candidates:
            candidates.sort(key=lambda f: f["dims"][0] * f["dims"][1], reverse=True)
            counters = [candidates[0]]

    if counters:
        counters.sort(key=lambda f: f["dims"][0] * f["dims"][1], reverse=True)
        main_counter = counters[0]
        ct_z = counter_top_z or float(main_counter["aabb_max"][2])
        ct_cx = float(main_counter["center"][0])
        ct_cy = float(main_counter["center"][1])
        ct_dx = float(main_counter["dims"][0])
        ct_dy = float(main_counter["dims"][1])
    else:
        ct_z = counter_top_z or 0.85
        ct_cx = float((scene_min[0] + scene_max[0]) / 2)
        ct_cy = float((scene_min[1] + scene_max[1]) / 2)
        ct_dx = 0.8
        ct_dy = 1.0

    scene_cx = float((scene_min[0] + scene_max[0]) / 2)
    if ct_cx > scene_cx:
        robot_x = ct_cx - ct_dx / 2 - robot_standoff
        robot_yaw = 0.0
    else:
        robot_x = ct_cx + ct_dx / 2 + robot_standoff
        robot_yaw = 180.0

    walk_dx = -0.35 if robot_yaw == 0.0 else 0.35

    collision_cuboids = []
    obstacles_2d = []
    for fxtr in fixture_aabbs:
        if not _is_large_obstacle(fxtr["name"]):
            continue
        collision_cuboids.append(CollisionCuboid(
            name=fxtr["name"],
            pose=[fxtr["center"][0], fxtr["center"][1], fxtr["center"][2], 1.0, 0.0, 0.0, 0.0],
            dims=fxtr["dims"],
        ))
        obstacles_2d.append(Obstacle2D(
            aabb_min=[fxtr["aabb_min"][0], fxtr["aabb_min"][1]],
            aabb_max=[fxtr["aabb_max"][0], fxtr["aabb_max"][1]],
        ))

    base_name = os.path.splitext(os.path.basename(usd_path))[0]

    return LwBenchSceneInfo(
        scene_name=f"lwbench_custom_{base_name}",
        layout_id=None,
        style_id=None,
        usd_path=usd_path,
        scene_range_min=scene_min,
        scene_range_max=scene_max,
        counter_top_z=ct_z,
        counter_center_xy=[ct_cx, ct_cy],
        counter_dims_xy=[ct_dx, ct_dy],
        robot_init_pos=[robot_x, ct_cy, 0.0],
        robot_init_yaw_deg=robot_yaw,
        walk_target_offset=[walk_dx, 0.0, 0.0],
        object_align_pos=[ct_cx, ct_cy, ct_z],
        collision_cuboids=collision_cuboids,
        obstacles_2d=obstacles_2d,
        fixture_names=[f["name"] for f in fixture_aabbs],
    )


# ---------------------------------------------------------------------------
# Mock scene generation (for testing without USD/SDK)
# ---------------------------------------------------------------------------
def generate_mock_kitchen_scene(
    layout_id: int = 0,
    style_id: int = 0,
    counter_top_z: float = 0.85,
    counter_center_xy: tuple[float, float] = (0.50, 0.0),
    counter_dims_xy: tuple[float, float] = (0.80, 1.20),
    robot_standoff: float = 0.65,
) -> LwBenchSceneInfo:
    """Generate a mock kitchen scene config for testing without USD or SDK.

    Uses realistic kitchen counter dimensions based on LW-BenchHub defaults.
    """
    scene_name = f"lwbench_kitchen_{layout_id}_{style_id}"
    ct_cx, ct_cy = counter_center_xy
    ct_dx, ct_dy = counter_dims_xy

    robot_x = ct_cx - ct_dx / 2 - robot_standoff
    robot_yaw = 0.0

    counter_cuboid = CollisionCuboid(
        name="counter_0",
        pose=[ct_cx, ct_cy, counter_top_z / 2, 1.0, 0.0, 0.0, 0.0],
        dims=[ct_dx, ct_dy, counter_top_z],
    )
    counter_obstacle = Obstacle2D(
        aabb_min=[ct_cx - ct_dx / 2, ct_cy - ct_dy / 2],
        aabb_max=[ct_cx + ct_dx / 2, ct_cy + ct_dy / 2],
    )

    return LwBenchSceneInfo(
        scene_name=scene_name,
        layout_id=layout_id,
        style_id=style_id,
        usd_path="mock://kitchen_scene",
        scene_range_min=[-2.0, -2.0, 0.0],
        scene_range_max=[2.0, 2.0, 2.5],
        counter_top_z=counter_top_z,
        counter_center_xy=[ct_cx, ct_cy],
        counter_dims_xy=[ct_dx, ct_dy],
        robot_init_pos=[robot_x, ct_cy, 0.0],
        robot_init_yaw_deg=robot_yaw,
        walk_target_offset=[-0.35, 0.0, 0.0],
        object_align_pos=[ct_cx, ct_cy, counter_top_z],
        collision_cuboids=[counter_cuboid],
        obstacles_2d=[counter_obstacle],
        fixture_names=["counter_0"],
    )


# ---------------------------------------------------------------------------
# Generate SceneConfig dict for bridge/scene_config.py
# ---------------------------------------------------------------------------
def scene_info_to_scene_config_dict(info: LwBenchSceneInfo) -> dict[str, Any]:
    """Convert LwBenchSceneInfo to a dict compatible with SceneConfig fields."""
    return {
        "arena_scene_name": info.scene_name,
        "arena_background": info.scene_name,
        "table_z": info.counter_top_z,
        "object_align_pos": tuple(info.object_align_pos),
        "clip_z": (0.05, 0.40),
        "clip_xy_min": (
            info.counter_center_xy[0] - info.counter_dims_xy[0] / 2,
            info.counter_center_xy[1] - info.counter_dims_xy[1] / 2,
        ),
        "clip_xy_max": (
            info.counter_center_xy[0] + info.counter_dims_xy[0] / 2,
            info.counter_center_xy[1] + info.counter_dims_xy[1] / 2,
        ),
        "base_height": 0.75,
        "default_base_pos_w": tuple(info.robot_init_pos),
        "default_base_yaw_deg": info.robot_init_yaw_deg,
        "default_walk_target_offset": tuple(info.walk_target_offset),
        "right_wrist_pos_obj": (-0.18, -0.04, 0.08),
        "replay_base_height": 0.78,
    }


def scene_info_to_collision_dicts(info: LwBenchSceneInfo) -> dict[str, Any]:
    """Convert LwBenchSceneInfo to collision config dicts for CuRobo planner."""
    return {
        "collision_cuboids": [asdict(c) for c in info.collision_cuboids],
        "obstacles_2d": [
            [obs.aabb_min, obs.aabb_max] for obs in info.obstacles_2d
        ],
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--layout-id", type=int, default=None, help="Kitchen layout ID (0-9)")
    parser.add_argument("--style-id", type=int, default=None, help="Kitchen style ID (0-9)")
    parser.add_argument("--local-usd", default=None, help="Path to local USD file (overrides layout/style)")
    parser.add_argument("--output-json", required=True, help="Output JSON path for scene config")
    parser.add_argument("--robot-standoff", type=float, default=0.65, help="Distance from counter to robot (m)")
    parser.add_argument("--counter-top-z", type=float, default=None, help="Override counter top Z height")
    parser.add_argument("--standalone", action="store_true", help="Use standalone USD extraction (no LW-BenchHub SDK)")
    parser.add_argument("--mock", action="store_true", help="Generate mock scene config for testing (no USD/SDK needed)")
    return parser


def main() -> None:
    args = _make_parser().parse_args()

    if args.mock:
        info = generate_mock_kitchen_scene(
            layout_id=args.layout_id or 0,
            style_id=args.style_id or 0,
            counter_top_z=args.counter_top_z or 0.85,
            robot_standoff=args.robot_standoff,
        )
    elif args.standalone or args.local_usd:
        usd_path = args.local_usd
        if not usd_path:
            print("Error: --local-usd required when using --standalone mode")
            sys.exit(1)
        info = extract_scene_from_usd_standalone(
            usd_path=usd_path,
            counter_top_z=args.counter_top_z,
            robot_standoff=args.robot_standoff,
        )
    else:
        info = extract_scene_via_sdk(
            layout_id=args.layout_id,
            style_id=args.style_id,
            robot_standoff=args.robot_standoff,
        )

    # Build output
    output = {
        "scene_info": info.to_dict(),
        "scene_config": scene_info_to_scene_config_dict(info),
        "collision": scene_info_to_collision_dicts(info),
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"[lwbench_adapter] Scene: {info.scene_name}")
    print(f"[lwbench_adapter] USD: {info.usd_path}")
    print(f"[lwbench_adapter] Counter top Z: {info.counter_top_z:.3f}")
    print(f"[lwbench_adapter] Counter center: ({info.counter_center_xy[0]:.2f}, {info.counter_center_xy[1]:.2f})")
    print(f"[lwbench_adapter] Robot init: ({info.robot_init_pos[0]:.2f}, {info.robot_init_pos[1]:.2f}, {info.robot_init_pos[2]:.2f}) yaw={info.robot_init_yaw_deg:.0f}°")
    print(f"[lwbench_adapter] Collision cuboids: {len(info.collision_cuboids)}")
    print(f"[lwbench_adapter] 2D obstacles: {len(info.obstacles_2d)}")
    print(f"[lwbench_adapter] Fixtures found: {len(info.fixture_names)}")
    print(f"[lwbench_adapter] Output: {args.output_json}")


if __name__ == "__main__":
    main()
