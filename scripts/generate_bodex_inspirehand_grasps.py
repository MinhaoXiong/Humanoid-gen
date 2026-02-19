#!/usr/bin/env python3
"""Generate InspireHand grasps using BODex GraspSolver."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime

import torch
import numpy as np

# Mimic joint mapping: independent_joint -> [(mimic_joint, multiplier, offset), ...]
INSPIRE_MIMIC_MAP = {
    "R_thumb_proximal_pitch_joint": [
        ("R_thumb_intermediate_joint", 1.6, 0.0),
        ("R_thumb_distal_joint", 2.4, 0.0),
    ],
    "R_index_proximal_joint": [("R_index_intermediate_joint", 1.0, 0.0)],
    "R_middle_proximal_joint": [("R_middle_intermediate_joint", 1.0, 0.0)],
    "R_ring_proximal_joint": [("R_ring_intermediate_joint", 1.0, 0.0)],
    "R_pinky_proximal_joint": [("R_pinky_intermediate_joint", 1.0, 0.0)],
}

INDEPENDENT_JOINTS = [
    "R_thumb_proximal_yaw_joint",
    "R_thumb_proximal_pitch_joint",
    "R_index_proximal_joint",
    "R_middle_proximal_joint",
    "R_ring_proximal_joint",
    "R_pinky_proximal_joint",
]

ALL_JOINTS_ORDER = [
    "R_thumb_proximal_yaw_joint",
    "R_thumb_proximal_pitch_joint",
    "R_thumb_intermediate_joint",
    "R_thumb_distal_joint",
    "R_index_proximal_joint",
    "R_index_intermediate_joint",
    "R_middle_proximal_joint",
    "R_middle_intermediate_joint",
    "R_ring_proximal_joint",
    "R_ring_intermediate_joint",
    "R_pinky_proximal_joint",
    "R_pinky_intermediate_joint",
]


def expand_mimic_joints(independent_q: torch.Tensor) -> torch.Tensor:
    """From 6 independent DOF, compute all 12 joint values."""
    batch = independent_q.shape[0]
    full_q = torch.zeros(batch, len(ALL_JOINTS_ORDER), device=independent_q.device)
    for i, jname in enumerate(INDEPENDENT_JOINTS):
        idx = ALL_JOINTS_ORDER.index(jname)
        full_q[:, idx] = independent_q[:, i]
        if jname in INSPIRE_MIMIC_MAP:
            for mimic_name, mult, offset in INSPIRE_MIMIC_MAP[jname]:
                midx = ALL_JOINTS_ORDER.index(mimic_name)
                full_q[:, midx] = independent_q[:, i] * mult + offset
    return full_q


def _ensure_bodex_importable():
    bodex_src = os.environ.get(
        "BODEX_CUROBO_SRC",
        os.path.join(os.path.dirname(__file__), "..", "..", "BODex", "src"),
    )
    bodex_src = os.path.abspath(bodex_src)
    if os.path.isdir(bodex_src) and bodex_src not in sys.path:
        sys.path.insert(0, bodex_src)
    import types
    if "setuptools_scm" not in sys.modules:
        stub = types.ModuleType("setuptools_scm")
        stub.get_version = lambda **_: "v0"
        sys.modules["setuptools_scm"] = stub


def _make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mesh-file", required=True, help="Object mesh (.obj/.stl) path.")
    p.add_argument(
        "--manip-cfg",
        default=os.path.join(os.path.dirname(__file__), "..", "configs", "bodex_inspire_hand_grasp.yml"),
    )
    p.add_argument("--parallel-world", type=int, default=1)
    p.add_argument("--output-pt", default=None)
    p.add_argument("--top-k", type=int, default=16)
    return p


def main() -> None:
    args = _make_parser().parse_args()
    _ensure_bodex_importable()

    from curobo.geom.sdf.world import WorldConfig
    from curobo.wrap.reacher.grasp_solver import GraspSolver, GraspSolverConfig
    from curobo.util_file import load_yaml, join_path, get_manip_configs_path

    manip_cfg_path = os.path.abspath(args.manip_cfg)
    manip_config_data = load_yaml(manip_cfg_path)

    mesh_path = os.path.abspath(args.mesh_file)
    obj_name = os.path.splitext(os.path.basename(mesh_path))[0]

    # Build a single-object floating world config
    world_cfg = WorldConfig.from_dict(
        {
            "mesh": {
                obj_name: {
                    "pose": [0, 0, 0, 1, 0, 0, 0],
                    "file_path": mesh_path,
                }
            }
        }
    )

    print(f"[bodex_grasp] Loading GraspSolver for {obj_name} ...")
    t0 = time.time()
    grasp_config = GraspSolverConfig.load_from_robot_config(
        world_model=[world_cfg],
        manip_name_list=[obj_name],
        manip_config_data=manip_config_data,
        obj_gravity_center=[[0.0, 0.0, 0.0]],
        obj_obb_length=[[0.1, 0.1, 0.1]],
        use_cuda_graph=False,
    )
    grasp_solver = GraspSolver(grasp_config)
    print(f"[bodex_grasp] Solver ready ({time.time() - t0:.1f}s). Solving ...")

    result = grasp_solver.solve_batch_env(return_seeds=grasp_solver.num_seeds)

    # Extract solutions: shape [n_env, n_seeds, horizon, dof]
    # solution[:, :, -1, :] is the final pose: first 7 = root pose, rest = joint angles
    sol = result.solution  # [n_env, n_seeds, horizon, 7+6]
    final_sol = sol[:, :, -1, :]  # [n_env, n_seeds, 13]
    grasp_err = result.grasp_error  # [n_env, n_seeds]
    dist_err = result.dist_error

    # Flatten envs
    final_sol = final_sol.reshape(-1, final_sol.shape[-1])
    grasp_err = grasp_err.reshape(-1)
    dist_err = dist_err.reshape(-1)

    # Sort by grasp error, take top-k
    top_k = min(args.top_k, final_sol.shape[0])
    sorted_idx = torch.argsort(grasp_err)[:top_k]

    wrist_pose_7d = final_sol[sorted_idx, :7]  # [K, 7]
    independent_q = final_sol[sorted_idx, 7:]   # [K, 6]
    full_q_12 = expand_mimic_joints(independent_q)  # [K, 12]

    output_data = {
        "object_name": obj_name,
        "mesh_path": mesh_path,
        "wrist_pose_7d": wrist_pose_7d.cpu(),
        "joint_angles_12": full_q_12.cpu(),
        "independent_q_6": independent_q.cpu(),
        "grasp_error": grasp_err[sorted_idx].cpu(),
        "dist_error": dist_err[sorted_idx].cpu(),
        "joint_names_12": ALL_JOINTS_ORDER,
        "independent_joint_names": INDEPENDENT_JOINTS,
    }

    if args.output_pt:
        out_path = os.path.abspath(args.output_pt)
    else:
        pack_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        out_dir = os.path.join(pack_root, "artifacts", "bodex_inspire_grasps")
        os.makedirs(out_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(out_dir, f"{obj_name}_top{top_k}_{ts}.pt")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    torch.save(output_data, out_path)
    print(f"[bodex_grasp] Saved {top_k} grasps -> {out_path}")
    print(f"[bodex_grasp] Best grasp_error={float(grasp_err[sorted_idx[0]]):.6f}")


if __name__ == "__main__":
    main()
