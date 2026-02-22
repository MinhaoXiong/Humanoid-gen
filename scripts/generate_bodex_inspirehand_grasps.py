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
from scipy.spatial.transform import Rotation

# Spider IK finger indices -> BODex 6 independent DOF
# Spider order: [thumb_yaw, thumb_pitch, thumb_inter, thumb_distal,
#                index_prox, index_inter, middle_prox, middle_inter,
#                ring_prox, ring_inter, pinky_prox, pinky_inter]
SPIDER_TO_BODEX_6 = [0, 1, 4, 6, 8, 10]

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


def load_spider_seed(npz_path: str, frame: int = -1) -> torch.Tensor:
    """Load spider IK output and convert to BODex seed_config [1, 1, 13].

    BODex seed format: [x, y, z, qw, qx, qy, qz, 6_independent_dof]
    Spider qpos format: [wrist_xyz(3), wrist_euler_xyz(3), finger_12dof, object_7dof]
    """
    data = np.load(npz_path)
    qpos = data["qpos"][frame]  # (25,)
    wrist_pos = qpos[:3]
    wrist_euler = qpos[3:6]
    finger_12 = qpos[6:18]

    # Euler XYZ -> quaternion (wxyz for BODex)
    quat_xyzw = Rotation.from_euler("xyz", wrist_euler).as_quat()
    quat_wxyz = [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]]

    # Extract 6 independent DOF from 12 spider joints
    independent_6 = finger_12[SPIDER_TO_BODEX_6]

    seed = np.concatenate([wrist_pos, quat_wxyz, independent_6])
    return torch.tensor(seed, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # [1,1,13]


def rank_by_human_distance(
    solutions: torch.Tensor,
    human_wrist_pose_7d: torch.Tensor,
    human_finger_6: torch.Tensor,
    lambda_rot: float = 1.0,
    lambda_finger: float = 0.5,
) -> torch.Tensor:
    """Rank BODex results by distance to human hand pose.

    d = ||t_robot - t_human||_2 + lambda_rot * arccos(|q_robot . q_human|) + lambda_finger * ||q_finger_diff||

    Args:
        solutions: [K, 13] (7 root pose + 6 finger DOF)
        human_wrist_pose_7d: [7] (x,y,z,qw,qx,qy,qz)
        human_finger_6: [6] independent finger DOF
        lambda_rot: weight for rotation distance
        lambda_finger: weight for finger distance

    Returns:
        distances: [K] lower is closer to human
    """
    pos_dist = torch.norm(solutions[:, :3] - human_wrist_pose_7d[:3], dim=-1)

    q_robot = solutions[:, 3:7]
    q_human = human_wrist_pose_7d[3:7]
    dot = torch.abs((q_robot * q_human).sum(dim=-1)).clamp(max=1.0)
    rot_dist = torch.acos(dot)

    finger_dist = torch.norm(solutions[:, 7:] - human_finger_6, dim=-1)

    return pos_dist + lambda_rot * rot_dist + lambda_finger * finger_dist


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
    p.add_argument("--seed-from-spider", default=None,
                   help="Spider IK output .npz to use as seed_config")
    p.add_argument("--seed-frame", type=int, default=-1,
                   help="Frame index from spider trajectory to use as seed")
    p.add_argument("--rank-by-human", action="store_true",
                   help="Rank results by distance to human hand pose (from spider)")
    p.add_argument("--lambda-rot", type=float, default=1.0)
    p.add_argument("--lambda-finger", type=float, default=0.5)
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

    # Module B: load spider seed if provided
    seed_config = None
    human_seed_13 = None
    if args.seed_from_spider:
        spider_seed = load_spider_seed(args.seed_from_spider, args.seed_frame)
        seed_config = spider_seed.cuda()
        human_seed_13 = spider_seed.squeeze()  # [13] for ranking
        print(f"[bodex_grasp] Using spider seed from {args.seed_from_spider} frame={args.seed_frame}")

    result = grasp_solver.solve_batch_env(
        return_seeds=grasp_solver.num_seeds,
        seed_config=seed_config,
    )

    sol = result.solution
    final_sol = sol[:, :, -1, :]
    grasp_err = result.grasp_error
    dist_err = result.dist_error

    final_sol = final_sol.reshape(-1, final_sol.shape[-1])
    grasp_err = grasp_err.reshape(-1)
    dist_err = dist_err.reshape(-1)

    # Module C: rank by human distance or grasp error
    top_k = min(args.top_k, final_sol.shape[0])
    if args.rank_by_human and human_seed_13 is not None:
        human_dist = rank_by_human_distance(
            final_sol, human_seed_13[:7].to(final_sol.device),
            human_seed_13[7:].to(final_sol.device),
            args.lambda_rot, args.lambda_finger,
        )
        sorted_idx = torch.argsort(human_dist)[:top_k]
        print(f"[bodex_grasp] Ranked by human distance (best={float(human_dist[sorted_idx[0]]):.4f})")
    else:
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
    if args.rank_by_human and human_seed_13 is not None:
        output_data["human_distance"] = human_dist[sorted_idx].cpu()
        output_data["human_seed_13"] = human_seed_13.cpu()

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
