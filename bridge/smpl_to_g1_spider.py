#!/usr/bin/env python3
"""SMPL-to-G1 bridge via Spider: convert Spider retargeting output
(trajectory_mjwp.npz) into Humanoid-gen-pack's object_kinematic_traj.npz format.

Spider outputs G1 full-body joint trajectories + object poses in MuJoCo format.
This script extracts the object trajectory and converts it to the format expected
by run_walk_to_grasp_todo.py (Step 1 replacement).

Usage:
    # Convert Spider output to Humanoid-gen-pack format
    python bridge/smpl_to_g1_spider.py \
        --spider-npz /path/to/trajectory_mjwp.npz \
        --spider-xml /path/to/scene.xml \
        --output-npz artifacts/object_kinematic_traj.npz \
        --output-debug-json artifacts/debug_traj.json \
        --scene lwbench_kitchen_0_0 \
        --object-name cracker_box

    # With explicit object body name in MuJoCo XML
    python bridge/smpl_to_g1_spider.py \
        --spider-npz /path/to/trajectory_mjwp.npz \
        --spider-xml /path/to/scene.xml \
        --object-body-name object_0 \
        --output-npz artifacts/object_kinematic_traj.npz

    # From Spider kinematic output (no physics optimization)
    python bridge/smpl_to_g1_spider.py \
        --spider-npz /path/to/trajectory_kinematic.npz \
        --spider-xml /path/to/scene.xml \
        --output-npz artifacts/object_kinematic_traj.npz
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Any

import numpy as np

_SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
_PACK_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
if _PACK_ROOT not in sys.path:
    sys.path.insert(0, _PACK_ROOT)

# Spider path
_SPIDER_ROOT = os.path.abspath(os.path.join(_PACK_ROOT, "..", "spider"))

TARGET_FPS = 50.0


# ---------------------------------------------------------------------------
# Quaternion helpers (MuJoCo uses wxyz, same as Humanoid-gen-pack)
# ---------------------------------------------------------------------------
def _quat_normalize(q: np.ndarray) -> np.ndarray:
    """Normalize quaternion(s) in wxyz format."""
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    norm = np.where(norm < 1e-12, 1.0, norm)
    return q / norm


def _quat_from_rotmat(R: np.ndarray) -> np.ndarray:
    """Convert rotation matrix to quaternion wxyz. Supports batched [N,3,3]."""
    try:
        from scipy.spatial.transform import Rotation
        if R.ndim == 2:
            q_xyzw = Rotation.from_matrix(R).as_quat()
            return np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])
        q_xyzw = Rotation.from_matrix(R).as_quat()  # [N, 4] xyzw
        return np.column_stack([q_xyzw[:, 3], q_xyzw[:, 0], q_xyzw[:, 1], q_xyzw[:, 2]])
    except ImportError:
        raise ImportError("scipy is required for rotation conversion")


# ---------------------------------------------------------------------------
# MuJoCo model parsing
# ---------------------------------------------------------------------------
def _load_mujoco_model(xml_path: str):
    """Load MuJoCo model and return (model, data)."""
    import mujoco
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    return model, data


def _find_object_body_id(model, object_body_name: str | None = None) -> tuple[int, str]:
    """Find the object body in the MuJoCo model.

    If object_body_name is given, use it directly.
    Otherwise, heuristically find the first free-joint body that isn't the robot.
    """
    import mujoco

    if object_body_name:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, object_body_name)
        if body_id < 0:
            raise ValueError(f"Body '{object_body_name}' not found in MuJoCo model")
        return body_id, object_body_name

    # Heuristic: find bodies with free joints that aren't part of the robot
    robot_keywords = ["pelvis", "torso", "leg", "arm", "hand", "finger", "foot", "knee", "hip", "shoulder", "elbow", "wrist"]
    for i in range(model.nbody):
        name = model.body(i).name
        lower_name = name.lower()
        if any(kw in lower_name for kw in robot_keywords):
            continue
        # Check if this body has a free joint
        jnt_start = model.body_jntadr[i]
        if jnt_start >= 0 and model.jnt_type[jnt_start] == mujoco.mjtJoint.mjJNT_FREE:
            return i, name

    raise ValueError("Could not find object body in MuJoCo model. Use --object-body-name to specify.")


def _extract_object_trajectory_from_qpos(
    model,
    data,
    qpos_traj: np.ndarray,
    object_body_id: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract object position and quaternion from qpos trajectory.

    MuJoCo free joint qpos: [x, y, z, qw, qx, qy, qz]

    Returns:
        pos: [T, 3] object positions in world frame
        quat_wxyz: [T, 4] object quaternions in wxyz format
    """
    import mujoco

    T = qpos_traj.shape[0]
    pos = np.zeros((T, 3), dtype=np.float64)
    quat_wxyz = np.zeros((T, 4), dtype=np.float64)

    for t in range(T):
        data.qpos[:] = qpos_traj[t]
        mujoco.mj_forward(model, data)
        # Body xpos and xquat are in world frame after mj_forward
        pos[t] = data.xpos[object_body_id].copy()
        quat_wxyz[t] = data.xquat[object_body_id].copy()  # MuJoCo uses wxyz

    return pos, quat_wxyz


def _extract_object_from_free_joint(
    model,
    qpos_traj: np.ndarray,
    object_body_id: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Fast extraction: directly read free joint qpos without forward kinematics.

    This works when the object has a free joint directly attached to world.
    """
    jnt_start = model.body_jntadr[object_body_id]
    qpos_start = model.jnt_qposadr[jnt_start]

    # Free joint: 7 qpos values [x, y, z, qw, qx, qy, qz]
    pos = qpos_traj[:, qpos_start:qpos_start + 3].copy()
    quat_wxyz = qpos_traj[:, qpos_start + 3:qpos_start + 7].copy()

    return pos, _quat_normalize(quat_wxyz)


# ---------------------------------------------------------------------------
# Trajectory alignment to scene
# ---------------------------------------------------------------------------
def _align_trajectory_to_scene(
    pos: np.ndarray,
    quat_wxyz: np.ndarray,
    scene: str,
    target_fps: float = TARGET_FPS,
    source_fps: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Align object trajectory to scene table surface.

    Applies scene-specific offsets from scene_config.py.
    """
    try:
        from bridge.scene_config import get_scene
        sc = get_scene(scene)
        table_z = sc.table_z
        align_pos = np.array(sc.object_align_pos, dtype=np.float64)
    except (ImportError, KeyError):
        table_z = 0.1
        align_pos = np.array([0.4, 0.0, 0.1], dtype=np.float64)

    # Translate so first frame object center is at align_pos
    offset = align_pos - pos[0]
    pos_aligned = pos + offset

    # Resample to target FPS if needed
    if source_fps is not None and abs(source_fps - target_fps) > 0.5:
        T_orig = pos_aligned.shape[0]
        duration = T_orig / source_fps
        T_new = max(2, int(round(duration * target_fps)))
        t_orig = np.linspace(0, 1, T_orig)
        t_new = np.linspace(0, 1, T_new)
        pos_resampled = np.zeros((T_new, 3), dtype=np.float64)
        for dim in range(3):
            pos_resampled[:, dim] = np.interp(t_new, t_orig, pos_aligned[:, dim])

        # Slerp quaternions
        try:
            from scipy.spatial.transform import Rotation, Slerp
            # Convert wxyz to xyzw for scipy
            q_xyzw = np.column_stack([quat_wxyz[:, 1], quat_wxyz[:, 2], quat_wxyz[:, 3], quat_wxyz[:, 0]])
            rots = Rotation.from_quat(q_xyzw)
            slerp = Slerp(t_orig, rots)
            rots_new = slerp(t_new)
            q_new_xyzw = rots_new.as_quat()
            quat_resampled = np.column_stack([q_new_xyzw[:, 3], q_new_xyzw[:, 0], q_new_xyzw[:, 1], q_new_xyzw[:, 2]])
        except ImportError:
            # Fallback: nearest neighbor
            indices = np.round(t_new * (T_orig - 1)).astype(int)
            quat_resampled = quat_wxyz[indices]

        return pos_resampled, quat_resampled

    return pos_aligned, quat_wxyz


# ---------------------------------------------------------------------------
# Main conversion
# ---------------------------------------------------------------------------
def convert_spider_to_kinematic_traj(
    spider_npz_path: str,
    spider_xml_path: str | None = None,
    object_body_name: str | None = None,
    scene: str = "kitchen_pick_and_place",
    object_name: str = "cracker_box",
    source_fps: float | None = None,
) -> dict[str, Any]:
    """Convert Spider trajectory NPZ to Humanoid-gen-pack object_kinematic_traj format.

    Returns dict with keys: object_pos_w, object_quat_wxyz, object_name, metadata.
    """
    data = np.load(spider_npz_path, allow_pickle=True)
    qpos_traj = data["qpos"]  # [T, nq]
    T = qpos_traj.shape[0]

    # Determine source FPS from Spider config
    if source_fps is None:
        # Spider default: ref_dt=0.0333333 → ~30 FPS
        source_fps = 30.0

    if spider_xml_path and os.path.isfile(spider_xml_path):
        import mujoco
        model, mj_data = _load_mujoco_model(spider_xml_path)
        body_id, body_name = _find_object_body_id(model, object_body_name)

        # Try fast extraction first
        jnt_start = model.body_jntadr[body_id]
        if jnt_start >= 0 and model.jnt_type[jnt_start] == mujoco.mjtJoint.mjJNT_FREE:
            pos, quat_wxyz = _extract_object_from_free_joint(model, qpos_traj, body_id)
        else:
            pos, quat_wxyz = _extract_object_trajectory_from_qpos(model, mj_data, qpos_traj, body_id)
    else:
        # Without XML: assume object free joint is at the end of qpos
        # Spider humanoid config: robot qpos first, then object qpos
        # Object free joint: last 7 values [x, y, z, qw, qx, qy, qz]
        nq = qpos_traj.shape[1]
        if nq >= 7:
            pos = qpos_traj[:, -7:-4].copy()
            quat_wxyz = qpos_traj[:, -4:].copy()
            quat_wxyz = _quat_normalize(quat_wxyz)
            body_name = object_body_name or "object_inferred"
        else:
            raise ValueError(f"qpos has only {nq} dimensions, cannot extract object pose")

    # Align to scene
    pos_aligned, quat_aligned = _align_trajectory_to_scene(
        pos, quat_wxyz, scene,
        target_fps=TARGET_FPS,
        source_fps=source_fps,
    )

    return {
        "object_pos_w": pos_aligned,
        "object_quat_wxyz": quat_aligned,
        "object_name": object_name,
        "metadata": {
            "source": "spider",
            "spider_npz": spider_npz_path,
            "spider_xml": spider_xml_path,
            "object_body": body_name if spider_xml_path else "inferred",
            "original_frames": T,
            "output_frames": pos_aligned.shape[0],
            "source_fps": source_fps,
            "target_fps": TARGET_FPS,
            "scene": scene,
        },
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--spider-npz", required=True, help="Path to Spider trajectory NPZ (trajectory_mjwp.npz)")
    parser.add_argument("--spider-xml", default=None, help="Path to Spider scene XML (for object body lookup)")
    parser.add_argument("--object-body-name", default=None, help="MuJoCo body name of the object")
    parser.add_argument("--output-npz", required=True, help="Output object_kinematic_traj.npz path")
    parser.add_argument("--output-debug-json", default=None, help="Optional debug JSON output")
    parser.add_argument("--scene", default="kitchen_pick_and_place", help="Scene name for alignment")
    parser.add_argument("--object-name", default="cracker_box", help="Object name for IsaacLab-Arena")
    parser.add_argument("--source-fps", type=float, default=None, help="Source trajectory FPS (default: auto-detect)")
    return parser


def main() -> None:
    args = _make_parser().parse_args()

    result = convert_spider_to_kinematic_traj(
        spider_npz_path=args.spider_npz,
        spider_xml_path=args.spider_xml,
        object_body_name=args.object_body_name,
        scene=args.scene,
        object_name=args.object_name,
        source_fps=args.source_fps,
    )

    # Save NPZ
    os.makedirs(os.path.dirname(os.path.abspath(args.output_npz)), exist_ok=True)
    np.savez(
        args.output_npz,
        object_pos_w=result["object_pos_w"],
        object_quat_wxyz=result["object_quat_wxyz"],
        object_name=result["object_name"],
    )

    print(f"[smpl_to_g1] Source: {args.spider_npz}")
    print(f"[smpl_to_g1] Frames: {result['metadata']['original_frames']} → {result['metadata']['output_frames']}")
    print(f"[smpl_to_g1] FPS: {result['metadata']['source_fps']} → {result['metadata']['target_fps']}")
    print(f"[smpl_to_g1] Scene: {args.scene}")
    print(f"[smpl_to_g1] Object: {args.object_name}")
    print(f"[smpl_to_g1] Output: {args.output_npz}")

    # Save debug JSON
    if args.output_debug_json:
        debug = {
            "metadata": result["metadata"],
            "object_pos_w_first": result["object_pos_w"][0].tolist(),
            "object_pos_w_last": result["object_pos_w"][-1].tolist(),
            "object_quat_wxyz_first": result["object_quat_wxyz"][0].tolist(),
        }
        with open(args.output_debug_json, "w", encoding="utf-8") as f:
            json.dump(debug, f, indent=2)
        print(f"[smpl_to_g1] Debug: {args.output_debug_json}")


if __name__ == "__main__":
    main()
