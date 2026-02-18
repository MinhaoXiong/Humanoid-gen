#!/usr/bin/env python3
"""Build G1 WBC+PINK replay actions from HOI object motion and optional BODex grasp."""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
from dataclasses import dataclass
from typing import Any

import h5py
import numpy as np
import torch


# 23D layout from isaaclab_arena_g1/.../action_constants.py
LEFT_HAND_STATE_IDX = 0
RIGHT_HAND_STATE_IDX = 1
LEFT_WRIST_POS_START_IDX = 2
LEFT_WRIST_POS_END_IDX = 5
LEFT_WRIST_QUAT_START_IDX = 5
LEFT_WRIST_QUAT_END_IDX = 9
RIGHT_WRIST_POS_START_IDX = 9
RIGHT_WRIST_POS_END_IDX = 12
RIGHT_WRIST_QUAT_START_IDX = 12
RIGHT_WRIST_QUAT_END_IDX = 16
NAV_CMD_START_IDX = 16
NAV_CMD_END_IDX = 19
BASE_HEIGHT_IDX = 19
TORSO_RPY_START_IDX = 20
TORSO_RPY_END_IDX = 23
ACTION_DIM = 23


@dataclass
class RelativeHandPose:
    pregrasp_pos_obj: np.ndarray
    pregrasp_quat_obj_wxyz: np.ndarray
    grasp_pos_obj: np.ndarray
    grasp_quat_obj_wxyz: np.ndarray
    source: str
    meta: dict[str, Any]


def _parse_csv_floats(text: str, expected_len: int, name: str) -> np.ndarray:
    values = [float(x.strip()) for x in text.split(",")]
    if len(values) != expected_len:
        raise ValueError(f"{name} expects {expected_len} values, got {len(values)}: {text}")
    return np.asarray(values, dtype=np.float64)


def _parse_optional_csv_floats(text: str | None, expected_len: int, name: str) -> np.ndarray | None:
    if text is None:
        return None
    trimmed = text.strip()
    if not trimmed or trimmed.lower() in {"none", "null"}:
        return None
    return _parse_csv_floats(trimmed, expected_len, name)


def _to_numpy(x: Any) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _normalize_quat_wxyz(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    norm = np.linalg.norm(q)
    if norm < 1e-12:
        raise ValueError("Quaternion norm is too small.")
    q = q / norm
    if q[0] < 0:
        q = -q
    return q


def _quat_to_rotmat_wxyz(q: np.ndarray) -> np.ndarray:
    q = _normalize_quat_wxyz(q)
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _rotmat_to_quat_wxyz(r: np.ndarray) -> np.ndarray:
    r = np.asarray(r, dtype=np.float64)
    tr = np.trace(r)
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2.0
        qw = 0.25 * s
        qx = (r[2, 1] - r[1, 2]) / s
        qy = (r[0, 2] - r[2, 0]) / s
        qz = (r[1, 0] - r[0, 1]) / s
    else:
        if r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
            s = math.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2.0
            qw = (r[2, 1] - r[1, 2]) / s
            qx = 0.25 * s
            qy = (r[0, 1] + r[1, 0]) / s
            qz = (r[0, 2] + r[2, 0]) / s
        elif r[1, 1] > r[2, 2]:
            s = math.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2.0
            qw = (r[0, 2] - r[2, 0]) / s
            qx = (r[0, 1] + r[1, 0]) / s
            qy = 0.25 * s
            qz = (r[1, 2] + r[2, 1]) / s
        else:
            s = math.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2.0
            qw = (r[1, 0] - r[0, 1]) / s
            qx = (r[0, 2] + r[2, 0]) / s
            qy = (r[1, 2] + r[2, 1]) / s
            qz = 0.25 * s
    return _normalize_quat_wxyz(np.array([qw, qx, qy, qz], dtype=np.float64))


def _slerp_wxyz(q0: np.ndarray, q1: np.ndarray, alpha: float) -> np.ndarray:
    q0 = _normalize_quat_wxyz(q0)
    q1 = _normalize_quat_wxyz(q1)
    dot = float(np.dot(q0, q1))
    if dot < 0:
        q1 = -q1
        dot = -dot
    dot = min(1.0, max(-1.0, dot))
    if dot > 0.9995:
        return _normalize_quat_wxyz((1.0 - alpha) * q0 + alpha * q1)
    theta_0 = math.acos(dot)
    theta = theta_0 * alpha
    sin_theta = math.sin(theta)
    sin_theta_0 = math.sin(theta_0)
    s0 = math.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0
    return _normalize_quat_wxyz(s0 * q0 + s1 * q1)


def _wrap_to_pi(x: float) -> float:
    return (x + math.pi) % (2.0 * math.pi) - math.pi


def _pose_matrix_from_pos_quat(pos: np.ndarray, quat_wxyz: np.ndarray) -> np.ndarray:
    t = np.eye(4, dtype=np.float64)
    t[:3, :3] = _quat_to_rotmat_wxyz(quat_wxyz)
    t[:3, 3] = np.asarray(pos, dtype=np.float64)
    return t


def _pos_quat_from_pose_matrix(t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.asarray(t[:3, 3], dtype=np.float64), _rotmat_to_quat_wxyz(t[:3, :3])


def _invert_pose_matrix(t: np.ndarray) -> np.ndarray:
    out = np.eye(4, dtype=np.float64)
    r = t[:3, :3]
    p = t[:3, 3]
    out[:3, :3] = r.T
    out[:3, 3] = -r.T @ p
    return out


def _resample_positions(pos: np.ndarray, src_fps: float, dst_fps: float) -> np.ndarray:
    if pos.shape[0] == 1 or abs(src_fps - dst_fps) < 1e-9:
        return pos.copy()
    src_len = pos.shape[0]
    duration = (src_len - 1) / src_fps
    dst_len = int(round(duration * dst_fps)) + 1
    t_src = np.arange(src_len, dtype=np.float64) / src_fps
    t_dst = np.arange(dst_len, dtype=np.float64) / dst_fps
    out = np.zeros((dst_len, pos.shape[1]), dtype=np.float64)
    for d in range(pos.shape[1]):
        out[:, d] = np.interp(t_dst, t_src, pos[:, d])
    return out


def _resample_rotmats(rotmats: np.ndarray, src_fps: float, dst_fps: float) -> np.ndarray:
    if rotmats.shape[0] == 1 or abs(src_fps - dst_fps) < 1e-9:
        return rotmats.copy()
    src_len = rotmats.shape[0]
    duration = (src_len - 1) / src_fps
    dst_len = int(round(duration * dst_fps)) + 1
    t_src = np.arange(src_len, dtype=np.float64) / src_fps
    t_dst = np.arange(dst_len, dtype=np.float64) / dst_fps
    q_src = np.stack([_rotmat_to_quat_wxyz(rotmats[i]) for i in range(src_len)], axis=0)
    out_quat = np.zeros((dst_len, 4), dtype=np.float64)
    for i, t in enumerate(t_dst):
        j = int(np.searchsorted(t_src, t, side="right") - 1)
        j = max(0, min(j, src_len - 2))
        t0 = t_src[j]
        t1 = t_src[j + 1]
        alpha = 0.0 if t1 <= t0 else (t - t0) / (t1 - t0)
        out_quat[i] = _slerp_wxyz(q_src[j], q_src[j + 1], float(alpha))
    return np.stack([_quat_to_rotmat_wxyz(q) for q in out_quat], axis=0)


def _load_hoi_human_object_results(path: str) -> tuple[np.ndarray, np.ndarray, str]:
    original_torch_load = torch.load

    def _cpu_torch_load(*args, **kwargs):
        kwargs.setdefault("map_location", torch.device("cpu"))
        return original_torch_load(*args, **kwargs)

    torch.load = _cpu_torch_load
    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
    finally:
        torch.load = original_torch_load

    if "obj_pos" not in data or "obj_rot_mat" not in data:
        raise KeyError(f"{path} does not contain required keys: obj_pos, obj_rot_mat")

    obj_pos = _to_numpy(data["obj_pos"]).astype(np.float64)
    obj_rot = _to_numpy(data["obj_rot_mat"]).astype(np.float64)
    object_name = str(data.get("object_name", "unknown_object"))
    if obj_pos.ndim != 2 or obj_pos.shape[1] != 3:
        raise ValueError(f"obj_pos should have shape [T,3], got {obj_pos.shape}")
    if obj_rot.ndim != 3 or obj_rot.shape[1:] != (3, 3):
        raise ValueError(f"obj_rot_mat should have shape [T,3,3], got {obj_rot.shape}")
    if obj_pos.shape[0] != obj_rot.shape[0]:
        raise ValueError(f"obj_pos length {obj_pos.shape[0]} != obj_rot_mat length {obj_rot.shape[0]}")
    return obj_pos, obj_rot, object_name


def _maybe_unbox_numpy_item(value: Any) -> Any:
    if isinstance(value, np.ndarray) and value.dtype == object and value.shape == ():
        return value.item()
    return value


def _select_bodex_object_pose(world_cfg: dict[str, Any], manip_name: str | None) -> tuple[str, np.ndarray]:
    if "mesh" not in world_cfg:
        raise KeyError("BODex world_cfg does not contain 'mesh' entries.")
    mesh_dict = world_cfg["mesh"]
    if not isinstance(mesh_dict, dict) or len(mesh_dict) == 0:
        raise ValueError("BODex world_cfg['mesh'] is empty.")
    if manip_name and manip_name in mesh_dict:
        key = manip_name
    else:
        key = sorted(mesh_dict.keys())[0]
    pose = np.asarray(mesh_dict[key]["pose"], dtype=np.float64)
    if pose.shape[0] != 7:
        raise ValueError(f"Object pose should have 7 values [x,y,z,qw,qx,qy,qz], got {pose.shape}")
    return key, pose


def _extract_relative_pose_from_bodex(
    bodex_grasp_npy: str,
    grasp_id: int,
    pregrasp_stage_idx: int,
    grasp_stage_idx: int,
) -> RelativeHandPose:
    raw = np.load(bodex_grasp_npy, allow_pickle=True)
    if isinstance(raw, np.ndarray) and raw.shape == ():
        data = raw.item()
    elif isinstance(raw, dict):
        data = raw
    else:
        raise ValueError(f"Unexpected BODex npy format: {type(raw)}")

    robot_pose = _maybe_unbox_numpy_item(data["robot_pose"])
    robot_pose = np.asarray(robot_pose, dtype=np.float64)
    while robot_pose.ndim > 3 and robot_pose.shape[0] == 1:
        robot_pose = robot_pose[0]
    if robot_pose.ndim == 2:
        robot_pose = robot_pose[None, :, :]
    if robot_pose.ndim != 3:
        raise ValueError(f"BODex robot_pose should be [N,Stage,Q], got {robot_pose.shape}")
    if not (0 <= grasp_id < robot_pose.shape[0]):
        raise IndexError(f"grasp_id {grasp_id} is out of range for robot_pose first dim {robot_pose.shape[0]}")

    stages = robot_pose[grasp_id]
    for idx, stage_name in [(pregrasp_stage_idx, "pregrasp_stage_idx"), (grasp_stage_idx, "grasp_stage_idx")]:
        if idx < 0 or idx >= stages.shape[0]:
            raise IndexError(f"{stage_name} {idx} out of range, stage count is {stages.shape[0]}")
    if stages.shape[1] < 7:
        raise ValueError(f"BODex robot_pose last dim should be >=7 (root pose + q), got {stages.shape[1]}")

    pregrasp_root = stages[pregrasp_stage_idx, :7]
    grasp_root = stages[grasp_stage_idx, :7]

    world_cfg = _maybe_unbox_numpy_item(data.get("world_cfg"))
    if not isinstance(world_cfg, dict):
        raise ValueError("BODex grasp npy does not contain a valid dict world_cfg.")
    manip_name = data.get("manip_name", None)
    if isinstance(manip_name, np.ndarray):
        manip_name = str(manip_name.tolist())
    elif manip_name is not None:
        manip_name = str(manip_name)
    mesh_key, obj_pose = _select_bodex_object_pose(world_cfg, manip_name)

    t_obj_w = _pose_matrix_from_pos_quat(obj_pose[:3], obj_pose[3:])
    t_w_obj = _invert_pose_matrix(t_obj_w)
    t_pre_w = _pose_matrix_from_pos_quat(pregrasp_root[:3], pregrasp_root[3:])
    t_grasp_w = _pose_matrix_from_pos_quat(grasp_root[:3], grasp_root[3:])

    t_pre_obj = t_w_obj @ t_pre_w
    t_grasp_obj = t_w_obj @ t_grasp_w
    pre_pos_obj, pre_quat_obj = _pos_quat_from_pose_matrix(t_pre_obj)
    grasp_pos_obj, grasp_quat_obj = _pos_quat_from_pose_matrix(t_grasp_obj)

    return RelativeHandPose(
        pregrasp_pos_obj=pre_pos_obj,
        pregrasp_quat_obj_wxyz=pre_quat_obj,
        grasp_pos_obj=grasp_pos_obj,
        grasp_quat_obj_wxyz=grasp_quat_obj,
        source="bodex_grasp_npy",
        meta={
            "grasp_npy": os.path.abspath(bodex_grasp_npy),
            "grasp_id": grasp_id,
            "pregrasp_stage_idx": pregrasp_stage_idx,
            "grasp_stage_idx": grasp_stage_idx,
            "selected_mesh_key": mesh_key,
        },
    )


def _build_relative_pose_from_manual_args(args: argparse.Namespace) -> RelativeHandPose:
    return RelativeHandPose(
        pregrasp_pos_obj=_parse_csv_floats(args.pregrasp_pos_obj, 3, "pregrasp_pos_obj"),
        pregrasp_quat_obj_wxyz=_normalize_quat_wxyz(
            _parse_csv_floats(args.pregrasp_quat_obj_wxyz, 4, "pregrasp_quat_obj_wxyz")
        ),
        grasp_pos_obj=_parse_csv_floats(args.grasp_pos_obj, 3, "grasp_pos_obj"),
        grasp_quat_obj_wxyz=_normalize_quat_wxyz(_parse_csv_floats(args.grasp_quat_obj_wxyz, 4, "grasp_quat_obj_wxyz")),
        source="manual",
        meta={},
    )


def _build_base_plan_from_object(
    obj_pos_w: np.ndarray,
    obj_rot_w: np.ndarray,
    target_fps: float,
    base_offset_obj_xy: np.ndarray,
    max_nav_speed: float,
    nav_end_idx: int,
    yaw_face_object: bool,
    initial_base_yaw: float,
) -> tuple[np.ndarray, np.ndarray]:
    n = obj_pos_w.shape[0]
    obj_yaw = np.arctan2(obj_rot_w[:, 1, 0], obj_rot_w[:, 0, 0])
    base_xy = np.zeros((n, 2), dtype=np.float64)
    base_yaw = np.zeros((n,), dtype=np.float64)

    for i in range(n):
        c = math.cos(float(obj_yaw[i]))
        s = math.sin(float(obj_yaw[i]))
        offset_w = np.array(
            [
                c * base_offset_obj_xy[0] - s * base_offset_obj_xy[1],
                s * base_offset_obj_xy[0] + c * base_offset_obj_xy[1],
            ],
            dtype=np.float64,
        )
        base_xy[i] = obj_pos_w[i, :2] + offset_w
        if yaw_face_object:
            direction = obj_pos_w[i, :2] - base_xy[i]
            if np.linalg.norm(direction) < 1e-9:
                base_yaw[i] = initial_base_yaw if i == 0 else base_yaw[i - 1]
            else:
                base_yaw[i] = math.atan2(float(direction[1]), float(direction[0]))
        else:
            base_yaw[i] = initial_base_yaw if i == 0 else base_yaw[i - 1]

    nav_cmd = np.zeros((n, 3), dtype=np.float64)
    for i in range(n - 1):
        if i > nav_end_idx:
            break
        dxy_w = (base_xy[i + 1] - base_xy[i]) * target_fps
        dyaw = _wrap_to_pi(float(base_yaw[i + 1] - base_yaw[i])) * target_fps
        yaw = float(base_yaw[i])
        vx = math.cos(yaw) * dxy_w[0] + math.sin(yaw) * dxy_w[1]
        vy = -math.sin(yaw) * dxy_w[0] + math.cos(yaw) * dxy_w[1]
        nav_cmd[i, 0] = float(np.clip(vx, -max_nav_speed, max_nav_speed))
        nav_cmd[i, 1] = float(np.clip(vy, -max_nav_speed, max_nav_speed))
        nav_cmd[i, 2] = float(np.clip(dyaw, -max_nav_speed, max_nav_speed))

    if n > 1:
        nav_cmd[-1] = nav_cmd[-2]

    return np.stack([base_xy[:, 0], base_xy[:, 1], np.zeros(n), base_yaw], axis=1), nav_cmd


def _apply_object_traj_constraints(
    obj_pos_w: np.ndarray,
    target_fps: float,
    scale_xyz: np.ndarray,
    offset_w: np.ndarray,
    align_first_pos_w: np.ndarray | None,
    align_last_pos_w: np.ndarray | None,
    align_last_ramp_sec: float,
    clip_z_min: float | None,
    clip_z_max: float | None,
    clip_xy_min: np.ndarray | None,
    clip_xy_max: np.ndarray | None,
) -> tuple[np.ndarray, dict[str, Any]]:
    out = obj_pos_w.copy()
    ops: list[dict[str, Any]] = []

    if clip_z_min is not None and clip_z_max is not None and clip_z_min > clip_z_max:
        raise ValueError(f"clip_z_min {clip_z_min} cannot be greater than clip_z_max {clip_z_max}")
    if clip_xy_min is not None and clip_xy_max is not None:
        if np.any(clip_xy_min > clip_xy_max):
            raise ValueError(f"clip_xy_min {clip_xy_min.tolist()} cannot be greater than clip_xy_max {clip_xy_max.tolist()}")

    if not np.allclose(scale_xyz, np.ones(3), atol=1e-12):
        anchor = out[0].copy()
        out = (out - anchor[None, :]) * scale_xyz[None, :] + anchor[None, :]
        ops.append({"type": "scale_xyz", "value": scale_xyz.tolist(), "anchor_w": anchor.tolist()})

    if not np.allclose(offset_w, np.zeros(3), atol=1e-12):
        out += offset_w[None, :]
        ops.append({"type": "offset_w", "value": offset_w.tolist()})

    if align_first_pos_w is not None:
        delta = align_first_pos_w - out[0]
        out += delta[None, :]
        ops.append({"type": "align_first_pos_w", "target_w": align_first_pos_w.tolist(), "delta_w": delta.tolist()})

    if align_last_pos_w is not None:
        delta = align_last_pos_w - out[-1]
        if align_last_ramp_sec > 0.0:
            ramp_steps = max(2, int(round(align_last_ramp_sec * target_fps)) + 1)
            start_idx = max(0, out.shape[0] - ramp_steps)
            denom = max(1, out.shape[0] - 1 - start_idx)
            for idx in range(start_idx, out.shape[0]):
                alpha = float(idx - start_idx) / float(denom)
                out[idx] += alpha * delta
            ops.append(
                {
                    "type": "align_last_pos_w_ramp",
                    "target_w": align_last_pos_w.tolist(),
                    "delta_w": delta.tolist(),
                    "ramp_sec": float(align_last_ramp_sec),
                    "ramp_steps": int(out.shape[0] - start_idx),
                }
            )
        else:
            out += delta[None, :]
            ops.append({"type": "align_last_pos_w_global", "target_w": align_last_pos_w.tolist(), "delta_w": delta.tolist()})

    if clip_z_min is not None:
        out[:, 2] = np.maximum(out[:, 2], clip_z_min)
        ops.append({"type": "clip_z_min", "value": float(clip_z_min)})
    if clip_z_max is not None:
        out[:, 2] = np.minimum(out[:, 2], clip_z_max)
        ops.append({"type": "clip_z_max", "value": float(clip_z_max)})
    if clip_xy_min is not None:
        out[:, 0] = np.maximum(out[:, 0], clip_xy_min[0])
        out[:, 1] = np.maximum(out[:, 1], clip_xy_min[1])
        ops.append({"type": "clip_xy_min", "value": clip_xy_min.tolist()})
    if clip_xy_max is not None:
        out[:, 0] = np.minimum(out[:, 0], clip_xy_max[0])
        out[:, 1] = np.minimum(out[:, 1], clip_xy_max[1])
        ops.append({"type": "clip_xy_max", "value": clip_xy_max.tolist()})

    summary = {
        "enabled": len(ops) > 0,
        "ops": ops,
        "source_first_w": obj_pos_w[0].tolist(),
        "source_last_w": obj_pos_w[-1].tolist(),
        "source_min_w": obj_pos_w.min(axis=0).tolist(),
        "source_max_w": obj_pos_w.max(axis=0).tolist(),
        "result_first_w": out[0].tolist(),
        "result_last_w": out[-1].tolist(),
        "result_min_w": out.min(axis=0).tolist(),
        "result_max_w": out.max(axis=0).tolist(),
    }
    return out, summary


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hoi-pickle", required=True, help="Path to hoifhli human_object_results.pkl.")
    parser.add_argument("--output-hdf5", required=True, help="Output replay file path (.hdf5).")
    parser.add_argument(
        "--output-object-traj",
        default="object_kinematic_traj.npz",
        help="Output object kinematic trajectory path (.npz).",
    )
    parser.add_argument(
        "--output-debug-json",
        default="bridge_debug.json",
        help="Output debug metadata (.json).",
    )
    parser.add_argument("--episode-name", default="demo_0", help="HDF5 episode name.")

    parser.add_argument("--hoi-fps", type=float, default=30.0, help="Input HOI trajectory FPS.")
    parser.add_argument("--target-fps", type=float, default=50.0, help="Output replay FPS for Isaac.")
    parser.add_argument("--traj-scale-xyz", default="1.0,1.0,1.0", help="Object trajectory scale in world XYZ.")
    parser.add_argument("--traj-offset-w", default="0.0,0.0,0.0", help="Object trajectory world translation offset XYZ.")
    parser.add_argument(
        "--align-first-pos-w",
        default=None,
        help="If set, shifts trajectory so first object position becomes this world XYZ.",
    )
    parser.add_argument(
        "--align-last-pos-w",
        default=None,
        help="If set, aligns trajectory end position to this world XYZ.",
    )
    parser.add_argument(
        "--align-last-ramp-sec",
        type=float,
        default=0.0,
        help="When aligning end position, blend only last N seconds instead of shifting whole sequence.",
    )
    parser.add_argument("--clip-z-min", type=float, default=None, help="Lower bound for object Z in world frame.")
    parser.add_argument("--clip-z-max", type=float, default=None, help="Upper bound for object Z in world frame.")
    parser.add_argument("--clip-xy-min", default=None, help="Optional world XY lower bounds (x_min,y_min).")
    parser.add_argument("--clip-xy-max", default=None, help="Optional world XY upper bounds (x_max,y_max).")

    parser.add_argument(
        "--bodex-grasp-npy",
        default=None,
        help="Optional BODex grasp result .npy. If absent, manual object-frame hand pose is used.",
    )
    parser.add_argument("--bodex-grasp-id", type=int, default=0, help="Selected grasp id in BODex robot_pose.")
    parser.add_argument("--bodex-pregrasp-stage-idx", type=int, default=0, help="BODex stage index for pregrasp.")
    parser.add_argument("--bodex-grasp-stage-idx", type=int, default=1, help="BODex stage index for grasp.")

    parser.add_argument(
        "--pregrasp-pos-obj",
        default="-0.35,-0.08,0.10",
        help="Fallback pregrasp pos in object frame.",
    )
    parser.add_argument(
        "--pregrasp-quat-obj-wxyz",
        default="1.0,0.0,0.0,0.0",
        help="Fallback pregrasp quat in object frame (wxyz).",
    )
    parser.add_argument("--grasp-pos-obj", default="-0.28,-0.05,0.06", help="Fallback grasp pos in object frame.")
    parser.add_argument(
        "--grasp-quat-obj-wxyz",
        default="1.0,0.0,0.0,0.0",
        help="Fallback grasp quat in object frame (wxyz).",
    )

    parser.add_argument(
        "--left-wrist-pos",
        default="0.201,0.145,0.101",
        help="Left wrist pose (pelvis frame) position xyz.",
    )
    parser.add_argument(
        "--left-wrist-quat-wxyz",
        default="1.0,0.01,-0.008,-0.011",
        help="Left wrist pose (pelvis frame) quaternion wxyz.",
    )
    parser.add_argument("--base-height", type=float, default=0.75, help="Base height command.")
    parser.add_argument("--torso-rpy", default="0.0,0.0,0.0", help="Torso orientation command roll,pitch,yaw.")

    parser.add_argument(
        "--base-offset-obj-xy",
        default="-0.55,0.00",
        help="Base XY offset in object local frame for auto base planning.",
    )
    parser.add_argument("--max-nav-speed", type=float, default=0.4, help="Clip magnitude for nav (vx,vy,wz).")
    parser.add_argument(
        "--yaw-face-object",
        action="store_true",
        help="If set, base yaw faces object during navigation planning.",
    )
    parser.add_argument("--initial-base-yaw", type=float, default=0.0, help="Initial yaw used when yaw-face-object is off.")

    parser.add_argument(
        "--grasp-frame-ratio",
        type=float,
        default=0.7,
        help="If --grasp-time-sec is unset, grasp frame = ratio * (N-1).",
    )
    parser.add_argument("--grasp-time-sec", type=float, default=None, help="Absolute grasp time (seconds).")
    parser.add_argument("--approach-duration-sec", type=float, default=0.8, help="Pregrasp->grasp interpolation duration.")
    parser.add_argument("--close-duration-sec", type=float, default=0.2, help="Hand close stage duration.")
    parser.add_argument("--grasp-hold-duration-sec", type=float, default=0.6, help="Post-close hold duration.")

    parser.add_argument("--left-hand-state", type=float, default=0.0, help="Left hand state (0=open, 1=close).")
    parser.add_argument("--right-hand-open-state", type=float, default=0.0, help="Right hand open state.")
    parser.add_argument("--right-hand-close-state", type=float, default=1.0, help="Right hand close state.")

    parser.add_argument("--object-name-override", default=None, help="Optional override for output object name.")
    return parser


def main() -> None:
    args = _make_parser().parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.output_hdf5)) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.output_object_traj)) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.output_debug_json)) or ".", exist_ok=True)

    obj_pos_src, obj_rot_src, object_name_src = _load_hoi_human_object_results(args.hoi_pickle)
    object_name = args.object_name_override if args.object_name_override else object_name_src

    obj_pos = _resample_positions(obj_pos_src, args.hoi_fps, args.target_fps)
    obj_rot = _resample_rotmats(obj_rot_src, args.hoi_fps, args.target_fps)
    traj_scale_xyz = _parse_csv_floats(args.traj_scale_xyz, 3, "traj_scale_xyz")
    traj_offset_w = _parse_csv_floats(args.traj_offset_w, 3, "traj_offset_w")
    align_first_pos_w = _parse_optional_csv_floats(args.align_first_pos_w, 3, "align_first_pos_w")
    align_last_pos_w = _parse_optional_csv_floats(args.align_last_pos_w, 3, "align_last_pos_w")
    clip_xy_min = _parse_optional_csv_floats(args.clip_xy_min, 2, "clip_xy_min")
    clip_xy_max = _parse_optional_csv_floats(args.clip_xy_max, 2, "clip_xy_max")
    obj_pos, traj_constraint_summary = _apply_object_traj_constraints(
        obj_pos_w=obj_pos,
        target_fps=args.target_fps,
        scale_xyz=traj_scale_xyz,
        offset_w=traj_offset_w,
        align_first_pos_w=align_first_pos_w,
        align_last_pos_w=align_last_pos_w,
        align_last_ramp_sec=args.align_last_ramp_sec,
        clip_z_min=args.clip_z_min,
        clip_z_max=args.clip_z_max,
        clip_xy_min=clip_xy_min,
        clip_xy_max=clip_xy_max,
    )
    n = obj_pos.shape[0]
    if n < 2:
        raise ValueError("Trajectory needs at least 2 frames after resampling.")

    if args.bodex_grasp_npy:
        rel_pose = _extract_relative_pose_from_bodex(
            args.bodex_grasp_npy,
            args.bodex_grasp_id,
            args.bodex_pregrasp_stage_idx,
            args.bodex_grasp_stage_idx,
        )
    else:
        rel_pose = _build_relative_pose_from_manual_args(args)

    if args.grasp_time_sec is not None:
        grasp_idx = int(round(args.grasp_time_sec * args.target_fps))
    else:
        grasp_idx = int(round((n - 1) * args.grasp_frame_ratio))
    grasp_idx = int(np.clip(grasp_idx, 0, n - 1))
    approach_steps = max(1, int(round(args.approach_duration_sec * args.target_fps)))
    close_steps = max(1, int(round(args.close_duration_sec * args.target_fps)))
    hold_steps = max(1, int(round(args.grasp_hold_duration_sec * args.target_fps)))

    approach_start = max(0, grasp_idx - approach_steps)
    close_start = grasp_idx
    close_end = min(n - 1, close_start + close_steps - 1)
    hold_end = min(n - 1, close_end + hold_steps)
    nav_end = max(0, approach_start - 1)

    base_offset_obj_xy = _parse_csv_floats(args.base_offset_obj_xy, 2, "base_offset_obj_xy")
    base_pose_xyzyaw, nav_cmd = _build_base_plan_from_object(
        obj_pos_w=obj_pos,
        obj_rot_w=obj_rot,
        target_fps=args.target_fps,
        base_offset_obj_xy=base_offset_obj_xy,
        max_nav_speed=args.max_nav_speed,
        nav_end_idx=nav_end,
        yaw_face_object=args.yaw_face_object,
        initial_base_yaw=args.initial_base_yaw,
    )

    right_wrist_pos_pelvis = np.zeros((n, 3), dtype=np.float64)
    right_wrist_quat_pelvis = np.zeros((n, 4), dtype=np.float64)
    for i in range(n):
        if i <= approach_start:
            alpha = 0.0
        elif i >= grasp_idx:
            alpha = 1.0
        else:
            alpha = float(i - approach_start) / float(max(1, grasp_idx - approach_start))
        rel_pos = (1.0 - alpha) * rel_pose.pregrasp_pos_obj + alpha * rel_pose.grasp_pos_obj
        rel_quat = _slerp_wxyz(rel_pose.pregrasp_quat_obj_wxyz, rel_pose.grasp_quat_obj_wxyz, alpha)

        t_obj_w = np.eye(4, dtype=np.float64)
        t_obj_w[:3, :3] = obj_rot[i]
        t_obj_w[:3, 3] = obj_pos[i]
        t_hand_obj = _pose_matrix_from_pos_quat(rel_pos, rel_quat)
        t_hand_w = t_obj_w @ t_hand_obj

        base_yaw = base_pose_xyzyaw[i, 3]
        r_base_w = np.array(
            [
                [math.cos(base_yaw), -math.sin(base_yaw), 0.0],
                [math.sin(base_yaw), math.cos(base_yaw), 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        p_base_w = np.array([base_pose_xyzyaw[i, 0], base_pose_xyzyaw[i, 1], base_pose_xyzyaw[i, 2]], dtype=np.float64)

        p_hand_w = t_hand_w[:3, 3]
        r_hand_w = t_hand_w[:3, :3]
        p_hand_pelvis = r_base_w.T @ (p_hand_w - p_base_w)
        r_hand_pelvis = r_base_w.T @ r_hand_w

        right_wrist_pos_pelvis[i] = p_hand_pelvis
        right_wrist_quat_pelvis[i] = _rotmat_to_quat_wxyz(r_hand_pelvis)

    left_wrist_pos = _parse_csv_floats(args.left_wrist_pos, 3, "left_wrist_pos")
    left_wrist_quat = _normalize_quat_wxyz(_parse_csv_floats(args.left_wrist_quat_wxyz, 4, "left_wrist_quat_wxyz"))
    torso_rpy = _parse_csv_floats(args.torso_rpy, 3, "torso_rpy")

    actions = np.zeros((n, ACTION_DIM), dtype=np.float32)
    actions[:, LEFT_HAND_STATE_IDX] = np.float32(args.left_hand_state)
    actions[:, RIGHT_HAND_STATE_IDX] = np.float32(args.right_hand_open_state)
    actions[close_start : hold_end + 1, RIGHT_HAND_STATE_IDX] = np.float32(args.right_hand_close_state)

    actions[:, LEFT_WRIST_POS_START_IDX:LEFT_WRIST_POS_END_IDX] = left_wrist_pos.astype(np.float32)
    actions[:, LEFT_WRIST_QUAT_START_IDX:LEFT_WRIST_QUAT_END_IDX] = left_wrist_quat.astype(np.float32)
    actions[:, RIGHT_WRIST_POS_START_IDX:RIGHT_WRIST_POS_END_IDX] = right_wrist_pos_pelvis.astype(np.float32)
    actions[:, RIGHT_WRIST_QUAT_START_IDX:RIGHT_WRIST_QUAT_END_IDX] = right_wrist_quat_pelvis.astype(np.float32)
    actions[:, NAV_CMD_START_IDX:NAV_CMD_END_IDX] = nav_cmd.astype(np.float32)
    actions[:, BASE_HEIGHT_IDX] = np.float32(args.base_height)
    actions[:, TORSO_RPY_START_IDX:TORSO_RPY_END_IDX] = torso_rpy.astype(np.float32)

    with h5py.File(args.output_hdf5, "w") as f:
        grp = f.create_group("data")
        grp.attrs["total"] = int(n)
        grp.attrs["env_args"] = json.dumps({"env_name": "", "type": 2, "generator": "hoi_bodex_g1_bridge"})
        ep = grp.create_group(args.episode_name)
        ep.attrs["num_samples"] = int(n)
        ep.create_dataset("actions", data=actions, compression="gzip")

    obj_quat = np.stack([_rotmat_to_quat_wxyz(r) for r in obj_rot], axis=0).astype(np.float32)
    np.savez_compressed(
        args.output_object_traj,
        object_name=np.array(object_name),
        fps=np.array([args.target_fps], dtype=np.float32),
        object_pos_w=obj_pos.astype(np.float32),
        object_quat_wxyz=obj_quat,
        object_rot_mat_w=obj_rot.astype(np.float32),
    )

    debug = {
        "inputs": {
            "hoi_pickle": os.path.abspath(args.hoi_pickle),
            "bodex_grasp_npy": os.path.abspath(args.bodex_grasp_npy) if args.bodex_grasp_npy else None,
            "hoi_fps": args.hoi_fps,
            "target_fps": args.target_fps,
        },
        "outputs": {
            "hdf5": os.path.abspath(args.output_hdf5),
            "object_traj": os.path.abspath(args.output_object_traj),
            "episode_name": args.episode_name,
        },
        "object_name": object_name,
        "traj_constraints": traj_constraint_summary,
        "length": {
            "src_frames": int(obj_pos_src.shape[0]),
            "dst_frames": int(n),
            "duration_sec": float((n - 1) / args.target_fps),
        },
        "stages": {
            "navigation": {"start": 0, "end": int(nav_end)},
            "pregrasp": {"start": int(nav_end + 1), "end": int(approach_start)},
            "approach": {"start": int(approach_start), "end": int(grasp_idx)},
            "grasp_close": {"start": int(close_start), "end": int(close_end)},
            "grasp_hold": {"start": int(close_end + 1), "end": int(hold_end)},
        },
        "relative_hand_pose": {
            "source": rel_pose.source,
            "pregrasp_pos_obj": rel_pose.pregrasp_pos_obj.tolist(),
            "pregrasp_quat_obj_wxyz": rel_pose.pregrasp_quat_obj_wxyz.tolist(),
            "grasp_pos_obj": rel_pose.grasp_pos_obj.tolist(),
            "grasp_quat_obj_wxyz": rel_pose.grasp_quat_obj_wxyz.tolist(),
            "meta": rel_pose.meta,
        },
        "sanity": {
            "nav_cmd_abs_max": np.max(np.abs(actions[:, NAV_CMD_START_IDX:NAV_CMD_END_IDX]), axis=0).tolist(),
            "right_wrist_pos_range": {
                "min": right_wrist_pos_pelvis.min(axis=0).tolist(),
                "max": right_wrist_pos_pelvis.max(axis=0).tolist(),
            },
            "right_wrist_quat_first": right_wrist_quat_pelvis[0].tolist(),
        },
    }
    with open(args.output_debug_json, "w", encoding="utf-8") as f:
        json.dump(debug, f, indent=2)

    print(f"[bridge] HOI frames: {obj_pos_src.shape[0]} @ {args.hoi_fps:.3f}Hz -> {n} @ {args.target_fps:.3f}Hz")
    if traj_constraint_summary["enabled"]:
        print(f"[bridge] Applied trajectory constraints: {len(traj_constraint_summary['ops'])} ops")
    print(f"[bridge] Action replay saved: {os.path.abspath(args.output_hdf5)}")
    print(f"[bridge] Object kinematic traj saved: {os.path.abspath(args.output_object_traj)}")
    print(f"[bridge] Debug metadata saved: {os.path.abspath(args.output_debug_json)}")


if __name__ == "__main__":
    main()
