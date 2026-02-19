#!/usr/bin/env python3
"""Build replay actions for arm-follow-object debugging.

Supports:
1) Base-fixed arm-follow (original behavior)
2) Optional walk-to-grasp prefix phase (navigation velocity commands)
3) Optional CEDex grasp import to override right wrist pose in object frame
"""

from __future__ import annotations

import argparse
import json
import math
import os
from typing import Any

import h5py
import numpy as np


# 23D action layout
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


def _pose_matrix_from_pos_quat(pos: np.ndarray, quat_wxyz: np.ndarray) -> np.ndarray:
    t = np.eye(4, dtype=np.float64)
    t[:3, :3] = _quat_to_rotmat_wxyz(quat_wxyz)
    t[:3, 3] = np.asarray(pos, dtype=np.float64)
    return t


def _wrap_angle_rad(a: float) -> float:
    return float((a + math.pi) % (2.0 * math.pi) - math.pi)


def _normalize_rows(v: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return v / norms


def _rot6d_to_rotmat(rot6d: np.ndarray) -> np.ndarray:
    data = np.asarray(rot6d, dtype=np.float64)
    if data.ndim == 1:
        data = data[None, :]
    if data.ndim != 2 or data.shape[1] != 6:
        raise ValueError(f"rot6d should have shape [N,6], got {data.shape}")
    x = _normalize_rows(data[:, 0:3])
    y_raw = data[:, 3:6]
    z = _normalize_rows(np.cross(x, y_raw))
    y = np.cross(z, x)
    out = np.stack([x, y, z], axis=2)
    return out


def _load_object_traj(path: str) -> tuple[str, np.ndarray, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    if "object_pos_w" not in data or "object_quat_wxyz" not in data:
        raise KeyError(f"{path} must contain object_pos_w and object_quat_wxyz.")
    raw_name: Any = data.get("object_name", "unknown_object")
    if isinstance(raw_name, np.ndarray):
        if raw_name.shape == ():
            raw_name = raw_name.item()
        elif raw_name.size > 0:
            raw_name = raw_name.reshape(-1)[0]
    name = str(raw_name)
    pos = np.asarray(data["object_pos_w"], dtype=np.float64)
    quat = np.asarray(data["object_quat_wxyz"], dtype=np.float64)
    if pos.ndim != 2 or pos.shape[1] != 3:
        raise ValueError(f"object_pos_w should be [T,3], got {pos.shape}")
    if quat.ndim != 2 or quat.shape[1] != 4:
        raise ValueError(f"object_quat_wxyz should be [T,4], got {quat.shape}")
    if pos.shape[0] != quat.shape[0]:
        raise ValueError(f"object_pos_w length {pos.shape[0]} != object_quat_wxyz length {quat.shape[0]}")
    return name, pos, quat


def _load_cedex_wrist_pose_from_pt(
    grasp_pt_path: str,
    grasp_index: int,
    wrist_pos_offset: np.ndarray,
    wrist_quat_offset_wxyz: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    try:
        import torch
    except Exception as exc:
        raise RuntimeError("CEDex grasp import requires torch to be available.") from exc

    payload = torch.load(grasp_pt_path, map_location="cpu")
    if isinstance(payload, dict):
        grasp_entries = [payload]
    elif isinstance(payload, list):
        grasp_entries = payload
    else:
        raise TypeError(
            f"Unsupported CEDex grasp payload type: {type(payload)}. Expected dict or list of dicts."
        )

    if len(grasp_entries) == 0:
        raise ValueError(f"No grasp entries found in {grasp_pt_path}.")

    idx = int(grasp_index)
    if idx < 0:
        idx += len(grasp_entries)
    if idx < 0 or idx >= len(grasp_entries):
        raise IndexError(f"grasp_index={grasp_index} out of range for {len(grasp_entries)} entries.")

    entry = grasp_entries[idx]
    if isinstance(entry, dict) and "q" in entry:
        q_raw = entry["q"]
    else:
        q_raw = entry

    q = np.asarray(q_raw, dtype=np.float64).reshape(-1)
    if q.shape[0] < 9:
        raise ValueError(f"CEDex grasp vector must have >=9 values (xyz + rot6d), got {q.shape[0]}")

    hand_pos_obj = q[0:3]
    hand_rot_obj = _rot6d_to_rotmat(q[3:9])[0]
    hand_quat_obj = _rotmat_to_quat_wxyz(hand_rot_obj)

    t_hand_obj = _pose_matrix_from_pos_quat(hand_pos_obj, hand_quat_obj)
    t_wrist_hand = _pose_matrix_from_pos_quat(wrist_pos_offset, wrist_quat_offset_wxyz)
    t_wrist_obj = t_hand_obj @ t_wrist_hand

    wrist_pos_obj = t_wrist_obj[:3, 3]
    wrist_quat_obj = _rotmat_to_quat_wxyz(t_wrist_obj[:3, :3])

    meta: dict[str, Any] = {
        "grasp_pt_path": os.path.abspath(grasp_pt_path),
        "num_grasps": len(grasp_entries),
        "selected_index": int(idx),
        "q_dim": int(q.shape[0]),
    }
    if isinstance(entry, dict):
        meta["entry_keys"] = sorted(entry.keys())
        if "object_name" in entry:
            meta["object_name"] = str(entry["object_name"])
        if "robot_name" in entry:
            meta["robot_name"] = str(entry["robot_name"])
    return wrist_pos_obj, wrist_quat_obj, meta


def _build_walk_to_grasp_nav_cmds(
    start_base_pos_w: np.ndarray,
    start_base_yaw: float,
    target_base_pos_w: np.ndarray,
    target_base_yaw: float,
    max_lin_speed: float,
    max_ang_speed: float,
    dt: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    if max_lin_speed <= 0.0:
        raise ValueError("--walk-nav-max-lin-speed must be > 0.")
    if max_ang_speed <= 0.0:
        raise ValueError("--walk-nav-max-ang-speed must be > 0.")
    if dt <= 0.0:
        raise ValueError("--walk-nav-dt must be > 0.")

    start_xy = np.asarray(start_base_pos_w[:2], dtype=np.float64)
    target_xy = np.asarray(target_base_pos_w[:2], dtype=np.float64)
    delta_xy = target_xy - start_xy
    dist = float(np.linalg.norm(delta_xy))

    heading_to_target = float(math.atan2(delta_xy[1], delta_xy[0])) if dist > 1e-9 else float(start_base_yaw)
    turn1 = _wrap_angle_rad(heading_to_target - float(start_base_yaw))
    turn2 = _wrap_angle_rad(float(target_base_yaw) - heading_to_target)

    turn1_steps = int(math.ceil(abs(turn1) / (max_ang_speed * dt))) if abs(turn1) > 1e-6 else 0
    move_steps = int(math.ceil(dist / (max_lin_speed * dt))) if dist > 1e-9 else 0
    turn2_steps = int(math.ceil(abs(turn2) / (max_ang_speed * dt))) if abs(turn2) > 1e-6 else 0

    nav_cmds = np.zeros((turn1_steps + move_steps + turn2_steps, 3), dtype=np.float32)
    cursor = 0
    if turn1_steps > 0:
        nav_cmds[cursor : cursor + turn1_steps, 2] = np.float32(math.copysign(max_ang_speed, turn1))
        cursor += turn1_steps
    if move_steps > 0:
        nav_cmds[cursor : cursor + move_steps, 0] = np.float32(max_lin_speed)
        cursor += move_steps
    if turn2_steps > 0:
        nav_cmds[cursor : cursor + turn2_steps, 2] = np.float32(math.copysign(max_ang_speed, turn2))

    debug = {
        "start_xy": start_xy.tolist(),
        "target_xy": target_xy.tolist(),
        "dist_xy": float(dist),
        "heading_to_target_rad": float(heading_to_target),
        "target_yaw_rad": float(target_base_yaw),
        "turn1_rad": float(turn1),
        "turn2_rad": float(turn2),
        "turn1_steps": int(turn1_steps),
        "move_steps": int(move_steps),
        "turn2_steps": int(turn2_steps),
        "nav_steps": int(nav_cmds.shape[0]),
        "dt": float(dt),
        "max_lin_speed": float(max_lin_speed),
        "max_ang_speed": float(max_ang_speed),
    }
    return nav_cmds, debug


def _build_walk_nav_cmds_from_subgoals(
    start_base_pos_w: np.ndarray,
    start_base_yaw: float,
    subgoals: list[dict],
    max_lin_speed: float,
    max_ang_speed: float,
    dt: float,
) -> tuple[np.ndarray, float, dict[str, Any]]:
    """Generate nav_cmds from a list of subgoals [{"xy": [x,y], "yaw": rad}, ...]."""
    segments = []
    cur_pos = np.asarray(start_base_pos_w[:2], dtype=np.float64)
    cur_yaw = float(start_base_yaw)
    for sg in subgoals:
        tgt_xy = np.asarray(sg["xy"], dtype=np.float64)
        tgt_yaw = float(sg["yaw"])
        seg_cmds, _ = _build_walk_to_grasp_nav_cmds(
            start_base_pos_w=np.array([cur_pos[0], cur_pos[1], 0.0]),
            start_base_yaw=cur_yaw,
            target_base_pos_w=np.array([tgt_xy[0], tgt_xy[1], 0.0]),
            target_base_yaw=tgt_yaw,
            max_lin_speed=max_lin_speed,
            max_ang_speed=max_ang_speed,
            dt=dt,
        )
        segments.append(seg_cmds)
        cur_pos = tgt_xy
        cur_yaw = tgt_yaw
    nav_cmds = np.concatenate(segments, axis=0) if segments else np.zeros((0, 3), dtype=np.float32)
    debug = {"num_subgoals": len(subgoals), "total_nav_steps": int(nav_cmds.shape[0])}
    return nav_cmds, cur_yaw, debug


def _resolve_walk_target_base_pose(
    base_pos_w: np.ndarray,
    object_pos_w: np.ndarray,
    object_quat_wxyz: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, float, dict[str, Any]]:
    explicit_target = _parse_optional_csv_floats(args.walk_target_base_pos_w, 3, "walk_target_base_pos_w")
    target_pos_w = np.asarray(base_pos_w, dtype=np.float64).copy()

    if explicit_target is not None:
        target_pos_w[:] = explicit_target
    else:
        offset = _parse_csv_floats(args.walk_target_offset_obj_w, 3, "walk_target_offset_obj_w")
        if args.walk_target_offset_frame == "object":
            delta = _quat_to_rotmat_wxyz(object_quat_wxyz) @ offset
        else:
            delta = offset
        target_pos_w = np.asarray(object_pos_w, dtype=np.float64) + delta
    target_pos_w[2] = base_pos_w[2]

    if args.walk_target_yaw_mode == "face_object":
        yaw = math.atan2(object_pos_w[1] - target_pos_w[1], object_pos_w[0] - target_pos_w[0])
    elif args.walk_target_yaw_mode == "base_yaw":
        yaw = float(args.base_yaw)
    else:
        yaw = math.radians(float(args.walk_target_yaw_deg))

    debug = {
        "target_base_pos_w": target_pos_w.tolist(),
        "target_yaw_rad": float(yaw),
        "target_yaw_mode": args.walk_target_yaw_mode,
        "target_offset_frame": args.walk_target_offset_frame,
    }
    return target_pos_w, float(yaw), debug


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kin-traj-path", required=True, help="Input object kinematic trajectory npz.")
    parser.add_argument("--output-hdf5", required=True, help="Output replay actions hdf5.")
    parser.add_argument("--output-debug-json", default=None, help="Optional debug json path.")
    parser.add_argument("--episode-name", default="demo_0", help="HDF5 episode name.")

    parser.add_argument("--base-pos-w", default="0.0,0.0,0.0", help="Base world position xyz for arm-follow stage.")
    parser.add_argument("--base-yaw", type=float, default=0.0, help="Base yaw for arm-follow stage (rad).")
    parser.add_argument("--base-height", type=float, default=0.75, help="Base height command.")
    parser.add_argument("--torso-rpy", default="0.0,0.0,0.0", help="Torso orientation command r,p,y.")

    parser.add_argument("--left-hand-state", type=float, default=0.0, help="Left hand open/close state.")
    parser.add_argument("--right-hand-state", type=float, default=0.0, help="Right hand open/close state.")
    parser.add_argument("--left-wrist-pos", default="0.201,0.145,0.101", help="Left wrist fixed pelvis-frame xyz.")
    parser.add_argument("--left-wrist-quat-wxyz", default="1.0,0.01,-0.008,-0.011", help="Left wrist fixed pelvis-frame quat.")

    parser.add_argument(
        "--right-wrist-pos-obj",
        default="-0.28,-0.05,0.06",
        help="Right wrist target in object frame xyz.",
    )
    parser.add_argument(
        "--right-wrist-quat-obj-wxyz",
        default="0.70710678,0.0,-0.70710678,0.0",
        help="Right wrist target in object frame quat.",
    )
    parser.add_argument(
        "--right-wrist-quat-control",
        choices=["follow_object", "constant_pelvis"],
        default="follow_object",
        help="How to generate right wrist orientation command.",
    )
    parser.add_argument(
        "--right-wrist-quat-pelvis-wxyz",
        default="1.0,0.0,0.0,0.0",
        help="Constant right wrist quat command in pelvis frame (used when right-wrist-quat-control=constant_pelvis).",
    )

    parser.add_argument(
        "--walk-to-grasp",
        action="store_true",
        help="Enable a walk-to-grasp phase before arm-follow.",
    )
    parser.add_argument(
        "--walk-target-base-pos-w",
        default=None,
        help="Optional explicit walk target base xyz in world frame.",
    )
    parser.add_argument(
        "--walk-target-offset-obj-w",
        default="-0.35,0.0,0.0",
        help="If walk-target-base-pos-w is not provided, target base is object_pos + this offset.",
    )
    parser.add_argument(
        "--walk-target-offset-frame",
        choices=["object", "world"],
        default="object",
        help="Interpret walk-target-offset-obj-w in object frame or world frame.",
    )
    parser.add_argument(
        "--walk-target-yaw-mode",
        choices=["face_object", "fixed", "base_yaw"],
        default="face_object",
        help="How to set the base yaw at end of walk phase.",
    )
    parser.add_argument(
        "--walk-target-yaw-deg",
        type=float,
        default=0.0,
        help="Used when walk-target-yaw-mode=fixed.",
    )
    parser.add_argument("--walk-nav-max-lin-speed", type=float, default=0.22, help="Walk phase linear speed command.")
    parser.add_argument("--walk-nav-max-ang-speed", type=float, default=0.55, help="Walk phase angular speed command.")
    parser.add_argument("--walk-nav-dt", type=float, default=0.02, help="Action step size used for open-loop nav planning.")
    parser.add_argument(
        "--walk-pregrasp-hold-steps",
        type=int,
        default=25,
        help="Extra zero-velocity stabilization steps after navigation and before arm-follow.",
    )
    parser.add_argument(
        "--walk-nav-subgoals-json",
        default=None,
        help="JSON string of pre-computed nav subgoals from planner. Overrides internal straight-line nav.",
    )
    parser.add_argument(
        "--right-wrist-nav-pos",
        default="0.201,-0.145,0.101",
        help="Right wrist pelvis-frame xyz used during walk phase.",
    )
    parser.add_argument(
        "--right-wrist-nav-quat-wxyz",
        default="1.0,0.0,0.0,0.0",
        help="Right wrist pelvis-frame quat used during walk phase.",
    )

    parser.add_argument(
        "--cedex-grasp-pt",
        default=None,
        help="Optional CEDex generated_grasps .pt file. If set, override right wrist object-frame pose.",
    )
    parser.add_argument("--cedex-grasp-index", type=int, default=0, help="Grasp index in cedex grasp file.")
    parser.add_argument(
        "--cedex-wrist-pos-offset",
        default="0.0,0.0,0.0",
        help="Offset from CEDex hand frame to wrist frame, xyz in hand frame.",
    )
    parser.add_argument(
        "--cedex-wrist-quat-offset-wxyz",
        default="1.0,0.0,0.0,0.0",
        help="Offset rotation from CEDex hand frame to wrist frame (wxyz).",
    )
    return parser


def main() -> None:
    args = _make_parser().parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.output_hdf5)) or ".", exist_ok=True)
    if args.output_debug_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_debug_json)) or ".", exist_ok=True)

    object_name, obj_pos_w, obj_quat_wxyz = _load_object_traj(args.kin_traj_path)
    n_obj = obj_pos_w.shape[0]
    if n_obj < 2:
        raise ValueError("Trajectory should contain at least 2 frames.")

    left_wrist_pos = _parse_csv_floats(args.left_wrist_pos, 3, "left_wrist_pos")
    left_wrist_quat = _normalize_quat_wxyz(_parse_csv_floats(args.left_wrist_quat_wxyz, 4, "left_wrist_quat_wxyz"))
    right_wrist_pos_obj = _parse_csv_floats(args.right_wrist_pos_obj, 3, "right_wrist_pos_obj")
    right_wrist_quat_obj = _normalize_quat_wxyz(
        _parse_csv_floats(args.right_wrist_quat_obj_wxyz, 4, "right_wrist_quat_obj_wxyz")
    )
    right_wrist_quat_pelvis_const = _normalize_quat_wxyz(
        _parse_csv_floats(args.right_wrist_quat_pelvis_wxyz, 4, "right_wrist_quat_pelvis_wxyz")
    )
    right_wrist_nav_pos = _parse_csv_floats(args.right_wrist_nav_pos, 3, "right_wrist_nav_pos")
    right_wrist_nav_quat = _normalize_quat_wxyz(
        _parse_csv_floats(args.right_wrist_nav_quat_wxyz, 4, "right_wrist_nav_quat_wxyz")
    )
    base_pos_w = _parse_csv_floats(args.base_pos_w, 3, "base_pos_w")
    torso_rpy = _parse_csv_floats(args.torso_rpy, 3, "torso_rpy")

    cedex_debug: dict[str, Any] | None = None
    if args.cedex_grasp_pt:
        cedex_wrist_pos_offset = _parse_csv_floats(args.cedex_wrist_pos_offset, 3, "cedex_wrist_pos_offset")
        cedex_wrist_quat_offset = _normalize_quat_wxyz(
            _parse_csv_floats(args.cedex_wrist_quat_offset_wxyz, 4, "cedex_wrist_quat_offset_wxyz")
        )
        right_wrist_pos_obj, right_wrist_quat_obj, cedex_meta = _load_cedex_wrist_pose_from_pt(
            grasp_pt_path=args.cedex_grasp_pt,
            grasp_index=args.cedex_grasp_index,
            wrist_pos_offset=cedex_wrist_pos_offset,
            wrist_quat_offset_wxyz=cedex_wrist_quat_offset,
        )
        cedex_debug = {
            **cedex_meta,
            "right_wrist_pos_obj": right_wrist_pos_obj.tolist(),
            "right_wrist_quat_obj_wxyz": right_wrist_quat_obj.tolist(),
            "wrist_pos_offset": cedex_wrist_pos_offset.tolist(),
            "wrist_quat_offset_wxyz": cedex_wrist_quat_offset.tolist(),
        }

    nav_cmds = np.zeros((0, 3), dtype=np.float32)
    walk_debug: dict[str, Any] | None = None
    arm_base_pos_w = np.asarray(base_pos_w, dtype=np.float64)
    arm_base_yaw = float(args.base_yaw)
    pregrasp_hold_steps = 0

    if args.walk_to_grasp:
        subgoals_json = getattr(args, "walk_nav_subgoals_json", None)
        if subgoals_json:
            subgoals = json.loads(subgoals_json)
            # Use the last subgoal as the arm-follow base pose
            last_sg = subgoals[-1]
            arm_base_pos_w = np.array([last_sg["xy"][0], last_sg["xy"][1], base_pos_w[2]], dtype=np.float64)
            nav_cmds, arm_base_yaw, walk_nav_debug = _build_walk_nav_cmds_from_subgoals(
                start_base_pos_w=base_pos_w,
                start_base_yaw=float(args.base_yaw),
                subgoals=subgoals,
                max_lin_speed=float(args.walk_nav_max_lin_speed),
                max_ang_speed=float(args.walk_nav_max_ang_speed),
                dt=float(args.walk_nav_dt),
            )
            walk_target_debug = {
                "target_base_pos_w": arm_base_pos_w.tolist(),
                "target_yaw_rad": float(arm_base_yaw),
                "source": "external_subgoals",
            }
        else:
            arm_base_pos_w, arm_base_yaw, walk_target_debug = _resolve_walk_target_base_pose(
                base_pos_w=base_pos_w,
                object_pos_w=obj_pos_w[0],
                object_quat_wxyz=obj_quat_wxyz[0],
                args=args,
            )
            nav_cmds, walk_nav_debug = _build_walk_to_grasp_nav_cmds(
                start_base_pos_w=base_pos_w,
                start_base_yaw=float(args.base_yaw),
                target_base_pos_w=arm_base_pos_w,
                target_base_yaw=arm_base_yaw,
                max_lin_speed=float(args.walk_nav_max_lin_speed),
                max_ang_speed=float(args.walk_nav_max_ang_speed),
                dt=float(args.walk_nav_dt),
            )
        pregrasp_hold_steps = max(0, int(args.walk_pregrasp_hold_steps))
        walk_debug = {
            "enabled": True,
            **walk_target_debug,
            **walk_nav_debug,
            "pregrasp_hold_steps": int(pregrasp_hold_steps),
        }

    c = math.cos(arm_base_yaw)
    s = math.sin(arm_base_yaw)
    r_base_w = np.array(
        [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )

    t_hand_obj = _pose_matrix_from_pos_quat(right_wrist_pos_obj, right_wrist_quat_obj)
    right_wrist_pos_pelvis = np.zeros((n_obj, 3), dtype=np.float64)
    right_wrist_quat_pelvis = np.zeros((n_obj, 4), dtype=np.float64)
    for i in range(n_obj):
        t_obj_w = _pose_matrix_from_pos_quat(obj_pos_w[i], obj_quat_wxyz[i])
        t_hand_w = t_obj_w @ t_hand_obj
        p_hand_w = t_hand_w[:3, 3]
        r_hand_w = t_hand_w[:3, :3]
        p_hand_pelvis = r_base_w.T @ (p_hand_w - arm_base_pos_w)
        right_wrist_pos_pelvis[i] = p_hand_pelvis
        if args.right_wrist_quat_control == "constant_pelvis":
            right_wrist_quat_pelvis[i] = right_wrist_quat_pelvis_const
        else:
            r_hand_pelvis = r_base_w.T @ r_hand_w
            right_wrist_quat_pelvis[i] = _rotmat_to_quat_wxyz(r_hand_pelvis)

    nav_steps = int(nav_cmds.shape[0])
    prefix_steps = nav_steps + pregrasp_hold_steps
    total_steps = prefix_steps + n_obj

    actions = np.zeros((total_steps, ACTION_DIM), dtype=np.float32)
    actions[:, LEFT_HAND_STATE_IDX] = np.float32(args.left_hand_state)
    actions[:, RIGHT_HAND_STATE_IDX] = np.float32(args.right_hand_state)
    actions[:, LEFT_WRIST_POS_START_IDX:LEFT_WRIST_POS_END_IDX] = left_wrist_pos.astype(np.float32)
    actions[:, LEFT_WRIST_QUAT_START_IDX:LEFT_WRIST_QUAT_END_IDX] = left_wrist_quat.astype(np.float32)
    actions[:, BASE_HEIGHT_IDX] = np.float32(args.base_height)
    actions[:, TORSO_RPY_START_IDX:TORSO_RPY_END_IDX] = torso_rpy.astype(np.float32)

    if prefix_steps > 0:
        actions[:prefix_steps, RIGHT_WRIST_POS_START_IDX:RIGHT_WRIST_POS_END_IDX] = right_wrist_nav_pos.astype(np.float32)
        actions[
            :prefix_steps, RIGHT_WRIST_QUAT_START_IDX:RIGHT_WRIST_QUAT_END_IDX
        ] = right_wrist_nav_quat.astype(np.float32)
    if nav_steps > 0:
        actions[:nav_steps, NAV_CMD_START_IDX:NAV_CMD_END_IDX] = nav_cmds.astype(np.float32)

    arm_start = prefix_steps
    actions[arm_start:, RIGHT_WRIST_POS_START_IDX:RIGHT_WRIST_POS_END_IDX] = right_wrist_pos_pelvis.astype(np.float32)
    actions[arm_start:, RIGHT_WRIST_QUAT_START_IDX:RIGHT_WRIST_QUAT_END_IDX] = right_wrist_quat_pelvis.astype(np.float32)
    actions[arm_start:, NAV_CMD_START_IDX:NAV_CMD_END_IDX] = 0.0

    with h5py.File(args.output_hdf5, "w") as f:
        grp = f.create_group("data")
        grp.attrs["total"] = int(total_steps)
        grp.attrs["env_args"] = json.dumps({"env_name": "", "type": 2, "generator": "arm_follow_object_debug"})
        ep = grp.create_group(args.episode_name)
        ep.attrs["num_samples"] = int(total_steps)
        ep.create_dataset("actions", data=actions, compression="gzip")

    debug: dict[str, Any] = {
        "input": {
            "kin_traj_path": os.path.abspath(args.kin_traj_path),
            "object_name": object_name,
            "object_frames": int(n_obj),
        },
        "config": {
            "base_pos_w": base_pos_w.tolist(),
            "base_yaw": float(args.base_yaw),
            "arm_base_pos_w": arm_base_pos_w.tolist(),
            "arm_base_yaw": float(arm_base_yaw),
            "right_wrist_pos_obj": right_wrist_pos_obj.tolist(),
            "right_wrist_quat_obj_wxyz": right_wrist_quat_obj.tolist(),
            "right_wrist_quat_control": args.right_wrist_quat_control,
            "right_wrist_quat_pelvis_wxyz": right_wrist_quat_pelvis_const.tolist(),
        },
        "replay": {
            "total_steps": int(total_steps),
            "nav_steps": int(nav_steps),
            "pregrasp_hold_steps": int(pregrasp_hold_steps),
            "arm_follow_steps": int(n_obj),
            "recommended_kin_start_step": int(prefix_steps),
        },
        "sanity": {
            "right_wrist_pos_pelvis_min": right_wrist_pos_pelvis.min(axis=0).tolist(),
            "right_wrist_pos_pelvis_max": right_wrist_pos_pelvis.max(axis=0).tolist(),
            "nav_cmd_abs_max": np.max(np.abs(actions[:, NAV_CMD_START_IDX:NAV_CMD_END_IDX]), axis=0).tolist(),
        },
        "output_hdf5": os.path.abspath(args.output_hdf5),
        "episode_name": args.episode_name,
    }
    if walk_debug is not None:
        debug["walk_to_grasp"] = walk_debug
    if cedex_debug is not None:
        debug["cedex"] = cedex_debug

    if args.output_debug_json:
        with open(args.output_debug_json, "w", encoding="utf-8") as f:
            json.dump(debug, f, indent=2)

    print(f"[arm_follow] actions saved: {os.path.abspath(args.output_hdf5)}")
    print(
        "[arm_follow] "
        f"total_steps={total_steps}, nav_steps={nav_steps}, pregrasp_hold_steps={pregrasp_hold_steps}, "
        f"recommended_kin_start_step={prefix_steps}"
    )
    if args.output_debug_json:
        print(f"[arm_follow] debug saved: {os.path.abspath(args.output_debug_json)}")


if __name__ == "__main__":
    main()
