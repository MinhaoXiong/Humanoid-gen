#!/usr/bin/env python3
"""Build replay actions for base-fixed arm-follow-object debugging."""

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


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kin-traj-path", required=True, help="Input object kinematic trajectory npz.")
    parser.add_argument("--output-hdf5", required=True, help="Output replay actions hdf5.")
    parser.add_argument("--output-debug-json", default=None, help="Optional debug json path.")
    parser.add_argument("--episode-name", default="demo_0", help="HDF5 episode name.")

    parser.add_argument("--base-pos-w", default="0.0,0.0,0.0", help="Fixed base world position xyz.")
    parser.add_argument("--base-yaw", type=float, default=0.0, help="Fixed base yaw (rad).")
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
    return parser


def main() -> None:
    args = _make_parser().parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.output_hdf5)) or ".", exist_ok=True)
    if args.output_debug_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_debug_json)) or ".", exist_ok=True)

    object_name, obj_pos_w, obj_quat_wxyz = _load_object_traj(args.kin_traj_path)
    n = obj_pos_w.shape[0]
    if n < 2:
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
    base_pos_w = _parse_csv_floats(args.base_pos_w, 3, "base_pos_w")
    torso_rpy = _parse_csv_floats(args.torso_rpy, 3, "torso_rpy")

    c = math.cos(args.base_yaw)
    s = math.sin(args.base_yaw)
    r_base_w = np.array(
        [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )

    t_hand_obj = _pose_matrix_from_pos_quat(right_wrist_pos_obj, right_wrist_quat_obj)
    right_wrist_pos_pelvis = np.zeros((n, 3), dtype=np.float64)
    right_wrist_quat_pelvis = np.zeros((n, 4), dtype=np.float64)

    for i in range(n):
        t_obj_w = _pose_matrix_from_pos_quat(obj_pos_w[i], obj_quat_wxyz[i])
        t_hand_w = t_obj_w @ t_hand_obj
        p_hand_w = t_hand_w[:3, 3]
        r_hand_w = t_hand_w[:3, :3]
        p_hand_pelvis = r_base_w.T @ (p_hand_w - base_pos_w)
        right_wrist_pos_pelvis[i] = p_hand_pelvis
        if args.right_wrist_quat_control == "constant_pelvis":
            right_wrist_quat_pelvis[i] = right_wrist_quat_pelvis_const
        else:
            r_hand_pelvis = r_base_w.T @ r_hand_w
            right_wrist_quat_pelvis[i] = _rotmat_to_quat_wxyz(r_hand_pelvis)

    actions = np.zeros((n, ACTION_DIM), dtype=np.float32)
    actions[:, LEFT_HAND_STATE_IDX] = np.float32(args.left_hand_state)
    actions[:, RIGHT_HAND_STATE_IDX] = np.float32(args.right_hand_state)
    actions[:, LEFT_WRIST_POS_START_IDX:LEFT_WRIST_POS_END_IDX] = left_wrist_pos.astype(np.float32)
    actions[:, LEFT_WRIST_QUAT_START_IDX:LEFT_WRIST_QUAT_END_IDX] = left_wrist_quat.astype(np.float32)
    actions[:, RIGHT_WRIST_POS_START_IDX:RIGHT_WRIST_POS_END_IDX] = right_wrist_pos_pelvis.astype(np.float32)
    actions[:, RIGHT_WRIST_QUAT_START_IDX:RIGHT_WRIST_QUAT_END_IDX] = right_wrist_quat_pelvis.astype(np.float32)
    actions[:, NAV_CMD_START_IDX:NAV_CMD_END_IDX] = 0.0  # keep base/pelvis fixed
    actions[:, BASE_HEIGHT_IDX] = np.float32(args.base_height)
    actions[:, TORSO_RPY_START_IDX:TORSO_RPY_END_IDX] = torso_rpy.astype(np.float32)

    with h5py.File(args.output_hdf5, "w") as f:
        grp = f.create_group("data")
        grp.attrs["total"] = int(n)
        grp.attrs["env_args"] = json.dumps({"env_name": "", "type": 2, "generator": "arm_follow_object_debug"})
        ep = grp.create_group(args.episode_name)
        ep.attrs["num_samples"] = int(n)
        ep.create_dataset("actions", data=actions, compression="gzip")

    debug = {
        "input": {
            "kin_traj_path": os.path.abspath(args.kin_traj_path),
            "object_name": object_name,
            "frames": int(n),
        },
        "config": {
            "base_pos_w": base_pos_w.tolist(),
            "base_yaw": float(args.base_yaw),
            "right_wrist_pos_obj": right_wrist_pos_obj.tolist(),
            "right_wrist_quat_obj_wxyz": right_wrist_quat_obj.tolist(),
            "right_wrist_quat_control": args.right_wrist_quat_control,
            "right_wrist_quat_pelvis_wxyz": right_wrist_quat_pelvis_const.tolist(),
        },
        "sanity": {
            "right_wrist_pos_pelvis_min": right_wrist_pos_pelvis.min(axis=0).tolist(),
            "right_wrist_pos_pelvis_max": right_wrist_pos_pelvis.max(axis=0).tolist(),
            "nav_cmd_abs_max": np.max(np.abs(actions[:, NAV_CMD_START_IDX:NAV_CMD_END_IDX]), axis=0).tolist(),
        },
        "output_hdf5": os.path.abspath(args.output_hdf5),
        "episode_name": args.episode_name,
    }

    if args.output_debug_json:
        with open(args.output_debug_json, "w", encoding="utf-8") as f:
            json.dump(debug, f, indent=2)

    print(f"[arm_follow] actions saved: {os.path.abspath(args.output_hdf5)}")
    if args.output_debug_json:
        print(f"[arm_follow] debug saved: {os.path.abspath(args.output_debug_json)}")


if __name__ == "__main__":
    main()
