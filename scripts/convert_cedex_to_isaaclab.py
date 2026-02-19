#!/usr/bin/env python3
"""Convert a CEDex generated_grasps .pt into Isaac replay wrist pose arguments."""

from __future__ import annotations

import argparse
import json
import math
import os
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


def _pose_matrix_from_pos_quat(pos: np.ndarray, quat_wxyz: np.ndarray) -> np.ndarray:
    t = np.eye(4, dtype=np.float64)
    t[:3, :3] = _quat_to_rotmat_wxyz(quat_wxyz)
    t[:3, 3] = np.asarray(pos, dtype=np.float64)
    return t


def _normalize_rows(v: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return v / norms


def _rot6d_to_rotmat(rot6d: np.ndarray) -> np.ndarray:
    data = np.asarray(rot6d, dtype=np.float64)
    if data.ndim == 1:
        data = data[None, :]
    if data.ndim != 2 or data.shape[1] != 6:
        raise ValueError(f"rot6d should be [N,6], got {data.shape}")
    x = _normalize_rows(data[:, 0:3])
    y_raw = data[:, 3:6]
    z = _normalize_rows(np.cross(x, y_raw))
    y = np.cross(z, x)
    return np.stack([x, y, z], axis=2)


def _format_csv(v: np.ndarray) -> str:
    return ",".join(f"{float(x):.8f}" for x in v)


def _load_cedex_entry(grasp_pt_path: str, grasp_index: int) -> tuple[np.ndarray, dict[str, Any]]:
    try:
        import torch
    except Exception as exc:
        raise RuntimeError("This script requires torch to load .pt files.") from exc

    payload = torch.load(grasp_pt_path, map_location="cpu")
    if isinstance(payload, dict):
        entries = [payload]
    elif isinstance(payload, list):
        entries = payload
    else:
        raise TypeError(f"Unsupported payload type: {type(payload)} (expected dict or list).")
    if len(entries) == 0:
        raise ValueError(f"No entries found in {grasp_pt_path}")

    idx = int(grasp_index)
    if idx < 0:
        idx += len(entries)
    if idx < 0 or idx >= len(entries):
        raise IndexError(f"grasp_index={grasp_index} out of range [0, {len(entries)-1}]")

    entry = entries[idx]
    q_raw = entry["q"] if isinstance(entry, dict) and "q" in entry else entry
    q = np.asarray(q_raw, dtype=np.float64).reshape(-1)
    if q.shape[0] < 9:
        raise ValueError(f"Expected q with at least 9 values (xyz+rot6d), got {q.shape[0]}")

    meta: dict[str, Any] = {
        "grasp_pt_path": os.path.abspath(grasp_pt_path),
        "selected_index": int(idx),
        "num_entries": int(len(entries)),
        "q_dim": int(q.shape[0]),
    }
    if isinstance(entry, dict):
        meta["entry_keys"] = sorted(entry.keys())
        if "object_name" in entry:
            meta["object_name"] = str(entry["object_name"])
        if "robot_name" in entry:
            meta["robot_name"] = str(entry["robot_name"])
    return q, meta


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cedex-grasp-pt", required=True, help="Path to CEDex generated_grasps/*.pt file.")
    parser.add_argument("--grasp-index", type=int, default=0, help="Index of grasp entry to use.")
    parser.add_argument(
        "--wrist-pos-offset",
        default="0.0,0.0,0.0",
        help="Offset xyz from CEDex hand frame to replay wrist frame.",
    )
    parser.add_argument(
        "--wrist-quat-offset-wxyz",
        default="1.0,0.0,0.0,0.0",
        help="Offset quaternion from CEDex hand frame to replay wrist frame.",
    )
    parser.add_argument("--output-json", default=None, help="Optional output json path.")
    return parser


def main() -> None:
    args = _make_parser().parse_args()
    q, meta = _load_cedex_entry(args.cedex_grasp_pt, args.grasp_index)

    hand_pos_obj = q[0:3]
    hand_rot_obj = _rot6d_to_rotmat(q[3:9])[0]
    hand_quat_obj = _rotmat_to_quat_wxyz(hand_rot_obj)

    wrist_pos_offset = _parse_csv_floats(args.wrist_pos_offset, 3, "wrist_pos_offset")
    wrist_quat_offset = _normalize_quat_wxyz(
        _parse_csv_floats(args.wrist_quat_offset_wxyz, 4, "wrist_quat_offset_wxyz")
    )

    t_hand_obj = _pose_matrix_from_pos_quat(hand_pos_obj, hand_quat_obj)
    t_wrist_hand = _pose_matrix_from_pos_quat(wrist_pos_offset, wrist_quat_offset)
    t_wrist_obj = t_hand_obj @ t_wrist_hand

    wrist_pos_obj = t_wrist_obj[:3, 3]
    wrist_quat_obj = _rotmat_to_quat_wxyz(t_wrist_obj[:3, :3])

    out = {
        "meta": meta,
        "hand_pose_obj": {
            "pos": hand_pos_obj.tolist(),
            "quat_wxyz": hand_quat_obj.tolist(),
        },
        "wrist_pose_obj": {
            "pos": wrist_pos_obj.tolist(),
            "quat_wxyz": wrist_quat_obj.tolist(),
        },
        "replay_args": {
            "right_wrist_pos_obj": _format_csv(wrist_pos_obj),
            "right_wrist_quat_obj_wxyz": _format_csv(wrist_quat_obj),
        },
    }

    if args.output_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_json)) or ".", exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print(f"[convert_cedex] json saved: {os.path.abspath(args.output_json)}")

    print("[convert_cedex] Replay arguments:")
    print(f"--right-wrist-pos-obj={out['replay_args']['right_wrist_pos_obj']}")
    print(f"--right-wrist-quat-obj-wxyz={out['replay_args']['right_wrist_quat_obj_wxyz']}")
    print("[convert_cedex] Full output:")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
