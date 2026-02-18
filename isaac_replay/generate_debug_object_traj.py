#!/usr/bin/env python3
"""Generate synthetic object kinematic trajectories for Isaac debug runs."""

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


def _parse_optional_csv_floats(text: str | None, expected_len: int, name: str) -> np.ndarray | None:
    if text is None:
        return None
    trimmed = text.strip()
    if not trimmed or trimmed.lower() in {"none", "null"}:
        return None
    return _parse_csv_floats(trimmed, expected_len, name)


def _yaw_to_quat_wxyz(yaw_rad: np.ndarray) -> np.ndarray:
    half = 0.5 * yaw_rad
    w = np.cos(half)
    z = np.sin(half)
    return np.stack([w, np.zeros_like(w), np.zeros_like(w), z], axis=1)


def _yaw_to_rotmat(yaw_rad: np.ndarray) -> np.ndarray:
    c = np.cos(yaw_rad)
    s = np.sin(yaw_rad)
    out = np.zeros((yaw_rad.shape[0], 3, 3), dtype=np.float64)
    out[:, 0, 0] = c
    out[:, 0, 1] = -s
    out[:, 1, 0] = s
    out[:, 1, 1] = c
    out[:, 2, 2] = 1.0
    return out


def _interp_keyframes(frames: int, key_ts: np.ndarray, key_pos: np.ndarray) -> np.ndarray:
    t = np.linspace(0.0, 1.0, frames)
    out = np.zeros((frames, 3), dtype=np.float64)
    for dim in range(3):
        out[:, dim] = np.interp(t, key_ts, key_pos[:, dim])
    return out


def _scene_defaults(scene: str) -> tuple[np.ndarray, np.ndarray]:
    if scene == "galileo_locomanip":
        return np.array([0.5785, 0.18, 0.0707], dtype=np.float64), np.array([-0.2450, -1.6272, 0.0707], dtype=np.float64)
    if scene == "kitchen_pick_and_place":
        return np.array([0.4, 0.0, 0.1], dtype=np.float64), np.array([0.2, 0.4, 0.1], dtype=np.float64)
    return np.array([0.4, 0.0, 0.1], dtype=np.float64), np.array([0.6, -0.2, 0.1], dtype=np.float64)


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="Output .npz path.")
    parser.add_argument("--output-debug-json", default=None, help="Optional debug json output path.")
    parser.add_argument("--object-name", default="brown_box", help="object_name stored in trajectory npz.")
    parser.add_argument("--fps", type=float, default=50.0, help="Trajectory FPS.")
    parser.add_argument("--duration-sec", type=float, default=8.0, help="Trajectory duration in seconds.")
    parser.add_argument(
        "--pattern",
        choices=["line", "circle", "lift_place"],
        default="lift_place",
        help="Synthetic trajectory pattern.",
    )
    parser.add_argument(
        "--scene-preset",
        choices=["none", "galileo_locomanip", "kitchen_pick_and_place"],
        default="galileo_locomanip",
        help="Preset start/end values matching an Isaac scene.",
    )
    parser.add_argument("--start-pos-w", default=None, help="Optional override for start position (x,y,z).")
    parser.add_argument("--end-pos-w", default=None, help="Optional override for end position (x,y,z).")
    parser.add_argument("--center-xy", default=None, help="Circle center override (x,y).")
    parser.add_argument("--radius", type=float, default=0.25, help="Circle radius.")
    parser.add_argument("--circle-turns", type=float, default=1.0, help="Number of turns for circle pattern.")
    parser.add_argument("--lift-height", type=float, default=0.35, help="Extra z height used by lift_place pattern.")
    parser.add_argument("--yaw-deg", type=float, default=0.0, help="Initial object yaw angle in degrees.")
    parser.add_argument("--yaw-spin-deg", type=float, default=0.0, help="Additional yaw spin across sequence.")
    parser.add_argument(
        "--circle-face-tangent",
        action="store_true",
        help="For circle pattern, orient object along tangent direction.",
    )
    return parser


def main() -> None:
    args = _make_parser().parse_args()
    if args.fps <= 0:
        raise ValueError("fps must be > 0")
    if args.duration_sec <= 0:
        raise ValueError("duration-sec must be > 0")

    frames = int(round(args.fps * args.duration_sec)) + 1
    if frames < 2:
        raise ValueError("Trajectory should contain at least 2 frames.")

    preset_start, preset_end = _scene_defaults(args.scene_preset)
    start = _parse_optional_csv_floats(args.start_pos_w, 3, "start_pos_w")
    end = _parse_optional_csv_floats(args.end_pos_w, 3, "end_pos_w")
    start = preset_start if start is None else start
    end = preset_end if end is None else end

    t = np.linspace(0.0, 1.0, frames, dtype=np.float64)
    if args.pattern == "line":
        pos = (1.0 - t)[:, None] * start[None, :] + t[:, None] * end[None, :]
        yaw = np.deg2rad(args.yaw_deg + args.yaw_spin_deg * t)
    elif args.pattern == "lift_place":
        z_peak = max(float(start[2]), float(end[2])) + float(args.lift_height)
        key_ts = np.array([0.0, 0.25, 0.75, 1.0], dtype=np.float64)
        key_pos = np.array(
            [
                start,
                np.array([start[0], start[1], z_peak], dtype=np.float64),
                np.array([end[0], end[1], z_peak], dtype=np.float64),
                end,
            ],
            dtype=np.float64,
        )
        pos = _interp_keyframes(frames, key_ts, key_pos)
        yaw = np.deg2rad(args.yaw_deg + args.yaw_spin_deg * t)
    else:
        center_xy = _parse_optional_csv_floats(args.center_xy, 2, "center_xy")
        if center_xy is None:
            center_xy = 0.5 * (start[:2] + end[:2])
        theta = 2.0 * math.pi * args.circle_turns * t
        pos = np.zeros((frames, 3), dtype=np.float64)
        pos[:, 0] = center_xy[0] + args.radius * np.cos(theta)
        pos[:, 1] = center_xy[1] + args.radius * np.sin(theta)
        pos[:, 2] = start[2]
        if args.circle_face_tangent:
            yaw = np.arctan2(np.gradient(pos[:, 1]), np.gradient(pos[:, 0]))
        else:
            yaw = np.deg2rad(args.yaw_deg + args.yaw_spin_deg * t)

    quat_wxyz = _yaw_to_quat_wxyz(yaw).astype(np.float32)
    rot_mat = _yaw_to_rotmat(yaw).astype(np.float32)

    out_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    np.savez_compressed(
        out_path,
        object_name=np.array(args.object_name),
        fps=np.array([args.fps], dtype=np.float32),
        object_pos_w=pos.astype(np.float32),
        object_quat_wxyz=quat_wxyz,
        object_rot_mat_w=rot_mat,
    )

    summary: dict[str, Any] = {
        "output": out_path,
        "pattern": args.pattern,
        "scene_preset": args.scene_preset,
        "frames": int(frames),
        "fps": float(args.fps),
        "duration_sec": float(args.duration_sec),
        "start_pos_w": pos[0].tolist(),
        "end_pos_w": pos[-1].tolist(),
        "pos_min_w": pos.min(axis=0).tolist(),
        "pos_max_w": pos.max(axis=0).tolist(),
    }

    if args.output_debug_json:
        debug_path = os.path.abspath(args.output_debug_json)
        os.makedirs(os.path.dirname(debug_path) or ".", exist_ok=True)
        with open(debug_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    print(f"[debug_traj] saved: {out_path}")
    print(f"[debug_traj] frames={frames}, pattern={args.pattern}, start={pos[0].tolist()}, end={pos[-1].tolist()}")


if __name__ == "__main__":
    main()
