#!/usr/bin/env python3
"""Convert a HOIDiNi final.pickle to bridge/build_replay.py compatible pkl."""

from __future__ import annotations

import argparse
import json
import os
import pickle
from typing import Any

import numpy as np
import torch


def _to_numpy(x: Any) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _axis_angle_to_rotmat(axis_angle: np.ndarray) -> np.ndarray:
    vec = np.asarray(axis_angle, dtype=np.float64)
    if vec.shape != (3,):
        raise ValueError(f"axis-angle must be [3], got {vec.shape}")
    theta = float(np.linalg.norm(vec))
    if theta < 1e-12:
        return np.eye(3, dtype=np.float64)
    axis = vec / theta
    x, y, z = axis
    c = float(np.cos(theta))
    s = float(np.sin(theta))
    one_c = 1.0 - c
    return np.array(
        [
            [c + x * x * one_c, x * y * one_c - z * s, x * z * one_c + y * s],
            [y * x * one_c + z * s, c + y * y * one_c, y * z * one_c - x * s],
            [z * x * one_c - y * s, z * y * one_c + x * s, c + z * z * one_c],
        ],
        dtype=np.float64,
    )


def _load_pickle_cpu(path: str) -> dict[str, Any]:
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

    if not isinstance(data, dict):
        raise ValueError(f"Expected dict in {path}, got {type(data)}")
    return data


def _extract_smpldata_fields(data: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    smpldata = data.get("smpldata")
    if smpldata is None:
        raise KeyError("HOIDiNi final pickle has no key 'smpldata'.")

    if isinstance(smpldata, dict):
        trans_obj = smpldata.get("trans_obj")
        poses_obj = smpldata.get("poses_obj")
    else:
        trans_obj = getattr(smpldata, "trans_obj", None)
        poses_obj = getattr(smpldata, "poses_obj", None)

    if trans_obj is None or poses_obj is None:
        raise KeyError("Missing smpldata.trans_obj or smpldata.poses_obj in HOIDiNi pickle.")

    trans_obj_np = _to_numpy(trans_obj).astype(np.float64)
    poses_obj_np = _to_numpy(poses_obj).astype(np.float64)

    if trans_obj_np.ndim != 2 or trans_obj_np.shape[1] != 3:
        raise ValueError(f"smpldata.trans_obj should be [T,3], got {trans_obj_np.shape}")
    if poses_obj_np.ndim != 2 or poses_obj_np.shape[1] != 3:
        raise ValueError(f"smpldata.poses_obj should be [T,3] axis-angle, got {poses_obj_np.shape}")
    if trans_obj_np.shape[0] != poses_obj_np.shape[0]:
        raise ValueError(
            f"Length mismatch: trans_obj {trans_obj_np.shape[0]} vs poses_obj {poses_obj_np.shape[0]}"
        )
    return trans_obj_np, poses_obj_np


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hoidini-final-pickle", required=True, help="Input HOIDiNi final.pickle")
    parser.add_argument("--output-pickle", required=True, help="Output bridge-compatible pickle")
    parser.add_argument(
        "--output-debug-json",
        default=None,
        help="Optional debug metadata json path",
    )
    parser.add_argument(
        "--object-name-override",
        default=None,
        help="Optional override for output object_name (e.g., cracker_box in kitchen scene)",
    )
    return parser


def main() -> None:
    args = _make_parser().parse_args()
    in_path = os.path.abspath(args.hoidini_final_pickle)
    out_path = os.path.abspath(args.output_pickle)
    dbg_path = os.path.abspath(args.output_debug_json) if args.output_debug_json else None

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    if dbg_path:
        os.makedirs(os.path.dirname(dbg_path) or ".", exist_ok=True)

    data = _load_pickle_cpu(in_path)
    obj_pos, obj_axis_angle = _extract_smpldata_fields(data)
    obj_rot_mat = np.stack([_axis_angle_to_rotmat(v) for v in obj_axis_angle], axis=0)

    src_object_name = str(data.get("object_name", "unknown_object"))
    dst_object_name = args.object_name_override.strip() if args.object_name_override else src_object_name

    out_data = {
        "obj_pos": obj_pos,
        "obj_rot_mat": obj_rot_mat,
        "object_name": dst_object_name,
    }
    with open(out_path, "wb") as f:
        pickle.dump(out_data, f)

    if dbg_path:
        debug = {
            "input_final_pickle": in_path,
            "output_bridge_pickle": out_path,
            "source_object_name": src_object_name,
            "output_object_name": dst_object_name,
            "text": str(data.get("text", "")),
            "frames": int(obj_pos.shape[0]),
            "source_pos_min": obj_pos.min(axis=0).tolist(),
            "source_pos_max": obj_pos.max(axis=0).tolist(),
        }
        with open(dbg_path, "w", encoding="utf-8") as f:
            json.dump(debug, f, indent=2, ensure_ascii=False)

    print(f"[convert_hoidini] input: {in_path}")
    print(f"[convert_hoidini] output: {out_path}")
    print(f"[convert_hoidini] frames: {obj_pos.shape[0]}")
    print(f"[convert_hoidini] object: {src_object_name} -> {dst_object_name}")


if __name__ == "__main__":
    main()
