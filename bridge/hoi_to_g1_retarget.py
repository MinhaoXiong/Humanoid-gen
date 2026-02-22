#!/usr/bin/env python3
"""HOI-to-G1 retargeting: convert hoifhli/HOIDiNi pkl into G1-compatible
object kinematic trajectories (object_kinematic_traj.npz).

Pipeline:
  1. Load HOI pkl (unified loader for hoifhli & HOIDiNi)
  2. Coordinate transform (Y-up → Z-up if needed)
  3. Compute human wrist→object relative pose per frame
  4. Retarget human wrist to G1 workspace via MuJoCo IK
  5. Reconstruct object trajectory preserving relative pose
  6. Align to scene table, clip, resample, output NPZ
"""

import argparse
import json
import math
import os
import pickle

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PACK_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
G1_XML = "/home/ubuntu/DATA2/workspace/xmh/spider/spider/assets/robots/unitree_g1/scene_simple.xml"

SMPLX_PELVIS = 0
SMPLX_RIGHT_WRIST = 21

import sys as _sys
if PACK_ROOT not in _sys.path:
    _sys.path.insert(0, PACK_ROOT)
from bridge.scene_config import scene_defaults_for_retarget as _get_scene_defaults

TARGET_FPS = 50.0
HOI_FPS = 30.0


# ---------------------------------------------------------------------------
# Rotation helpers
# ---------------------------------------------------------------------------
def rotmat_to_quat_wxyz(R):
    q = Rotation.from_matrix(R).as_quat()  # xyzw
    return np.array([q[3], q[0], q[1], q[2]])


def batch_rotmat_to_quat_wxyz(R):
    q = Rotation.from_matrix(R).as_quat()  # [N,4] xyzw
    return np.column_stack([q[:, 3], q[:, 0], q[:, 1], q[:, 2]])


# ---------------------------------------------------------------------------
# Coordinate transform: Y-up → Z-up
# ---------------------------------------------------------------------------
_C = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64)


def yup_to_zup_pos(p):
    """(x,y,z) → (x, -z, y)"""
    out = np.empty_like(p)
    out[..., 0] = p[..., 0]
    out[..., 1] = -p[..., 2]
    out[..., 2] = p[..., 1]
    return out


def yup_to_zup_rot(R):
    if R.ndim == 2:
        return _C @ R @ _C.T
    return np.einsum("ij,njk,lk->nil", _C, R, _C)


# ---------------------------------------------------------------------------
# Unified HOI loader
# ---------------------------------------------------------------------------
def _np(x):
    if hasattr(x, "cpu"):
        x = x.cpu()
    if hasattr(x, "numpy"):
        return x.numpy()
    return np.asarray(x)


def load_hoi_pkl(path):
    """Load hoifhli or HOIDiNi pkl → dict with Z-up numpy arrays."""
    import torch
    _orig = torch.load
    torch.load = lambda *a, **kw: _orig(*a, **{**kw, "map_location": "cpu"})
    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
    finally:
        torch.load = _orig

    obj_pos = _np(data["obj_pos"]).astype(np.float64)
    obj_rot = _np(data["obj_rot_mat"]).astype(np.float64)
    name = str(data.get("object_name", "unknown"))

    jnts = None
    if "human_jnts_pos" in data:
        jnts = _np(data["human_jnts_pos"]).astype(np.float64)
        # Auto-detect Y-up: pelvis mean height in Y > 0.5
        if jnts[:, SMPLX_PELVIS, 1].mean() > 0.5:
            jnts = yup_to_zup_pos(jnts)
            obj_pos = yup_to_zup_pos(obj_pos)
            obj_rot = yup_to_zup_rot(obj_rot)

    return dict(jnts=jnts, obj_pos=obj_pos, obj_rot=obj_rot, name=name)


# ---------------------------------------------------------------------------
# Resample
# ---------------------------------------------------------------------------
def resample_pos(p, src_fps, tgt_fps):
    T = p.shape[0]
    dur = (T - 1) / src_fps
    N = max(2, int(round(dur * tgt_fps)) + 1)
    ts, tt = np.linspace(0, dur, T), np.linspace(0, dur, N)
    out = np.empty((N, p.shape[1]))
    for d in range(p.shape[1]):
        out[:, d] = np.interp(tt, ts, p[:, d])
    return out


def resample_rot(R, src_fps, tgt_fps):
    T = R.shape[0]
    dur = (T - 1) / src_fps
    N = max(2, int(round(dur * tgt_fps)) + 1)
    ts, tt = np.linspace(0, dur, T), np.linspace(0, dur, N)
    return Slerp(ts, Rotation.from_matrix(R))(tt).as_matrix()


# ---------------------------------------------------------------------------
# G1 MuJoCo IK retargeter
# ---------------------------------------------------------------------------
class G1Retargeter:
    """Retarget human wrist trajectory to G1 via MuJoCo mocap IK."""

    def __init__(self, xml_path=G1_XML, base_height=0.75):
        spec = mujoco.MjSpec.from_file(xml_path)
        mb = spec.worldbody.add_body(name="tgt_rwrist", mocap=True)
        mb.add_site(name="tgt_rwrist", type=mujoco.mjtGeom.mjGEOM_SPHERE,
                     size=[0.01]*3, rgba=[0,1,0,0.5])
        eq = spec.add_equality()
        eq.type = mujoco.mjtEq.mjEQ_WELD
        eq.name = "weld_rw"
        eq.name1 = "tgt_rwrist"
        eq.name2 = "right_wrist_yaw_link"
        eq.objtype = mujoco.mjtObj.mjOBJ_BODY
        eq.solimp = [0.9, 0.95, 0.001, 0.5, 2]
        eq.solref = [0.02, 1]
        self.m = spec.compile()
        self.m.opt.timestep = 0.002
        self.m.opt.iterations = 50
        self.m.opt.ls_iterations = 100
        self.m.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
        self.m.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_ACTUATION
        self.d = mujoco.MjData(self.m)
        self.base_h = base_height
        bid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "tgt_rwrist")
        self._mid = self.m.body_mocapid[bid]
        self._sid = mujoco.mj_name2id(
            self.m, mujoco.mjtObj.mjOBJ_SITE, "right_palm")

    def _reset(self):
        mujoco.mj_resetData(self.m, self.d)
        self.d.qpos[2] = self.base_h
        self.d.qpos[3] = 1.0
        mujoco.mj_forward(self.m, self.d)

    def run(self, targets_w, n_sub=10):
        """targets_w: [T,3]. Returns palm_pos [T,3], palm_rot [T,3,3]."""
        T = targets_w.shape[0]
        pos = np.zeros((T, 3))
        rot = np.zeros((T, 3, 3))
        self._reset()
        for t in range(T):
            self.d.mocap_pos[self._mid] = targets_w[t]
            for _ in range(n_sub):
                mujoco.mj_step(self.m, self.d)
            mujoco.mj_kinematics(self.m, self.d)
            pos[t] = self.d.site_xpos[self._sid].copy()
            rot[t] = self.d.site_xmat[self._sid].reshape(3, 3).copy()
        return pos, rot


# ---------------------------------------------------------------------------
# Object trajectory reconstruction
# ---------------------------------------------------------------------------
def reconstruct_obj_traj(h_wrist, h_obj_pos, h_obj_rot, g1_palm, g1_rot):
    """Preserve hand→object relative pose, reconstruct in G1 frame.
    All inputs [T, ...]. Returns obj_pos_g1 [T,3], obj_rot_g1 [T,3,3]."""
    T = h_wrist.shape[0]
    op = np.zeros((T, 3))
    oR = np.zeros((T, 3, 3))
    for t in range(T):
        dp = h_obj_pos[t] - h_wrist[t]
        op[t] = g1_palm[t] + g1_rot[t] @ dp / max(np.linalg.norm(dp), 1e-6) * np.linalg.norm(g1_rot[t].T @ dp)
        oR[t] = g1_rot[t] @ h_obj_rot[t]
    return op, oR


def reconstruct_obj_traj_simple(h_wrist, h_obj_pos, h_obj_rot, g1_palm, g1_rot):
    """Simpler version: just translate relative offset, keep original rotation."""
    T = h_wrist.shape[0]
    # Compute scale factor: ratio of G1 arm reach to human arm reach
    h_range = np.ptp(h_wrist, axis=0)
    g_range = np.ptp(g1_palm, axis=0)
    scale = np.where(h_range > 1e-4, g_range / h_range, 1.0)

    op = np.zeros((T, 3))
    for t in range(T):
        dp = h_obj_pos[t] - h_wrist[t]
        op[t] = g1_palm[t] + dp * scale
    return op, h_obj_rot.copy()


# ---------------------------------------------------------------------------
# Align & clip object trajectory to scene
# ---------------------------------------------------------------------------
def align_and_clip(obj_pos, obj_rot, scene):
    """Align first frame to scene table, clip to workspace bounds."""
    sd = _get_scene_defaults(scene)
    align = np.array(sd.get("align_pos", [0.4, 0.0, 0.1]))

    # Translate so first frame lands at align position
    offset = align - obj_pos[0]
    obj_pos = obj_pos + offset

    # Clip Z
    zmin, zmax = sd.get("clip_z", (None, None))
    if zmin is not None:
        obj_pos[:, 2] = np.maximum(obj_pos[:, 2], zmin)
    if zmax is not None:
        obj_pos[:, 2] = np.minimum(obj_pos[:, 2], zmax)

    # Clip XY
    xy_min, xy_max = sd.get("clip_xy", (None, None))
    if xy_min is not None:
        xy_min = np.array(xy_min)
        obj_pos[:, :2] = np.maximum(obj_pos[:, :2], xy_min)
    if xy_max is not None:
        xy_max = np.array(xy_max)
        obj_pos[:, :2] = np.minimum(obj_pos[:, :2], xy_max)

    return obj_pos, obj_rot


# ---------------------------------------------------------------------------
# Save standard NPZ
# ---------------------------------------------------------------------------
def save_npz(path, obj_pos, obj_rot, object_name, fps=TARGET_FPS):
    quat = batch_rotmat_to_quat_wxyz(obj_rot)
    # Ensure w >= 0
    quat[quat[:, 0] < 0] *= -1
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savez(
        path,
        object_name=object_name,
        fps=np.array([fps], dtype=np.float32),
        object_pos_w=obj_pos.astype(np.float32),
        object_quat_wxyz=quat.astype(np.float32),
        object_rot_mat_w=obj_rot.astype(np.float32),
    )
    print(f"Saved {path}  frames={obj_pos.shape[0]}  fps={fps}")


# ---------------------------------------------------------------------------
# Fallback: scale-only adaptation (no Spider IK, for HOIDiNi-only pkls)
# ---------------------------------------------------------------------------
def adapt_obj_only(obj_pos, obj_rot, scene):
    """When no human body data, use heuristic scaling to G1 workspace."""
    sd = _get_scene_defaults(scene)
    align = np.array(sd.get("align_pos", [0.4, 0.0, 0.1]))
    # Center trajectory around align position
    center = obj_pos.mean(axis=0)
    # Scale to fit G1 workspace (~0.3m reach)
    span = np.ptp(obj_pos, axis=0)
    max_span = max(span.max(), 1e-4)
    scale = min(0.25 / max_span, 1.0)  # cap at 0.25m range
    obj_pos = (obj_pos - center) * scale + align
    return obj_pos, obj_rot


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="HOI pkl → G1 object trajectory NPZ")
    p.add_argument("--hoi-pickle", required=True)
    p.add_argument("--output-npz", required=True)
    p.add_argument("--output-debug-json", default=None)
    p.add_argument("--scene", default="kitchen_pick_and_place")
    p.add_argument("--hoi-fps", type=float, default=HOI_FPS)
    p.add_argument("--target-fps", type=float, default=TARGET_FPS)
    p.add_argument("--g1-xml", default=G1_XML)
    p.add_argument("--no-retarget", action="store_true",
                   help="Skip IK retarget, use scale-only fallback")
    p.add_argument("--object-name-override", default=None)
    args = p.parse_args()

    print(f"[retarget] Loading {args.hoi_pickle}")
    hoi = load_hoi_pkl(args.hoi_pickle)
    obj_name = args.object_name_override or hoi["name"]
    print(f"[retarget] Object: {obj_name}  frames: {hoi['obj_pos'].shape[0]}"
          f"  has_body: {hoi['jnts'] is not None}")

    sd = _get_scene_defaults(args.scene)
    base_h = sd.get("base_height", 0.75)

    if hoi["jnts"] is not None and not args.no_retarget:
        # --- Full retarget path ---
        h_wrist = hoi["jnts"][:, SMPLX_RIGHT_WRIST]  # [T,3]
        h_obj = hoi["obj_pos"]
        h_rot = hoi["obj_rot"]

        # Scale human wrist targets into G1 reachable zone
        # G1 palm is ~0.3m from pelvis center; human wrist ~0.6m
        pelvis = hoi["jnts"][:, SMPLX_PELVIS]
        h_rel = h_wrist - pelvis  # wrist relative to pelvis
        # Scale to G1 proportions and place in front of robot
        g1_targets = h_rel * 0.5  # ~half human scale
        g1_targets[:, 0] += 0.35  # forward
        g1_targets[:, 2] += base_h  # at pelvis height

        print(f"[retarget] Running G1 IK for {len(g1_targets)} frames...")
        rt = G1Retargeter(args.g1_xml, base_height=base_h)
        g1_palm, g1_rot = rt.run(g1_targets)

        print("[retarget] Reconstructing object trajectory...")
        obj_pos, obj_rot = reconstruct_obj_traj_simple(
            h_wrist, h_obj, h_rot, g1_palm, g1_rot)
        method = "spider_ik_retarget"
    else:
        # --- Fallback: scale-only ---
        print("[retarget] No body data or --no-retarget, using scale fallback")
        obj_pos, obj_rot = hoi["obj_pos"], hoi["obj_rot"]
        method = "scale_only"

    # Align to scene and clip
    obj_pos, obj_rot = align_and_clip(obj_pos, obj_rot, args.scene)

    # Resample to target FPS
    obj_pos = resample_pos(obj_pos, args.hoi_fps, args.target_fps)
    obj_rot = resample_rot(obj_rot, args.hoi_fps, args.target_fps)

    # Save
    save_npz(args.output_npz, obj_pos, obj_rot, obj_name, args.target_fps)

    # Debug JSON
    if args.output_debug_json:
        info = dict(
            method=method, scene=args.scene, object_name=obj_name,
            hoi_pickle=args.hoi_pickle, frames=int(obj_pos.shape[0]),
            fps=args.target_fps,
            obj_pos_range=dict(
                x=[float(obj_pos[:,0].min()), float(obj_pos[:,0].max())],
                y=[float(obj_pos[:,1].min()), float(obj_pos[:,1].max())],
                z=[float(obj_pos[:,2].min()), float(obj_pos[:,2].max())],
            ),
        )
        os.makedirs(os.path.dirname(args.output_debug_json) or ".", exist_ok=True)
        with open(args.output_debug_json, "w") as f:
            json.dump(info, f, indent=2)
        print(f"[retarget] Debug info → {args.output_debug_json}")

    print("[retarget] Done.")


if __name__ == "__main__":
    main()
