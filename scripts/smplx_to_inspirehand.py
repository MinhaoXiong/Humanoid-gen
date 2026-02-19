#!/usr/bin/env python3
"""Module A: SMPLX/MANO → InspireHand retargeting.

Converts human hand poses (SMPLX 45-param axis-angle or MANO 5-fingertip keypoints)
to InspireHand 12 joint angles + wrist SE3 pose, compatible with BODex seed_config.

Approach: MuJoCo constraint-based IK (same as spider/preprocess/ik.py).
  1. Load spider's InspireHand MuJoCo model (right.xml)
  2. Place mocap bodies at MANO fingertip 3D positions
  3. MuJoCo constraint solver finds InspireHand joint angles
  4. Extract 12 joint angles + wrist SE3

References:
  - spider/preprocess/ik.py: constraint-based IK
  - spider/assets/robots/inspire/right.xml: InspireHand MuJoCo model
"""

import os
import copy

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

# ── Paths ──────────────────────────────────────────────────────────────────
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SPIDER_ROOT = os.path.join(
    os.path.dirname(_THIS_DIR), "repos", "spider"  # fallback
)
# Allow override via env var
SPIDER_ROOT = os.environ.get(
    "SPIDER_ROOT",
    os.path.join(os.path.dirname(_THIS_DIR), "..", "spider"),
)
INSPIRE_XML = os.path.join(SPIDER_ROOT, "spider", "assets", "robots", "inspire", "right.xml")

# ── MANO fingertip indices (standard 21-joint MANO) ───────────────────────
# 0=wrist, 1-4=thumb, 5-8=index, 9-12=middle, 13-16=ring, 17-20=pinky
# Tips: thumb=4, index=8, middle=12, ring=16, pinky=20
MANO_TIP_INDICES = [4, 8, 12, 16, 20]

# SMPLX hand joints: 15 joints × 3 axis-angle = 45 params
# Joint order: index(3), index(3), index(3), middle(3)×3, pinky(3)×3, ring(3)×3, thumb(3)×3
SMPLX_HAND_NUM_JOINTS = 15

# InspireHand joint names in spider's MuJoCo model (right hand)
INSPIRE_FINGER_JOINTS = [
    "right_thumb_proximal_yaw_joint",
    "right_thumb_proximal_pitch_joint",
    "right_thumb_intermediate_joint",
    "right_thumb_distal_joint",
    "right_index_proximal_joint",
    "right_index_intermediate_joint",
    "right_middle_proximal_joint",
    "right_middle_intermediate_joint",
    "right_ring_proximal_joint",
    "right_ring_intermediate_joint",
    "right_pinky_proximal_joint",
    "right_pinky_intermediate_joint",
]

# Wrist 6-DOF joints (translation + rotation)
INSPIRE_WRIST_JOINTS = [
    "right_pos_x",
    "right_pos_y",
    "right_pos_z",
    "right_rot_x",
    "right_rot_y",
    "right_rot_z",
]

# Spider fingertip site names in InspireHand model
INSPIRE_TIP_SITES = [
    "right_thumb_tip",
    "right_index_tip",
    "right_middle_tip",
    "right_ring_tip",
    "right_pinky_tip",
]


def _resolve_inspire_xml():
    """Find the InspireHand MuJoCo XML."""
    candidates = [
        INSPIRE_XML,
        os.path.join(SPIDER_ROOT, "spider", "assets", "robots", "inspire", "right.xml"),
        os.path.join(_SPIDER_ROOT, "spider", "assets", "robots", "inspire", "right.xml"),
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(
        f"InspireHand right.xml not found. Set SPIDER_ROOT env var. Tried: {candidates}"
    )


class InspireHandRetargeter:
    """Retarget MANO/SMPLX hand poses to InspireHand 12 joint angles + wrist SE3."""

    def __init__(self, xml_path: str | None = None, sim_dt: float = 0.005):
        xml_path = xml_path or _resolve_inspire_xml()
        self.sim_dt = sim_dt

        # Build IK model with mocap bodies + equality constraints
        self.ik_model, self.ik_data = self._build_ik_model(xml_path)

        # Cache joint indices and qpos addresses
        self._finger_joint_ids = [
            mujoco.mj_name2id(self.ik_model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in INSPIRE_FINGER_JOINTS
        ]
        self._wrist_joint_ids = [
            mujoco.mj_name2id(self.ik_model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in INSPIRE_WRIST_JOINTS
        ]
        self._finger_qpos_addrs = [
            self.ik_model.jnt_qposadr[jid] for jid in self._finger_joint_ids
        ]
        self._wrist_qpos_addrs = [
            self.ik_model.jnt_qposadr[jid] for jid in self._wrist_joint_ids
        ]

        # Mocap body indices (palm + 5 fingertips)
        self._palm_mocap_id = self._get_mocap_id("mocap_palm")
        self._tip_mocap_ids = [
            self._get_mocap_id(f"mocap_{name.split('_')[1]}")
            for name in INSPIRE_TIP_SITES
        ]

        # Cache palm site offset in body frame (for wrist→palm conversion)
        palm_sid = mujoco.mj_name2id(self.ik_model, mujoco.mjtObj.mjOBJ_SITE, "right_palm")
        self._palm_offset_pos = self.ik_model.site_pos[palm_sid].copy()
        self._palm_offset_quat = self.ik_model.site_quat[palm_sid].copy()

        # Finger joint ranges for random init
        self._finger_ranges = np.array([
            self.ik_model.jnt_range[jid] for jid in self._finger_joint_ids
        ])

    def _get_mocap_id(self, body_name: str) -> int:
        body_id = mujoco.mj_name2id(self.ik_model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        return self.ik_model.body_mocapid[body_id]

    def _build_ik_model(self, xml_path: str):
        """Add mocap bodies + equality constraints for IK solving."""
        spec = mujoco.MjSpec.from_file(xml_path)

        # Disable joint limits for IK (spider does this too)
        for j in spec.joints:
            j.actfrclimited = 0

        # Sites to track: palm + 5 fingertips
        site_names = ["right_palm"] + list(INSPIRE_TIP_SITES)
        mocap_names = ["mocap_palm", "mocap_thumb", "mocap_index",
                       "mocap_middle", "mocap_ring", "mocap_pinky"]

        for i, (site_name, mocap_name) in enumerate(zip(site_names, mocap_names)):
            # Add mocap body
            b = spec.worldbody.add_body(name=mocap_name, mocap=True)
            b.add_site(
                name=mocap_name,
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=[0.008, 0.008, 0.008],
                rgba=[0.0, 1.0, 0.0, 0.5],
                group=1,
            )

            # Add equality constraint
            constraint_data = np.zeros(11)
            if "palm" in site_name:
                # Palm: WELD constraint (position + orientation)
                eq_type = mujoco.mjtEq.mjEQ_WELD
                constraint_data[10] = 10.0  # torque_scale
                e = spec.add_equality(
                    name=f"eq_{site_name}",
                    type=eq_type,
                    name1=site_name,
                    name2=mocap_name,
                    objtype=mujoco.mjtObj.mjOBJ_SITE,
                    data=constraint_data,
                )
                e.solref = [0.02, 1.0]
                e.solimp = [0.0, 0.95, 0.01, 0.5, 2.0]
            else:
                # Fingertip: CONNECT constraint (position only)
                eq_type = mujoco.mjtEq.mjEQ_CONNECT
                constraint_data[10] = 1.0
                e = spec.add_equality(
                    name=f"eq_{site_name}",
                    type=eq_type,
                    name1=site_name,
                    name2=mocap_name,
                    objtype=mujoco.mjtObj.mjOBJ_SITE,
                    data=constraint_data,
                )
                e.solref = [0.01, 1.0]
                e.solimp = [0.0, 0.95, 0.01, 0.5, 2.0]

        model = spec.compile()
        model.opt.timestep = self.sim_dt
        model.opt.iterations = 20
        model.opt.ls_iterations = 50
        model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
        model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_ACTUATION

        data = mujoco.MjData(model)
        return model, data

    def retarget_from_fingertips(
        self,
        wrist_pos: np.ndarray,
        wrist_quat: np.ndarray,
        fingertip_positions: np.ndarray,
        num_ik_steps: int = 60,
        num_init_guesses: int = 4,
    ) -> dict:
        """Retarget from MANO-style fingertip keypoints.

        Args:
            wrist_pos: [3] wrist position in world frame
            wrist_quat: [4] wrist quaternion (wxyz) in world frame
            fingertip_positions: [5, 3] fingertip positions (thumb, index, middle, ring, pinky)
            num_ik_steps: number of MuJoCo steps for IK convergence
            num_init_guesses: number of random initial guesses to try

        Returns:
            dict with keys:
                'finger_q_12': [12] InspireHand joint angles (spider MuJoCo order)
                'wrist_pos': [3] solved wrist position
                'wrist_quat': [4] solved wrist quaternion (wxyz)
                'wrist_mat': [4, 4] solved wrist SE3 matrix
                'seed_config_13': [13] BODex seed (7D wrist pose + 6D independent DOF)
        """
        best_qpos = None
        best_cost = np.inf

        # Compute palm site target from wrist pose (palm has offset in body frame)
        wrist_rot = Rotation.from_quat([wrist_quat[1], wrist_quat[2],
                                         wrist_quat[3], wrist_quat[0]])  # wxyz→xyzw
        palm_world_pos = wrist_pos + wrist_rot.apply(self._palm_offset_pos)
        # Palm site quat in world = wrist_rot * palm_offset_rot
        palm_offset_rot = Rotation.from_quat([
            self._palm_offset_quat[1], self._palm_offset_quat[2],
            self._palm_offset_quat[3], self._palm_offset_quat[0],
        ])
        palm_world_rot = wrist_rot * palm_offset_rot
        palm_world_quat_xyzw = palm_world_rot.as_quat()
        palm_world_quat = np.array([palm_world_quat_xyzw[3], palm_world_quat_xyzw[0],
                                     palm_world_quat_xyzw[1], palm_world_quat_xyzw[2]])

        # Cache tip site IDs
        tip_site_ids = [
            mujoco.mj_name2id(self.ik_model, mujoco.mjtObj.mjOBJ_SITE, name)
            for name in INSPIRE_TIP_SITES
        ]

        for trial in range(num_init_guesses):
            self.ik_data.qpos[:] = 0.0
            self.ik_data.qvel[:] = 0.0

            # Initialize wrist position and orientation
            for i, addr in enumerate(self._wrist_qpos_addrs[:3]):
                self.ik_data.qpos[addr] = wrist_pos[i]
            euler = wrist_rot.as_euler("xyz")
            for i, addr in enumerate(self._wrist_qpos_addrs[3:]):
                self.ik_data.qpos[addr] = euler[i]

            # Random finger initialization
            for j, addr in enumerate(self._finger_qpos_addrs):
                lo, hi = self._finger_ranges[j]
                self.ik_data.qpos[addr] = np.random.uniform(lo, hi)

            # Set mocap targets: palm site position (not wrist!)
            self.ik_data.mocap_pos[self._palm_mocap_id] = palm_world_pos
            self.ik_data.mocap_quat[self._palm_mocap_id] = palm_world_quat
            for i, mocap_id in enumerate(self._tip_mocap_ids):
                self.ik_data.mocap_pos[mocap_id] = fingertip_positions[i]

            # Run IK
            for _ in range(num_ik_steps):
                mujoco.mj_step(self.ik_model, self.ik_data)

            # Evaluate cost: sum of fingertip position errors
            mujoco.mj_forward(self.ik_model, self.ik_data)
            cost = sum(
                np.linalg.norm(self.ik_data.site_xpos[sid] - fingertip_positions[i])
                for i, sid in enumerate(tip_site_ids)
            )

            if cost < best_cost:
                best_cost = cost
                best_qpos = self.ik_data.qpos.copy()

        # Extract results from best solution
        finger_q = np.array([best_qpos[addr] for addr in self._finger_qpos_addrs])
        wrist_solved_pos = np.array([best_qpos[addr] for addr in self._wrist_qpos_addrs[:3]])
        wrist_solved_euler = np.array([best_qpos[addr] for addr in self._wrist_qpos_addrs[3:]])

        rot_solved = Rotation.from_euler("xyz", wrist_solved_euler)
        wrist_solved_quat = rot_solved.as_quat()  # xyzw
        wrist_solved_quat_wxyz = np.array([
            wrist_solved_quat[3], wrist_solved_quat[0],
            wrist_solved_quat[1], wrist_solved_quat[2],
        ])

        wrist_mat = np.eye(4)
        wrist_mat[:3, :3] = rot_solved.as_matrix()
        wrist_mat[:3, 3] = wrist_solved_pos

        # BODex seed: [7D wrist pose (pos + quat_wxyz) + 6D independent DOF]
        # InspireHand 6 independent DOF (BODex order):
        # [thumb_yaw, thumb_pitch, index, middle, ring, pinky]
        independent_6dof = np.array([
            finger_q[0],   # thumb_proximal_yaw
            finger_q[1],   # thumb_proximal_pitch
            finger_q[4],   # index_proximal
            finger_q[6],   # middle_proximal
            finger_q[8],   # ring_proximal
            finger_q[10],  # pinky_proximal
        ])

        seed_config = np.concatenate([
            wrist_solved_pos,
            wrist_solved_quat_wxyz,
            independent_6dof,
        ])

        return {
            "finger_q_12": finger_q,
            "wrist_pos": wrist_solved_pos,
            "wrist_quat": wrist_solved_quat_wxyz,
            "wrist_mat": wrist_mat,
            "seed_config_13": seed_config,
            "ik_cost": best_cost,
        }

    def retarget_from_smplx(
        self,
        hand_pose_45: np.ndarray,
        global_orient: np.ndarray,
        transl: np.ndarray,
        betas: np.ndarray | None = None,
        **kwargs,
    ) -> dict:
        """Retarget from SMPLX hand parameters.

        Args:
            hand_pose_45: [45] right hand pose (15 joints × 3 axis-angle)
            global_orient: [3] global orientation (axis-angle)
            transl: [3] global translation
            betas: [10] or [16] shape parameters (optional)

        Returns:
            Same as retarget_from_fingertips.
        """
        # Convert SMPLX hand pose to fingertip positions using forward kinematics
        wrist_pos, wrist_quat, fingertip_positions = smplx_hand_to_fingertips(
            hand_pose_45, global_orient, transl, betas
        )
        return self.retarget_from_fingertips(
            wrist_pos, wrist_quat, fingertip_positions, **kwargs
        )

    def retarget_sequence(
        self,
        wrist_positions: np.ndarray,
        wrist_quats: np.ndarray,
        fingertip_sequences: np.ndarray,
        **kwargs,
    ) -> dict:
        """Retarget a sequence of frames.

        Args:
            wrist_positions: [T, 3]
            wrist_quats: [T, 4] (wxyz)
            fingertip_sequences: [T, 5, 3]

        Returns:
            dict with arrays of shape [T, ...] for each output key.
        """
        T = wrist_positions.shape[0]
        results = {
            "finger_q_12": np.zeros((T, 12)),
            "wrist_pos": np.zeros((T, 3)),
            "wrist_quat": np.zeros((T, 4)),
            "wrist_mat": np.zeros((T, 4, 4)),
            "seed_config_13": np.zeros((T, 13)),
            "ik_cost": np.zeros(T),
        }

        for t in range(T):
            # For subsequent frames, use previous solution as warm start
            if t == 0:
                r = self.retarget_from_fingertips(
                    wrist_positions[t], wrist_quats[t],
                    fingertip_sequences[t], **kwargs,
                )
            else:
                # Warm start: skip random init, use previous qpos
                r = self.retarget_from_fingertips(
                    wrist_positions[t], wrist_quats[t],
                    fingertip_sequences[t],
                    num_init_guesses=1, **kwargs,
                )

            for key in results:
                results[key][t] = r[key]

        return results


def smplx_hand_to_fingertips(
    hand_pose_45: np.ndarray,
    global_orient: np.ndarray,
    transl: np.ndarray,
    betas: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert SMPLX hand parameters to wrist pose + fingertip positions.

    Uses SMPLX forward kinematics to get joint positions.

    Args:
        hand_pose_45: [45] right hand pose (15 joints × 3 axis-angle)
        global_orient: [3] global orientation (axis-angle)
        transl: [3] global translation
        betas: [10] or [16] shape parameters

    Returns:
        wrist_pos: [3]
        wrist_quat: [4] (wxyz)
        fingertip_positions: [5, 3] (thumb, index, middle, ring, pinky)
    """
    try:
        import smplx as smplx_lib
        import torch
    except ImportError:
        raise ImportError("smplx and torch required for SMPLX retargeting. "
                          "pip install smplx torch")

    # Find SMPLX model
    smplx_model_path = os.environ.get("SMPLX_MODEL_PATH", None)
    if smplx_model_path is None:
        # Try common locations
        candidates = [
            os.path.join(os.path.dirname(_THIS_DIR), "repos", "hoifhli_release",
                         "data", "smpl_all_models"),
            os.path.expanduser("~/.smplx/models"),
        ]
        for c in candidates:
            if os.path.isdir(c):
                smplx_model_path = c
                break
    if smplx_model_path is None:
        raise FileNotFoundError("SMPLX model not found. Set SMPLX_MODEL_PATH env var.")

    device = "cpu"
    model = smplx_lib.create(
        model_path=smplx_model_path,
        model_type="smplx",
        gender="male",
        batch_size=1,
        flat_hand_mean=True,
        use_pca=False,
    ).to(device)

    # Prepare inputs
    go = torch.tensor(global_orient, dtype=torch.float32).reshape(1, 3).to(device)
    tr = torch.tensor(transl, dtype=torch.float32).reshape(1, 3).to(device)
    rh = torch.tensor(hand_pose_45, dtype=torch.float32).reshape(1, 45).to(device)
    bt = None
    if betas is not None:
        bt = torch.tensor(betas, dtype=torch.float32).reshape(1, -1).to(device)

    output = model(
        global_orient=go,
        transl=tr,
        right_hand_pose=rh,
        betas=bt,
    )

    joints = output.joints[0].detach().cpu().numpy()  # [J, 3]

    # SMPLX joint indices for right hand:
    # Wrist = joint 21 (right_wrist)
    # Right hand fingertips: thumb=40, index=41, middle=42, ring=43, pinky=44
    # (These are the standard SMPLX joint indices)
    wrist_pos = joints[21]

    # Fingertip indices in SMPLX (right hand tips)
    # thumb_tip=40, index_tip=41, middle_tip=42, ring_tip=43, pinky_tip=44
    tip_indices = [40, 41, 42, 43, 44]
    fingertip_positions = joints[tip_indices]

    # Wrist orientation from global orient + body chain
    # Approximate: use the right wrist joint frame
    # For simplicity, use global_orient as base rotation
    rot = Rotation.from_rotvec(global_orient)
    quat_xyzw = rot.as_quat()
    wrist_quat = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])

    return wrist_pos, wrist_quat, fingertip_positions


def retarget_from_spider_npz(
    npz_path: str,
    xml_path: str | None = None,
    side: str = "right",
) -> dict:
    """Retarget from spider's trajectory_keypoints.npz format.

    Args:
        npz_path: path to trajectory_keypoints.npz
        xml_path: path to InspireHand MuJoCo XML
        side: "right" or "left"

    Returns:
        dict with retargeted sequence data.
    """
    data = np.load(npz_path)
    wrist_positions = data[f"qpos_wrist_{side}"][:, :3]  # [T, 3]
    wrist_quats = data[f"qpos_wrist_{side}"][:, 3:]      # [T, 4] wxyz
    fingertip_positions = data[f"qpos_finger_{side}"][:, :, :3]  # [T, 5, 3]

    retargeter = InspireHandRetargeter(xml_path=xml_path)
    return retargeter.retarget_sequence(
        wrist_positions, wrist_quats, fingertip_positions,
    )


# ── BODex integration helpers ─────────────────────────────────────────────

# Mapping from spider MuJoCo joint order to BODex order
# Spider (MuJoCo): thumb_yaw, thumb_pitch, thumb_inter, thumb_distal,
#                  index_prox, index_inter, middle_prox, middle_inter,
#                  ring_prox, ring_inter, pinky_prox, pinky_inter
# BODex: thumb_yaw, thumb_pitch, index, middle, ring, pinky,
#        thumb_inter, thumb_distal, index_inter, middle_inter, ring_inter, pinky_inter
SPIDER_TO_BODEX_12 = [0, 1, 4, 6, 8, 10, 2, 3, 5, 7, 9, 11]

# Mapping from spider 12-joint to URDF order (for Isaac Sim WBC)
# URDF: index_prox, index_inter, middle_prox, middle_inter, pinky_prox, pinky_inter,
#       ring_prox, ring_inter, thumb_yaw, thumb_pitch, thumb_inter, thumb_distal
SPIDER_TO_URDF_12 = [4, 5, 6, 7, 10, 11, 8, 9, 0, 1, 2, 3]


def to_bodex_seed(result: dict) -> np.ndarray:
    """Convert retargeting result to BODex seed_config [13].

    Returns [pos(3) + quat_wxyz(4) + 6_independent_dof].
    """
    return result["seed_config_13"]


def to_bodex_12dof(result: dict) -> np.ndarray:
    """Convert retargeting result to BODex 12-DOF order."""
    return result["finger_q_12"][SPIDER_TO_BODEX_12]


def to_urdf_12dof(result: dict) -> np.ndarray:
    """Convert retargeting result to URDF order (for Isaac Sim WBC)."""
    return result["finger_q_12"][SPIDER_TO_URDF_12]


# ── CLI ────────────────────────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(description="SMPLX → InspireHand retargeting")
    parser.add_argument("--input", required=True,
                        help="Path to trajectory_keypoints.npz or SMPLX params .npz")
    parser.add_argument("--output", default=None,
                        help="Output .npz path (default: input_retargeted.npz)")
    parser.add_argument("--xml", default=None,
                        help="Path to InspireHand right.xml")
    parser.add_argument("--side", default="right", choices=["right", "left"])
    parser.add_argument("--format", default="spider",
                        choices=["spider", "smplx"],
                        help="Input format")
    args = parser.parse_args()

    output_path = args.output or args.input.replace(".npz", "_retargeted.npz")

    if args.format == "spider":
        results = retarget_from_spider_npz(args.input, xml_path=args.xml, side=args.side)
    else:
        data = np.load(args.input)
        retargeter = InspireHandRetargeter(xml_path=args.xml)
        T = data["hand_pose"].shape[0]
        results_list = []
        for t in range(T):
            r = retargeter.retarget_from_smplx(
                hand_pose_45=data["hand_pose"][t],
                global_orient=data["global_orient"][t],
                transl=data["transl"][t],
                betas=data.get("betas", None),
            )
            results_list.append(r)
        # Stack results
        results = {
            k: np.stack([r[k] for r in results_list])
            for k in results_list[0]
        }

    # Also save BODex-compatible formats
    results["bodex_seed_13"] = np.array([
        to_bodex_seed({"seed_config_13": results["seed_config_13"][t]})
        for t in range(results["seed_config_13"].shape[0])
    ])
    results["bodex_12dof"] = np.array([
        to_bodex_12dof({"finger_q_12": results["finger_q_12"][t]})
        for t in range(results["finger_q_12"].shape[0])
    ])
    results["urdf_12dof"] = np.array([
        to_urdf_12dof({"finger_q_12": results["finger_q_12"][t]})
        for t in range(results["finger_q_12"].shape[0])
    ])

    np.savez(output_path, **results)
    print(f"Saved retargeted data to {output_path}")
    print(f"  finger_q_12: {results['finger_q_12'].shape}")
    print(f"  seed_config_13: {results['seed_config_13'].shape}")
    print(f"  mean IK cost: {results['ik_cost'].mean():.6f}")


if __name__ == "__main__":
    main()
