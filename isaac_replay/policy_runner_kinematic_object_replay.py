#!/usr/bin/env python3
"""Policy runner with kinematic object replay from a trajectory file."""

from __future__ import annotations

import json
import numpy as np
import os
import random
import re
import sys
import torch
import tqdm

# Ensure we import the IsaacLab-Arena checkout that ships with Humanoid-gen-pack,
# instead of an unrelated editable install elsewhere in the workspace.
_SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
_PACK_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
_LOCAL_ARENA_ROOT = os.path.join(_PACK_ROOT, "repos", "IsaacLab-Arena")
if os.path.isdir(os.path.join(_LOCAL_ARENA_ROOT, "isaaclab_arena")) and _LOCAL_ARENA_ROOT not in sys.path:
    sys.path.insert(0, _LOCAL_ARENA_ROOT)

from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
from isaaclab_arena.examples.example_environments.cli import (
    get_arena_builder_from_cli,
    get_isaaclab_arena_example_environment_cli_parser,
)
from isaaclab_arena.examples.policy_runner_cli import (
    add_gr00t_closedloop_arguments,
    add_replay_arguments,
    add_replay_lerobot_arguments,
    add_zero_action_arguments,
    create_policy,
)
from isaaclab_arena.utils.isaaclab_utils.simulation_app import SimulationAppContext


def _setup_policy_argument_parser_without_parse(args_parser):
    args_parser = get_isaaclab_arena_example_environment_cli_parser(args_parser)
    args_parser.add_argument(
        "--policy_type",
        type=str,
        choices=["zero_action", "replay", "replay_lerobot", "gr00t_closedloop"],
        default="zero_action",
        help="Type of policy to use. Ignored when --object-only is set.",
    )
    add_zero_action_arguments(args_parser)
    add_replay_arguments(args_parser)
    add_replay_lerobot_arguments(args_parser)
    add_gr00t_closedloop_arguments(args_parser)
    return args_parser


def _add_object_replay_args(parser):
    parser.add_argument(
        "--kin-traj-path",
        type=str,
        required=True,
        help="Path to object kinematic trajectory npz from tools/hoi_bodex_g1_bridge/build_replay.py",
    )
    parser.add_argument(
        "--kin-asset-name",
        type=str,
        default="pick_up_object",
        help="Asset name in env.scene to be overwritten by kinematic replay.",
    )
    parser.add_argument(
        "--kin-start-step",
        type=int,
        default=0,
        help="Simulation step offset to start applying object trajectory.",
    )
    parser.add_argument(
        "--kin-apply-timing",
        type=str,
        choices=["pre_step", "post_step"],
        default="pre_step",
        help="Apply object kinematic pose before or after env.step(action).",
    )
    parser.add_argument(
        "--kin-no-hold-last-pose",
        action="store_true",
        help="If set, do not keep writing last object pose after trajectory end.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Optional cap on simulation steps. By default, replay policy length is used.",
    )
    parser.add_argument(
        "--object-only",
        action="store_true",
        help="Only replay object motion. Robot sends zero actions and stands still.",
    )
    parser.add_argument(
        "--abort-on-reset",
        action="store_true",
        help="Abort immediately if any env terminates/truncates during replay.",
    )
    parser.add_argument(
        "--ignore-task-terminations",
        action="store_true",
        help="Disable non-timeout termination terms (e.g. success/object_dropped) during replay.",
    )
    parser.add_argument(
        "--use-hoi-object",
        action="store_true",
        help="Use HOIFHLI object mesh corresponding to trajectory object_name instead of pre-registered assets.",
    )
    parser.add_argument(
        "--hoi-root",
        type=str,
        default="/home/ubuntu/DATA2/workspace/xmh/hoifhli_release",
        help="Root path of hoifhli_release.",
    )
    parser.add_argument(
        "--hoi-runtime-asset-name",
        type=str,
        default=None,
        help="Optional runtime asset name. Default uses object_name from trajectory.",
    )
    parser.add_argument(
        "--hoi-object-scale",
        type=str,
        default="1.0,1.0,1.0",
        help="Runtime HOI object scale as sx,sy,sz.",
    )
    parser.add_argument(
        "--hoi-usd-cache-dir",
        type=str,
        default="/tmp/hoi_runtime_usd",
        help="Directory used to cache converted runtime USD meshes.",
    )
    parser.add_argument(
        "--debug-dump-dir",
        type=str,
        default=None,
        help="Optional directory to dump step-by-step debug traces (JSONL + metadata).",
    )
    parser.add_argument(
        "--debug-dump-every",
        type=int,
        default=1,
        help="Record one debug sample every N simulation steps.",
    )
    parser.add_argument(
        "--debug-dump-max-steps",
        type=int,
        default=2000,
        help="Maximum number of debug samples to write.",
    )


def _add_video_args(parser):
    parser.add_argument(
        "--save-video",
        action="store_true",
        help="Save first-person replay video from camera observations.",
    )
    parser.add_argument(
        "--video-output-dir",
        type=str,
        default="/home/ubuntu/DATA2/workspace/xmh/IsaacLab-Arena/.workflow_data/videos",
        help="Directory to save output videos.",
    )
    parser.add_argument(
        "--video-prefix",
        type=str,
        default="g1_kin_replay",
        help="Output filename prefix for saved videos.",
    )
    parser.add_argument(
        "--video-fps",
        type=int,
        default=30,
        help="Output video FPS.",
    )
    parser.add_argument(
        "--save-third-person",
        action="store_true",
        help="Also save a third-person and combined video (slower).",
    )


def _to_uint8_rgb(frame: np.ndarray) -> np.ndarray:
    arr = np.asarray(frame)
    if arr.ndim != 3:
        raise ValueError(f"Expected HxWxC frame, got shape: {arr.shape}")
    if arr.shape[2] == 4:
        arr = arr[:, :, :3]
    if np.issubdtype(arr.dtype, np.floating):
        max_val = float(np.nanmax(arr)) if arr.size > 0 else 1.0
        if max_val <= 1.5:
            arr = arr * 255.0
        arr = np.clip(arr, 0.0, 255.0).astype(np.uint8)
    else:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def _save_frames_to_video(frames, video_path, fps=30):
    import av

    if not frames:
        return
    container = av.open(video_path, mode="w")
    h, w = frames[0].shape[:2]
    stream = container.add_stream("h264", rate=fps)
    stream.width = w
    stream.height = h
    stream.pix_fmt = "yuv420p"
    stream.options = {"crf": "18", "profile:v": "high"}
    for frame_data in frames:
        av_frame = av.VideoFrame.from_ndarray(frame_data, format="rgb24")
        for packet in stream.encode(av_frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()
    print(f"Video saved to: {video_path} ({len(frames)} frames)")


def _create_third_person_camera(width=640, height=480, follow_offset=(2.0, -1.2, 1.4)):
    import omni.replicator.core as rep
    import omni.usd
    from pxr import Gf, UsdGeom

    stage = omni.usd.get_context().get_stage()
    cam_path = "/World/ThirdPersonCam"

    cam_prim = stage.DefinePrim(cam_path, "Camera")
    cam = UsdGeom.Camera(cam_prim)
    cam.GetFocalLengthAttr().Set(24.0)
    cam.GetHorizontalApertureAttr().Set(20.955)
    cam.GetClippingRangeAttr().Set(Gf.Vec2f(0.1, 20.0))

    xform = UsdGeom.Xformable(cam_prim)
    xform.ClearXformOpOrder()
    translate_op = xform.AddTranslateOp()
    rotate_op = xform.AddRotateXYZOp()
    translate_op.Set(Gf.Vec3d(*follow_offset))
    rotate_op.Set(Gf.Vec3f(-30.0, 0.0, 140.0))

    rp = rep.create.render_product(cam_path, (width, height))
    rgb_annot = rep.AnnotatorRegistry.get_annotator("rgb")
    rgb_annot.attach([rp])

    return rgb_annot, translate_op, rotate_op, np.array(follow_offset, dtype=np.float32)


def _extract_robot_pos(obs, env):
    policy_obs = obs.get("policy") if isinstance(obs, dict) else None
    if isinstance(policy_obs, dict):
        robot_pos = policy_obs.get("robot_pos")
        if robot_pos is not None:
            return robot_pos[0].detach().cpu().numpy()
    try:
        return env.scene["robot"].data.root_pos_w[0].detach().cpu().numpy()
    except Exception:
        return None


def _update_third_person_camera(translate_op, rotate_op, robot_pos, follow_offset):
    from pxr import Gf

    target = np.asarray(robot_pos, dtype=np.float32)[:3]
    cam_pos = target + follow_offset
    look = target - cam_pos
    horiz = float(np.linalg.norm(look[:2])) + 1e-6
    yaw_deg = float(np.degrees(np.arctan2(look[1], look[0])))
    pitch_deg = float(-np.degrees(np.arctan2(abs(look[2]), horiz)))
    translate_op.Set(Gf.Vec3d(float(cam_pos[0]), float(cam_pos[1]), float(cam_pos[2])))
    rotate_op.Set(Gf.Vec3f(pitch_deg, 0.0, yaw_deg))


def _read_third_person_frame(rgb_annot):
    data = rgb_annot.get_data()
    if data is None or len(data) == 0:
        return None
    return _to_uint8_rgb(np.array(data))


def _read_object_name_from_traj_npz(kin_traj_path: str) -> str:
    data = np.load(kin_traj_path, allow_pickle=True)
    if "object_name" not in data:
        raise KeyError(f"{kin_traj_path} does not contain key `object_name`.")
    raw_name = data["object_name"]
    if isinstance(raw_name, np.ndarray):
        if raw_name.shape == ():
            raw_name = raw_name.item()
        elif raw_name.size > 0:
            raw_name = raw_name.reshape(-1)[0]
    name = str(raw_name).strip()
    if not name:
        raise ValueError(f"Empty object_name in {kin_traj_path}.")
    return name


def _parse_scale_csv(text: str) -> tuple[float, float, float]:
    values = [float(x.strip()) for x in text.split(",")]
    if len(values) != 3:
        raise ValueError(f"--hoi-object-scale must have 3 values, got: {text}")
    return float(values[0]), float(values[1]), float(values[2])


def _sanitize_asset_name(name: str) -> str:
    safe = re.sub(r"[^0-9a-zA-Z_]", "_", name)
    if not safe:
        safe = "hoi_object"
    if safe[0].isdigit():
        safe = f"obj_{safe}"
    return safe


def _resolve_hoi_mesh_path(hoi_root: str, object_name: str, cache_dir: str) -> str:
    candidates = [
        os.path.join(hoi_root, "grasp_generation", "objects", f"{object_name}.obj"),
        os.path.join(hoi_root, "data", "processed_data", "captured_objects", f"{object_name}.obj"),
        os.path.join(hoi_root, "data", "processed_data", "rest_object_geo", f"{object_name}.obj"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path

    rest_ply = os.path.join(hoi_root, "data", "processed_data", "rest_object_geo", f"{object_name}.ply")
    if os.path.exists(rest_ply):
        import trimesh

        os.makedirs(cache_dir, exist_ok=True)
        out_obj = os.path.join(cache_dir, f"{object_name}.obj")
        mesh = trimesh.load(rest_ply, force="mesh")
        mesh.export(out_obj)
        return out_obj

    raise FileNotFoundError(
        f"Cannot find mesh for HOI object `{object_name}` under {hoi_root}. "
        "Expected .obj in grasp_generation/objects or processed_data/captured_objects, "
        "or .ply in processed_data/rest_object_geo."
    )


def _convert_mesh_to_runtime_usd(
    mesh_path: str,
    usd_cache_dir: str,
    asset_name: str,
    scale: tuple[float, float, float],
) -> str:
    from isaaclab.sim.converters import MeshConverter, MeshConverterCfg
    from isaaclab.sim.schemas import schemas_cfg

    os.makedirs(usd_cache_dir, exist_ok=True)
    cfg = MeshConverterCfg(
        asset_path=os.path.abspath(mesh_path),
        force_usd_conversion=True,
        usd_dir=os.path.abspath(usd_cache_dir),
        usd_file_name=f"{asset_name}.usd",
        make_instanceable=True,
        collision_props=schemas_cfg.CollisionPropertiesCfg(collision_enabled=True),
        mesh_collision_props=schemas_cfg.ConvexHullPropertiesCfg(),
        rigid_props=schemas_cfg.RigidBodyPropertiesCfg(),
        mass_props=schemas_cfg.MassPropertiesCfg(mass=0.5),
        scale=scale,
    )
    converter = MeshConverter(cfg)
    return converter.usd_path


def _register_runtime_object_asset(asset_name: str, usd_path: str, scale: tuple[float, float, float]) -> None:
    from isaaclab_arena.assets.asset_registry import AssetRegistry
    from isaaclab_arena.assets.object import Object

    registry = AssetRegistry()
    if registry.is_registered(asset_name):
        return

    def _init(self, prim_path=None, initial_pose=None):
        Object.__init__(
            self,
            name=asset_name,
            prim_path=prim_path,
            tags=["object"],
            usd_path=usd_path,
            scale=scale,
            initial_pose=initial_pose,
        )

    runtime_cls = type(
        f"RuntimeHoiObject_{asset_name}",
        (Object,),
        {
            "name": asset_name,
            "tags": ["object"],
            "usd_path": usd_path,
            "scale": scale,
            "__init__": _init,
        },
    )
    registry.register(runtime_cls)


def _prepare_runtime_hoi_object_asset(
    kin_traj_path: str,
    hoi_root: str,
    runtime_asset_name: str | None,
    object_scale: tuple[float, float, float],
    usd_cache_dir: str,
) -> tuple[str, str, str]:
    traj_object_name = _read_object_name_from_traj_npz(kin_traj_path)
    asset_name = _sanitize_asset_name(runtime_asset_name or traj_object_name)
    mesh_cache_dir = os.path.join(usd_cache_dir, "mesh_cache")
    mesh_path = _resolve_hoi_mesh_path(hoi_root, traj_object_name, mesh_cache_dir)
    usd_path = _convert_mesh_to_runtime_usd(mesh_path, usd_cache_dir, asset_name, object_scale)
    _register_runtime_object_asset(asset_name, usd_path, object_scale)
    return traj_object_name, asset_name, usd_path


class ObjectKinematicReplayer:
    def __init__(
        self,
        object_traj_path: str,
        object_asset_name: str,
        device: torch.device,
        hold_last_pose: bool,
        start_step: int,
    ):
        data = np.load(object_traj_path, allow_pickle=True)
        if "object_pos_w" not in data or "object_quat_wxyz" not in data:
            raise KeyError(
                f"{object_traj_path} must contain object_pos_w and object_quat_wxyz. "
                f"Found keys: {list(data.keys())}"
            )
        self.object_pos_w = torch.as_tensor(data["object_pos_w"], dtype=torch.float32, device=device)
        self.object_quat_wxyz = torch.as_tensor(data["object_quat_wxyz"], dtype=torch.float32, device=device)
        if self.object_pos_w.ndim != 2 or self.object_pos_w.shape[1] != 3:
            raise ValueError(f"object_pos_w should be [T,3], got {tuple(self.object_pos_w.shape)}")
        if self.object_quat_wxyz.ndim != 2 or self.object_quat_wxyz.shape[1] != 4:
            raise ValueError(f"object_quat_wxyz should be [T,4], got {tuple(self.object_quat_wxyz.shape)}")
        if self.object_pos_w.shape[0] != self.object_quat_wxyz.shape[0]:
            raise ValueError(
                f"object_pos_w length {self.object_pos_w.shape[0]} != object_quat_wxyz length {self.object_quat_wxyz.shape[0]}"
            )

        self.object_asset_name = object_asset_name
        self.hold_last_pose = hold_last_pose
        self.start_step = max(0, int(start_step))
        self.length = int(self.object_pos_w.shape[0])
        self.zero_vel = torch.zeros((1, 6), dtype=torch.float32, device=device)

    def _step_to_index(self, step: int) -> int | None:
        if step < self.start_step:
            return None
        idx = step - self.start_step
        if idx >= self.length:
            if not self.hold_last_pose:
                return None
            idx = self.length - 1
        return idx

    def apply(self, env, step: int) -> None:
        idx = self._step_to_index(step)
        if idx is None:
            return
        if env.num_envs != 1:
            raise ValueError("Kinematic object replay currently supports num_envs == 1 only.")
        asset = env.scene[self.object_asset_name]
        pose_w = torch.cat(
            [self.object_pos_w[idx].unsqueeze(0), self.object_quat_wxyz[idx].unsqueeze(0)],
            dim=-1,
        )
        env_ids = torch.tensor([0], device=env.device, dtype=torch.long)
        asset.write_root_pose_to_sim(pose_w, env_ids=env_ids)
        asset.write_root_velocity_to_sim(self.zero_vel, env_ids=env_ids)


def _disable_non_timeout_terminations(env) -> list[str]:
    disabled_terms: list[str] = []
    if not hasattr(env, "termination_manager"):
        return disabled_terms

    tm = env.termination_manager
    for term_name in list(getattr(tm, "active_terms", [])):
        term_cfg = tm.get_term_cfg(term_name)
        if term_cfg.time_out or term_name in {"time_out", "timeout"}:
            continue

        def _always_false_termination(local_env, **kwargs):  # noqa: ANN001
            return torch.zeros(local_env.num_envs, device=local_env.device, dtype=torch.bool)

        term_cfg.func = _always_false_termination
        term_cfg.params = {}
        tm.set_term_cfg(term_name, term_cfg)
        disabled_terms.append(term_name)

    if disabled_terms:
        print(f"[termination] Disabled non-timeout terms for replay: {disabled_terms}")
    return disabled_terms


def _to_serializable(value):
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _to_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_serializable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


class _RuntimeDebugLogger:
    def __init__(self, args_cli, env):
        self.enabled = bool(args_cli.debug_dump_dir)
        self.sample_every = max(1, int(getattr(args_cli, "debug_dump_every", 1)))
        self.max_samples = max(1, int(getattr(args_cli, "debug_dump_max_steps", 2000)))
        self.num_samples = 0
        self._fp = None
        self._step_path = None
        self._meta_path = None

        if not self.enabled:
            return

        dump_dir = os.path.abspath(args_cli.debug_dump_dir)
        os.makedirs(dump_dir, exist_ok=True)
        self._step_path = os.path.join(dump_dir, "runner_debug_steps.jsonl")
        self._meta_path = os.path.join(dump_dir, "runner_debug_meta.json")
        self._fp = open(self._step_path, "w", encoding="utf-8")

        meta: dict[str, object] = {
            "args": _to_serializable(vars(args_cli)),
            "paths": {},
            "env": {},
            "action_term": {},
        }
        try:
            import isaaclab_arena

            meta["paths"]["isaaclab_arena"] = os.path.abspath(isaaclab_arena.__file__)
        except Exception as exc:  # pragma: no cover - best-effort debug
            meta["paths"]["isaaclab_arena_error"] = f"{type(exc).__name__}: {exc}"
        try:
            import isaaclab

            meta["paths"]["isaaclab"] = os.path.abspath(isaaclab.__file__)
        except Exception as exc:  # pragma: no cover - best-effort debug
            meta["paths"]["isaaclab_error"] = f"{type(exc).__name__}: {exc}"

        try:
            meta["env"]["action_terms"] = list(getattr(env.action_manager, "active_terms", []))
            meta["env"]["action_term_dims"] = list(getattr(env.action_manager, "action_term_dim", []))
        except Exception as exc:  # pragma: no cover - best-effort debug
            meta["env"]["action_manager_error"] = f"{type(exc).__name__}: {exc}"
        try:
            meta["env"]["robot_joint_names"] = list(env.scene["robot"].data.joint_names)
        except Exception as exc:  # pragma: no cover - best-effort debug
            meta["env"]["robot_joint_names_error"] = f"{type(exc).__name__}: {exc}"
        try:
            g1_term = env.action_manager.get_term("g1_action")
            meta["action_term"]["type"] = type(g1_term).__name__
            if hasattr(g1_term, "action_dim"):
                meta["action_term"]["action_dim"] = int(g1_term.action_dim)
            if hasattr(g1_term, "wbc_g1_joints_order"):
                wbc_joint_order = list(g1_term.wbc_g1_joints_order.keys())
                meta["action_term"]["wbc_joint_order"] = wbc_joint_order
                meta["action_term"]["wbc_joint_order_len"] = len(wbc_joint_order)
                try:
                    from isaaclab_arena_g1.g1_whole_body_controller.wbc_policy.policy.policy_constants import G1_NUM_JOINTS

                    meta["action_term"]["policy_constant_g1_num_joints"] = int(G1_NUM_JOINTS)
                    if int(G1_NUM_JOINTS) != len(wbc_joint_order):
                        meta["action_term"]["joint_count_warning"] = (
                            f"G1_NUM_JOINTS={int(G1_NUM_JOINTS)} but wbc_joint_order_len={len(wbc_joint_order)}"
                        )
                except Exception as exc:  # pragma: no cover - best-effort debug
                    meta["action_term"]["policy_constant_error"] = f"{type(exc).__name__}: {exc}"
        except Exception as exc:  # pragma: no cover - best-effort debug
            meta["action_term"]["error"] = f"{type(exc).__name__}: {exc}"

        with open(self._meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        print(f"[debug_dump] metadata: {self._meta_path}")
        print(f"[debug_dump] step trace: {self._step_path}")

    def record(self, *, step: int, episode_step: int, env, actions, terminated, truncated, kin_asset_name: str):
        if not self.enabled:
            return
        if self.num_samples >= self.max_samples:
            return
        if (step % self.sample_every) != 0:
            return

        entry: dict[str, object] = {
            "step": int(step),
            "episode_step": int(episode_step),
            "terminated": _to_serializable(terminated),
            "truncated": _to_serializable(truncated),
        }
        try:
            entry["cmd_actions"] = _to_serializable(actions[0])
        except Exception:
            pass

        try:
            term = env.action_manager.get_term("g1_action")
            if hasattr(term, "raw_actions"):
                entry["term_raw_actions"] = _to_serializable(term.raw_actions[0])
            if hasattr(term, "processed_actions"):
                entry["term_processed_actions"] = _to_serializable(term.processed_actions[0])
            if hasattr(term, "navigate_cmd"):
                entry["term_navigate_cmd"] = _to_serializable(term.navigate_cmd[0])
            if hasattr(term, "_wbc_goal"):
                entry["term_wbc_goal"] = _to_serializable(term._wbc_goal)
        except Exception as exc:  # pragma: no cover - best-effort debug
            entry["term_error"] = f"{type(exc).__name__}: {exc}"

        try:
            robot = env.scene["robot"].data
            entry["robot_root_pos_w"] = _to_serializable(robot.root_pos_w[0])
            entry["robot_root_link_pos_w"] = _to_serializable(robot.root_link_pos_w[0])
            entry["robot_root_link_lin_vel_b"] = _to_serializable(robot.root_link_lin_vel_b[0])
            entry["robot_root_link_ang_vel_b"] = _to_serializable(robot.root_link_ang_vel_b[0])
            entry["robot_joint_pos"] = _to_serializable(robot.joint_pos[0])
            entry["robot_joint_vel"] = _to_serializable(robot.joint_vel[0])
        except Exception as exc:  # pragma: no cover - best-effort debug
            entry["robot_state_error"] = f"{type(exc).__name__}: {exc}"

        try:
            obj = env.scene[kin_asset_name].data
            entry["object_root_pos_w"] = _to_serializable(obj.root_pos_w[0])
        except Exception as exc:  # pragma: no cover - best-effort debug
            entry["object_state_error"] = f"{type(exc).__name__}: {exc}"

        assert self._fp is not None
        self._fp.write(json.dumps(entry, ensure_ascii=True) + "\n")
        self._fp.flush()
        self.num_samples += 1

    def close(self):
        if self._fp is not None:
            self._fp.close()
            self._fp = None


def main():
    args_parser = get_isaaclab_arena_cli_parser()
    args_cli, _ = args_parser.parse_known_args()

    with SimulationAppContext(args_cli):
        args_parser = _setup_policy_argument_parser_without_parse(args_parser)
        _add_object_replay_args(args_parser)
        _add_video_args(args_parser)
        args_cli = args_parser.parse_args()
        if args_cli.use_hoi_object:
            object_scale = _parse_scale_csv(args_cli.hoi_object_scale)
            traj_object_name, runtime_asset_name, runtime_usd = _prepare_runtime_hoi_object_asset(
                kin_traj_path=args_cli.kin_traj_path,
                hoi_root=args_cli.hoi_root,
                runtime_asset_name=args_cli.hoi_runtime_asset_name,
                object_scale=object_scale,
                usd_cache_dir=args_cli.hoi_usd_cache_dir,
            )
            # Ensure env scene object and kinematic overwrite object are consistent.
            args_cli.object = runtime_asset_name
            args_cli.kin_asset_name = runtime_asset_name
            print(
                "[HOI object] trajectory object_name="
                f"{traj_object_name}, runtime asset={runtime_asset_name}, usd={runtime_usd}"
            )
        object_only = getattr(args_cli, "object_only", False)
        if not object_only:
            if args_cli.policy_type == "replay" and args_cli.replay_file_path is None:
                raise ValueError("--replay_file_path is required when using --policy_type replay")
            if args_cli.policy_type == "replay_lerobot" and args_cli.config_yaml_path is None:
                raise ValueError("--config_yaml_path is required when using --policy_type replay_lerobot")
            if args_cli.policy_type == "gr00t_closedloop" and args_cli.policy_config_yaml_path is None:
                raise ValueError("--policy_config_yaml_path is required when using --policy_type gr00t_closedloop")

        arena_builder = get_arena_builder_from_cli(args_cli)
        env = arena_builder.make_registered()

        if args_cli.seed is not None:
            env.seed(args_cli.seed)
            torch.manual_seed(args_cli.seed)
            np.random.seed(args_cli.seed)
            random.seed(args_cli.seed)

        obs, _ = env.reset()
        if args_cli.ignore_task_terminations:
            _disable_non_timeout_terminations(env)
        if object_only:
            policy = None
            policy_steps = 0
        else:
            policy, policy_steps = create_policy(args_cli)
        tp_camera = _create_third_person_camera() if (args_cli.save_video and args_cli.save_third_person) else None
        fp_frames = []
        tp_frames = []

        object_replayer = ObjectKinematicReplayer(
            object_traj_path=args_cli.kin_traj_path,
            object_asset_name=args_cli.kin_asset_name,
            device=env.device,
            hold_last_pose=not args_cli.kin_no_hold_last_pose,
            start_step=args_cli.kin_start_step,
        )
        debug_logger = _RuntimeDebugLogger(args_cli, env)

        if object_only:
            step_budget = object_replayer.length
            if args_cli.max_steps is not None:
                step_budget = min(step_budget, args_cli.max_steps)
        else:
            step_budget = policy_steps if args_cli.max_steps is None else min(policy_steps, args_cli.max_steps)

        # lazy import to avoid app startup stalls
        from isaaclab_arena.metrics.metrics import compute_metrics

        def _build_reset_debug(step: int, info: dict | None) -> dict:
            detail: dict[str, object] = {"global_step": int(step)}
            if info is not None:
                detail["info_keys"] = sorted(list(info.keys()))
            if hasattr(env, "termination_manager"):
                try:
                    tm = env.termination_manager
                    terms: dict[str, object] = {}
                    for name in getattr(tm, "active_terms", []):
                        try:
                            value = tm.get_term(name)
                            if torch.is_tensor(value):
                                terms[str(name)] = value.detach().cpu().tolist()
                            else:
                                terms[str(name)] = value
                        except Exception as exc:  # pragma: no cover - best-effort debug
                            terms[str(name)] = f"error:{type(exc).__name__}:{exc}"
                    detail["termination_terms"] = terms
                except Exception as exc:  # pragma: no cover - best-effort debug
                    detail["termination_terms_error"] = f"{type(exc).__name__}: {exc}"
            try:
                detail["robot_root_pos_w"] = env.scene["robot"].data.root_pos_w[0].detach().cpu().tolist()
            except Exception:
                pass
            try:
                detail["object_root_pos_w"] = env.scene[args_cli.kin_asset_name].data.root_pos_w[0].detach().cpu().tolist()
            except Exception:
                pass
            return detail

        episode_step = 0
        for step in tqdm.tqdm(range(step_budget)):
            with torch.inference_mode():
                if args_cli.kin_apply_timing == "pre_step":
                    object_replayer.apply(env, episode_step)

                if object_only:
                    action_dim = env.action_space.shape[-1] if hasattr(env, "action_space") else 23
                    actions = torch.zeros((env.num_envs, action_dim), device=env.device)
                else:
                    actions = policy.get_action(env, obs)
                    if actions is None:
                        print(f"Policy returned None at step {step}, stopping replay.")
                        break
                obs, _, terminated, truncated, info = env.step(actions)

                if args_cli.kin_apply_timing == "post_step":
                    object_replayer.apply(env, episode_step)

                debug_logger.record(
                    step=step,
                    episode_step=episode_step,
                    env=env,
                    actions=actions,
                    terminated=terminated,
                    truncated=truncated,
                    kin_asset_name=args_cli.kin_asset_name,
                )

                if args_cli.save_video:
                    if "camera_obs" in obs:
                        cam_keys = list(obs["camera_obs"].keys())
                        if cam_keys:
                            cam_img = obs["camera_obs"][cam_keys[0]]
                            fp_frames.append(_to_uint8_rgb(cam_img[0].cpu().numpy()))

                    if tp_camera is not None:
                        tp_annotator, tp_translate_op, tp_rotate_op, tp_follow_offset = tp_camera
                        robot_pos = _extract_robot_pos(obs, env)
                        if robot_pos is not None:
                            _update_third_person_camera(tp_translate_op, tp_rotate_op, robot_pos, tp_follow_offset)
                        tp_frame = _read_third_person_frame(tp_annotator)
                        if tp_frame is not None:
                            tp_frames.append(tp_frame)

                if terminated.any() or truncated.any():
                    reset_detail = _build_reset_debug(step=step, info=info if isinstance(info, dict) else None)
                    if args_cli.abort_on_reset:
                        term_ids = terminated.nonzero().flatten().tolist()
                        trunc_ids = truncated.nonzero().flatten().tolist()
                        raise RuntimeError(
                            "Environment reset detected during replay "
                            f"(step={step}, terminated={term_ids}, truncated={trunc_ids}). "
                            f"Debug={json.dumps(reset_detail, ensure_ascii=True)}"
                        )
                    print(
                        f"Resetting policy for terminated env_ids: {terminated.nonzero().flatten()}"
                        f" and truncated env_ids: {truncated.nonzero().flatten()}"
                    )
                    print(f"Reset debug: {json.dumps(reset_detail, ensure_ascii=True)}")
                    env_ids = (terminated | truncated).nonzero().flatten()
                    if policy is not None:
                        policy.reset(env_ids=env_ids)
                    episode_step = 0
                    continue
                episode_step += 1

        metrics = compute_metrics(env)
        print(f"Metrics: {metrics}")

        if args_cli.save_video:
            os.makedirs(args_cli.video_output_dir, exist_ok=True)
            fp_path = os.path.join(args_cli.video_output_dir, f"{args_cli.video_prefix}_first_person.mp4")
            _save_frames_to_video(fp_frames, fp_path, fps=args_cli.video_fps)

            if tp_frames:
                tp_path = os.path.join(args_cli.video_output_dir, f"{args_cli.video_prefix}_third_person.mp4")
                _save_frames_to_video(tp_frames, tp_path, fps=args_cli.video_fps)

            if fp_frames and tp_frames:
                from PIL import Image

                min_len = min(len(fp_frames), len(tp_frames))
                combined = []
                for i in range(min_len):
                    fp = fp_frames[i]
                    tp = tp_frames[i]
                    if fp.shape[0] != tp.shape[0]:
                        tp = np.array(Image.fromarray(tp).resize((fp.shape[1], fp.shape[0])))
                    combined.append(np.concatenate([fp, tp], axis=1))
                combined_path = os.path.join(args_cli.video_output_dir, f"{args_cli.video_prefix}_combined.mp4")
                _save_frames_to_video(combined, combined_path, fps=args_cli.video_fps)
        debug_logger.close()
        env.close()


if __name__ == "__main__":
    main()
