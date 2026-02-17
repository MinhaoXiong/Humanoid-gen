#!/usr/bin/env python3
"""Policy runner with kinematic object replay from a trajectory file."""

from __future__ import annotations

import numpy as np
import os
import random
import torch
import tqdm

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
        required=True,
        help="Type of policy to use.",
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


def main():
    args_parser = get_isaaclab_arena_cli_parser()
    args_cli, _ = args_parser.parse_known_args()

    with SimulationAppContext(args_cli):
        args_parser = _setup_policy_argument_parser_without_parse(args_parser)
        _add_object_replay_args(args_parser)
        _add_video_args(args_parser)
        args_cli = args_parser.parse_args()
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
        policy, policy_steps = create_policy(args_cli)
        tp_camera = _create_third_person_camera() if (args_cli.save_video and args_cli.save_third_person) else None
        fp_frames = []
        tp_frames = []

        step_budget = policy_steps if args_cli.max_steps is None else min(policy_steps, args_cli.max_steps)
        object_replayer = ObjectKinematicReplayer(
            object_traj_path=args_cli.kin_traj_path,
            object_asset_name=args_cli.kin_asset_name,
            device=env.device,
            hold_last_pose=not args_cli.kin_no_hold_last_pose,
            start_step=args_cli.kin_start_step,
        )

        # lazy import to avoid app startup stalls
        from isaaclab_arena.metrics.metrics import compute_metrics

        for step in tqdm.tqdm(range(step_budget)):
            with torch.inference_mode():
                if args_cli.kin_apply_timing == "pre_step":
                    object_replayer.apply(env, step)

                actions = policy.get_action(env, obs)
                if actions is None:
                    print(f"Policy returned None at step {step}, stopping replay.")
                    break
                obs, _, terminated, truncated, _ = env.step(actions)

                if args_cli.kin_apply_timing == "post_step":
                    object_replayer.apply(env, step)

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
                    print(
                        f"Resetting policy for terminated env_ids: {terminated.nonzero().flatten()}"
                        f" and truncated env_ids: {truncated.nonzero().flatten()}"
                    )
                    env_ids = (terminated | truncated).nonzero().flatten()
                    policy.reset(env_ids=env_ids)

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
        env.close()


if __name__ == "__main__":
    main()
