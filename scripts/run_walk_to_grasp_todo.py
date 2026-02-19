#!/usr/bin/env python3
"""Run TODO walk-to-grasp pipeline end-to-end with explicit 8-step reporting."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
import os
import subprocess
import sys
import time
from typing import Any

import numpy as np

# Ensure `isaac_replay` is importable even when script is launched outside pack root.
_SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
_PACK_ROOT_HINT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
if _PACK_ROOT_HINT not in sys.path:
    sys.path.insert(0, _PACK_ROOT_HINT)

from isaac_replay.g1_curobo_planner import PlannerRequest, plan_walk_to_grasp


@dataclass
class StepRecord:
    index: int
    name: str
    status: str
    duration_sec: float
    command: list[str] | None
    detail: dict[str, Any]


def _parse_csv_floats(text: str, expected_len: int, name: str) -> np.ndarray:
    values = [float(x.strip()) for x in text.split(",")]
    if len(values) != expected_len:
        raise ValueError(f"{name} expects {expected_len} values, got {len(values)}: {text}")
    return np.asarray(values, dtype=np.float64)


def _run_cmd(cmd: list[str], cwd: str | None = None, env: dict[str, str] | None = None) -> tuple[int, str]:
    proc = subprocess.run(
        cmd,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    return int(proc.returncode), proc.stdout


def _scene_defaults(scene: str) -> tuple[str, str, str]:
    # scene_preset, base_start_pos_w, target_offset_obj_w
    if scene == "kitchen_pick_and_place":
        return "kitchen_pick_and_place", "-0.55,0.0,0.0", "-0.35,0.0,0.0"
    if scene == "galileo_g1_locomanip_pick_and_place":
        return "galileo_locomanip", "-0.80,0.18,0.0", "-0.45,0.0,0.0"
    return "none", "0.0,0.0,0.0", "-0.35,0.0,0.0"


def _float_list(v: np.ndarray) -> list[float]:
    return [float(x) for x in v.reshape(-1).tolist()]


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pack-root",
        default=None,
        help="Path to Humanoid-gen-pack. Default resolves from this script path.",
    )
    parser.add_argument("--out-dir", required=True, help="Output directory for this TODO run.")
    parser.add_argument("--scene", default="kitchen_pick_and_place")
    parser.add_argument("--object", default="cracker_box")
    parser.add_argument("--pattern", default="lift_place", choices=["line", "circle", "lift_place"])
    parser.add_argument(
        "--traj-start-pos-w",
        default=None,
        help="Optional object trajectory start xyz in world frame (overrides scene preset).",
    )
    parser.add_argument(
        "--traj-end-pos-w",
        default=None,
        help="Optional object trajectory end xyz in world frame (overrides scene preset).",
    )
    parser.add_argument(
        "--traj-lift-height",
        type=float,
        default=None,
        help="Optional lift height used by lift_place pattern.",
    )

    parser.add_argument("--planner", choices=["auto", "curobo", "open_loop"], default="auto")
    parser.add_argument("--strict-curobo", action="store_true")

    parser.add_argument("--base-pos-w", default=None, help="Start base position xyz for walk phase.")
    parser.add_argument("--base-yaw-deg", type=float, default=0.0)
    parser.add_argument("--walk-target-base-pos-w", default=None)
    parser.add_argument("--walk-target-offset-obj-w", default=None)
    parser.add_argument("--walk-target-offset-frame", choices=["object", "world"], default="object")
    parser.add_argument("--walk-target-yaw-mode", choices=["face_object", "fixed", "base_yaw"], default="face_object")
    parser.add_argument("--walk-target-yaw-deg", type=float, default=0.0)
    parser.add_argument("--walk-nav-max-lin-speed", type=float, default=0.22)
    parser.add_argument("--walk-nav-max-ang-speed", type=float, default=0.55)
    parser.add_argument("--walk-nav-dt", type=float, default=0.02)
    parser.add_argument("--walk-pregrasp-hold-steps", type=int, default=25)
    parser.add_argument(
        "--momagen-style",
        action="store_true",
        help="Use MoMaGen-style sampled start/target base pose with reachability-biased ranking.",
    )
    parser.add_argument("--momagen-start-dist-min", type=float, default=0.4)
    parser.add_argument("--momagen-start-dist-max", type=float, default=1.0)
    parser.add_argument("--momagen-target-dist-min", type=float, default=0.4)
    parser.add_argument("--momagen-target-dist-max", type=float, default=1.0)
    parser.add_argument("--momagen-min-travel-dist", type=float, default=0.25)
    parser.add_argument("--momagen-max-travel-dist", type=float, default=1.10)
    parser.add_argument("--momagen-sample-attempts", type=int, default=40)
    parser.add_argument("--momagen-sample-seed", type=int, default=13)
    parser.add_argument(
        "--replay-base-height",
        type=float,
        default=None,
        help="Optional base height action command forwarded to build_arm_follow_replay.py.",
    )

    parser.add_argument("--right-wrist-pos-obj", default="-0.20,-0.03,0.10")
    parser.add_argument("--right-wrist-quat-obj-wxyz", default="0.70710678,0.0,-0.70710678,0.0")
    parser.add_argument("--right-wrist-quat-control", choices=["follow_object", "constant_pelvis"], default="constant_pelvis")
    parser.add_argument("--right-wrist-quat-pelvis-wxyz", default="1.0,0.0,0.0,0.0")

    parser.add_argument("--cedex-grasp-pt", default=None)
    parser.add_argument("--cedex-grasp-index", type=int, default=0)
    parser.add_argument("--cedex-wrist-pos-offset", default="0.0,0.0,0.0")
    parser.add_argument("--cedex-wrist-quat-offset-wxyz", default="1.0,0.0,0.0,0.0")

    parser.add_argument("--device", default=os.environ.get("DEVICE", "cuda:0"))
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--no-runner", action="store_true", help="Only build replay artifacts, skip Isaac runner.")
    parser.add_argument("--isaac-python", default=os.environ.get("ISAAC_PYTHON", None))
    return parser


def main() -> None:
    args = _make_parser().parse_args()
    pack_root = os.path.abspath(args.pack_root or os.path.join(os.path.dirname(__file__), ".."))
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    isaac_root = os.path.join(pack_root, "repos", "IsaacLab-Arena")
    python_exec = args.isaac_python or sys.executable

    scene_preset, default_base_pos, default_target_offset = _scene_defaults(args.scene)
    base_pos_w = args.base_pos_w or default_base_pos
    walk_target_offset_obj_w = args.walk_target_offset_obj_w or default_target_offset

    kin_traj_path = os.path.join(out_dir, "object_kinematic_traj.npz")
    debug_traj_json = os.path.join(out_dir, "debug_traj.json")
    replay_hdf5 = os.path.join(out_dir, "replay_actions_arm_follow.hdf5")
    debug_replay_json = os.path.join(out_dir, "debug_replay.json")
    report_json = os.path.join(out_dir, "todo_run_report.json")

    steps: list[StepRecord] = []

    def append_step(
        index: int,
        name: str,
        status: str,
        detail: dict[str, Any] | None = None,
        command: list[str] | None = None,
        duration_sec: float = 0.0,
    ) -> None:
        steps.append(
            StepRecord(
                index=index,
                name=name,
                status=status,
                duration_sec=float(duration_sec),
                command=command,
                detail=detail or {},
            )
        )

    def run_step(index: int, name: str, fn):
        t0 = time.time()
        try:
            detail = fn()
            append_step(
                index=index,
                name=name,
                status="ok",
                duration_sec=float(time.time() - t0),
                command=detail.get("command") if isinstance(detail, dict) else None,
                detail=detail if isinstance(detail, dict) else {"result": detail},
            )
            return detail
        except Exception as exc:
            detail = {"error": f"{type(exc).__name__}: {exc}"}
            append_step(
                index=index,
                name=name,
                status="error",
                duration_sec=float(time.time() - t0),
                command=None,
                detail=detail,
            )
            raise

    # Step 1: Build synthetic object trajectory.
    def _step1_generate_traj():
        cmd = [
            python_exec,
            os.path.join(pack_root, "isaac_replay", "generate_debug_object_traj.py"),
            "--output",
            kin_traj_path,
            "--output-debug-json",
            debug_traj_json,
            "--object-name",
            args.object,
            "--pattern",
            args.pattern,
            "--scene-preset",
            scene_preset,
        ]
        if args.traj_start_pos_w:
            cmd.extend(["--start-pos-w", args.traj_start_pos_w])
        if args.traj_end_pos_w:
            cmd.extend(["--end-pos-w", args.traj_end_pos_w])
        if args.traj_lift_height is not None:
            cmd.extend(["--lift-height", f"{args.traj_lift_height}"])
        code, out = _run_cmd(cmd)
        if code != 0:
            raise RuntimeError(out)
        return {"command": cmd, "stdout_tail": out[-1500:]}

    run_step(1, "generate_object_traj", _step1_generate_traj)

    # Step 2: Planner input parse.
    def _step2_load_planner_inputs():
        obj_data = np.load(kin_traj_path, allow_pickle=True)
        obj_pos0_local = np.asarray(obj_data["object_pos_w"][0], dtype=np.float64)
        obj_quat0_local = np.asarray(obj_data["object_quat_wxyz"][0], dtype=np.float64)
        return {
            "object_pos0_w": _float_list(obj_pos0_local),
            "object_quat0_wxyz": _float_list(obj_quat0_local),
            "num_frames": int(obj_data["object_pos_w"].shape[0]),
        }

    step2_detail = run_step(2, "load_planner_inputs", _step2_load_planner_inputs)
    obj_pos0 = np.asarray(step2_detail["object_pos0_w"], dtype=np.float64)
    obj_quat0 = np.asarray(step2_detail["object_quat0_wxyz"], dtype=np.float64)

    # Step 3: CuRobo/open-loop walk planning.
    def _step3_plan_walk():
        # Compute wrist pose in world frame for IK check
        wrist_pos_obj = _parse_csv_floats(args.right_wrist_pos_obj, 3, "right_wrist_pos_obj")
        wrist_quat_obj = _parse_csv_floats(args.right_wrist_quat_obj_wxyz, 4, "right_wrist_quat_obj_wxyz")
        from isaac_replay.g1_curobo_planner import _quat_to_rotmat_wxyz, _normalize_quat_wxyz
        obj_rot = _quat_to_rotmat_wxyz(obj_quat0)
        wrist_pos_w = obj_pos0 + obj_rot @ wrist_pos_obj
        # Approximate wrist orientation in world
        wrist_quat_w = _normalize_quat_wxyz(wrist_quat_obj)

        req = PlannerRequest(
            planner=args.planner,
            strict_curobo=bool(args.strict_curobo),
            scene=args.scene,
            start_base_pos_w=tuple(_float_list(_parse_csv_floats(base_pos_w, 3, "base_pos_w"))),  # type: ignore[arg-type]
            start_base_yaw_rad=math.radians(float(args.base_yaw_deg)),
            object_pos_w=tuple(_float_list(obj_pos0)),  # type: ignore[arg-type]
            object_quat_wxyz=tuple(_float_list(obj_quat0)),  # type: ignore[arg-type]
            target_base_pos_w=(
                tuple(_float_list(_parse_csv_floats(args.walk_target_base_pos_w, 3, "walk_target_base_pos_w")))  # type: ignore[arg-type]
                if args.walk_target_base_pos_w
                else None
            ),
            target_offset_obj_w=tuple(_float_list(_parse_csv_floats(walk_target_offset_obj_w, 3, "walk_target_offset_obj_w"))),  # type: ignore[arg-type]
            target_offset_frame=args.walk_target_offset_frame,
            target_yaw_mode=args.walk_target_yaw_mode,
            target_yaw_deg=float(args.walk_target_yaw_deg),
            wrist_pos_w=tuple(_float_list(wrist_pos_w)),  # type: ignore[arg-type]
            wrist_quat_wxyz_ik=tuple(_float_list(wrist_quat_w)),  # type: ignore[arg-type]
            sample_start_base_pose=bool(args.momagen_style),
            start_sample_dist_min=float(args.momagen_start_dist_min),
            start_sample_dist_max=float(args.momagen_start_dist_max),
            start_sample_attempts=int(args.momagen_sample_attempts),
            start_sample_seed=int(args.momagen_sample_seed),
            sample_target_base_pose=bool(args.momagen_style),
            target_sample_dist_min=float(args.momagen_target_dist_min),
            target_sample_dist_max=float(args.momagen_target_dist_max),
            target_sample_attempts=int(args.momagen_sample_attempts),
            target_sample_seed=int(args.momagen_sample_seed) + 17,
            target_min_travel_dist=float(args.momagen_min_travel_dist),
            target_max_travel_dist=float(args.momagen_max_travel_dist),
        )
        result = plan_walk_to_grasp(req)
        detail = {"planner_result": result.to_dict()}
        if result.ik_result is not None:
            detail["ik_reachable"] = result.ik_result.reachable
            if not result.ik_result.reachable:
                detail["ik_warning"] = (
                    f"Wrist pose unreachable: {result.ik_result.reason}. "
                    "Consider adjusting target offset or wrist pose."
                )

        # Save MotionGen trajectory as npz for build_arm_follow_replay
        detail["_motion_gen_result"] = result.motion_gen_result
        if result.motion_gen_result is not None and result.motion_gen_result.success:
            detail["motion_gen_success"] = True
            detail["motion_gen_steps"] = result.motion_gen_result.num_steps
        else:
            detail["motion_gen_success"] = False
        return detail

    planner_detail = run_step(3, "plan_walk_to_grasp", _step3_plan_walk)
    planner_result = planner_detail["planner_result"]
    start_base_pos_w = planner_result.get(
        "start_base_pos_w",
        _float_list(_parse_csv_floats(base_pos_w, 3, "base_pos_w")),
    )
    start_base_pos_w_csv = f"{start_base_pos_w[0]},{start_base_pos_w[1]},{start_base_pos_w[2]}"
    start_base_yaw_rad = float(
        planner_result.get("start_base_yaw_rad", math.radians(float(args.base_yaw_deg)))
    )
    start_base_yaw_deg = float(math.degrees(start_base_yaw_rad))
    target_base_pos_w = planner_result["target_base_pos_w"]
    target_base_yaw_rad = float(planner_result["target_base_yaw_rad"])
    target_base_yaw_deg = float(math.degrees(target_base_yaw_rad))

    # Save MotionGen EE trajectory as npz (already in torso/pelvis frame from CuRobo FK)
    pregrasp_traj_npz = os.path.join(out_dir, "pregrasp_traj.npz")
    mg_result = planner_detail.get("_motion_gen_result")
    if mg_result is not None and mg_result.success and mg_result.ee_pos_trajectory:
        np.savez(
            pregrasp_traj_npz,
            ee_pos=np.array(mg_result.ee_pos_trajectory, dtype=np.float64),
            ee_quat_wxyz=np.array(mg_result.ee_quat_trajectory, dtype=np.float64),
        )
    else:
        pregrasp_traj_npz = None  # No trajectory to pass

    # Step 4: Build replay actions with walk-to-grasp + arm-follow.
    # Convert planner subgoals to JSON for build_arm_follow_replay.py
    nav_subgoals = planner_result.get("navigation_subgoals", [])
    subgoals_for_replay = [
        {"xy": [sg[0][0], sg[0][1]], "yaw": sg[0][2]}
        for sg in nav_subgoals
    ]

    def _step4_build_replay():
        cmd = [
            python_exec,
            os.path.join(pack_root, "isaac_replay", "build_arm_follow_replay.py"),
            "--kin-traj-path",
            kin_traj_path,
            "--output-hdf5",
            replay_hdf5,
            "--output-debug-json",
            debug_replay_json,
            f"--base-pos-w={start_base_pos_w_csv}",
            "--base-yaw",
            str(start_base_yaw_rad),
            f"--right-wrist-pos-obj={args.right_wrist_pos_obj}",
            f"--right-wrist-quat-obj-wxyz={args.right_wrist_quat_obj_wxyz}",
            "--right-wrist-quat-control",
            args.right_wrist_quat_control,
            f"--right-wrist-quat-pelvis-wxyz={args.right_wrist_quat_pelvis_wxyz}",
            "--left-hand-state",
            "0.0",
            "--right-hand-state",
            "0.0",
            "--walk-to-grasp",
            f"--walk-target-base-pos-w={target_base_pos_w[0]},{target_base_pos_w[1]},{target_base_pos_w[2]}",
            "--walk-target-yaw-mode",
            "fixed",
            "--walk-target-yaw-deg",
            f"{target_base_yaw_deg}",
            "--walk-nav-max-lin-speed",
            f"{args.walk_nav_max_lin_speed}",
            "--walk-nav-max-ang-speed",
            f"{args.walk_nav_max_ang_speed}",
            "--walk-nav-dt",
            f"{args.walk_nav_dt}",
            "--walk-pregrasp-hold-steps",
            f"{args.walk_pregrasp_hold_steps}",
        ]
        if args.replay_base_height is not None:
            cmd.extend(["--base-height", f"{args.replay_base_height}"])
        if len(subgoals_for_replay) > 1:
            cmd.extend(["--walk-nav-subgoals-json", json.dumps(subgoals_for_replay)])
        if pregrasp_traj_npz is not None:
            cmd.extend(["--pregrasp-traj-npz", pregrasp_traj_npz])
        if args.cedex_grasp_pt:
            cmd.extend(
                [
                    "--cedex-grasp-pt",
                    args.cedex_grasp_pt,
                    "--cedex-grasp-index",
                    str(args.cedex_grasp_index),
                    f"--cedex-wrist-pos-offset={args.cedex_wrist_pos_offset}",
                    f"--cedex-wrist-quat-offset-wxyz={args.cedex_wrist_quat_offset_wxyz}",
                ]
            )

        code, out = _run_cmd(cmd)
        if code != 0:
            raise RuntimeError(out)
        return {"command": cmd, "stdout_tail": out[-1500:]}

    run_step(4, "build_replay_actions", _step4_build_replay)

    # Step 5: Read replay debug metadata.
    def _step5_read_replay_metadata():
        with open(debug_replay_json, "r", encoding="utf-8") as f:
            replay_dbg_local = json.load(f)
        kin_start_step_local = int(replay_dbg_local.get("replay", {}).get("recommended_kin_start_step", 0))
        total_steps_local = int(replay_dbg_local.get("replay", {}).get("total_steps", 0))
        return {
            "replay_dbg": replay_dbg_local,
            "recommended_kin_start_step": kin_start_step_local,
            "total_steps": total_steps_local,
        }

    step5_detail = run_step(5, "read_replay_metadata", _step5_read_replay_metadata)
    replay_dbg = step5_detail["replay_dbg"]
    kin_start_step = int(step5_detail["recommended_kin_start_step"])
    total_steps = int(step5_detail["total_steps"])

    # Step 6: Run policy runner.
    if not args.no_runner:
        def _step6_run_runner():
            max_steps = int(args.max_steps if args.max_steps is not None else total_steps)
            cmd = [
                python_exec,
                os.path.join(pack_root, "isaac_replay", "policy_runner_kinematic_object_replay.py"),
                "--device",
                args.device,
                "--enable_cameras",
                "--policy_type",
                "replay",
                "--replay_file_path",
                replay_hdf5,
                "--episode_name",
                "demo_0",
                "--kin-traj-path",
                kin_traj_path,
                "--kin-asset-name",
                args.object,
                "--kin-start-step",
                str(kin_start_step),
                "--kin-apply-timing",
                "pre_step",
                "--max-steps",
                str(max_steps),
                args.scene,
                "--object",
                args.object,
                "--embodiment",
                "g1_wbc_pink",
            ]
            if args.scene == "kitchen_pick_and_place":
                cmd.extend([f"--g1-init-pos-w={start_base_pos_w_csv}", "--g1-init-yaw-deg", str(start_base_yaw_deg)])
            if args.headless:
                cmd.insert(2, "--headless")
            code, out = _run_cmd(cmd, cwd=isaac_root, env=os.environ.copy())
            if code != 0:
                raise RuntimeError(out[-4000:])
            return {"command": cmd, "stdout_tail": out[-1500:], "max_steps": max_steps}

        run_step(6, "run_isaac_policy_runner", _step6_run_runner)
    else:
        append_step(
            index=6,
            name="run_isaac_policy_runner",
            status="skipped",
            detail={
                "reason": "--no-runner enabled",
                "planned_max_steps": int(args.max_steps if args.max_steps is not None else total_steps),
            },
        )

    # Step 7: Artifact validation.
    def _step7_validate_artifacts():
        required_files = [kin_traj_path, debug_traj_json, replay_hdf5, debug_replay_json]
        checks = []
        for path in required_files:
            exists = os.path.isfile(path)
            size = int(os.path.getsize(path)) if exists else 0
            checks.append({"path": path, "exists": exists, "size_bytes": size})
        missing = [c["path"] for c in checks if not c["exists"]]
        if missing:
            raise FileNotFoundError(f"Missing required artifacts: {missing}")
        return {"artifacts": checks}

    run_step(7, "validate_artifacts", _step7_validate_artifacts)

    # Step 8: emit report.
    def _step8_prepare_report():
        return {"report_json": report_json}

    run_step(8, "write_report", _step8_prepare_report)

    report_status = "ok" if all(s.status in {"ok", "skipped"} for s in steps) else "error"
    report = {
        "pipeline": "todo_walk_to_grasp",
        "scene": args.scene,
        "object": args.object,
        "out_dir": out_dir,
        "planner": planner_result,
        "replay_debug": replay_dbg,
        "steps": [asdict(s) for s in steps],
        "status": report_status,
    }
    with open(report_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"[todo_pipeline] report: {report_json}")
    print(f"[todo_pipeline] replay: {replay_hdf5}")
    print(f"[todo_pipeline] debug: {debug_replay_json}")
    print(f"[todo_pipeline] status: {report['status']}")


if __name__ == "__main__":
    main()
