#!/usr/bin/env python3
"""Generate InspireHand grasps for a single object using CEDex-Grasp."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from typing import Any

import torch


DATASET_PATHS = {
    "contactdb": "human_contact/contactdb/cmap_dataset.pt",
    "ycb": "human_contact/ycb/cmap_dataset.pt",
}


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cedex-root",
        default="/home/ubuntu/DATA2/workspace/xmh/CEDex-Grasp",
        help="Path to CEDex-Grasp repository root.",
    )
    parser.add_argument("--dataset", choices=["contactdb", "ycb"], default="ycb")
    parser.add_argument(
        "--dataset-path",
        default=None,
        help="Optional direct path to cmap_dataset.pt. Overrides --dataset inferred path.",
    )
    parser.add_argument("--object-name", default="003_cracker_box", help="Exact obj_name in dataset.")
    parser.add_argument("--num-particles", type=int, default=64)
    parser.add_argument("--save-top-k", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--max-iter", type=int, default=300)
    parser.add_argument("--device", default=None, help="cuda/cpu. Default auto.")
    parser.add_argument(
        "--output-pt",
        default=None,
        help="Output .pt path. Default: Humanoid-gen-pack/artifacts/inspire_grasps/<name>_<timestamp>.pt",
    )
    parser.add_argument("--output-json", default=None, help="Optional output summary json path.")
    parser.add_argument("--list-objects", action="store_true", help="List object names in dataset and exit.")
    return parser


def _extract_top_k_grasps(
    record: tuple[torch.Tensor, torch.Tensor],
    robot_name: str,
    object_name: str,
    dataset_name: str,
    top_k: int,
) -> list[dict[str, Any]]:
    q_tra, energy = record
    final_q = q_tra.detach()
    final_energy = energy.detach()
    sorted_indices = torch.argsort(final_energy)
    top_k_indices = sorted_indices[:top_k]

    grasps: list[dict[str, Any]] = []
    for rank, idx in enumerate(top_k_indices):
        item = {
            "q": final_q[idx].cpu(),
            "energy": float(final_energy[idx].item()),
            "rank": int(rank),
            "object_name": f"{dataset_name}+{object_name}",
            "robot_name": robot_name,
        }
        grasps.append(item)
    return grasps


def main() -> None:
    args = _make_parser().parse_args()
    cedex_root = os.path.abspath(args.cedex_root)
    if not os.path.isdir(cedex_root):
        raise FileNotFoundError(f"CEDex root not found: {cedex_root}")

    if cedex_root not in sys.path:
        sys.path.insert(0, cedex_root)

    dataset_rel_path = DATASET_PATHS[args.dataset]
    if args.dataset_path:
        dataset_path = os.path.abspath(args.dataset_path)
    else:
        candidates = [
            os.path.join(cedex_root, dataset_rel_path),
            os.path.join(cedex_root, "data", dataset_rel_path),
        ]
        dataset_path = next((p for p in candidates if os.path.isfile(p)), "")
        if not dataset_path:
            raise FileNotFoundError(
                "Dataset file not found. Tried:\n- " + "\n- ".join(os.path.abspath(x) for x in candidates)
            )
    if not os.path.isfile(dataset_path):
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    dataset = torch.load(dataset_path)
    all_names = sorted({str(sample.get("obj_name", "")) for sample in dataset})
    if args.list_objects:
        print("\n".join(all_names))
        return

    selected = [sample for sample in dataset if str(sample.get("obj_name", "")) == args.object_name]
    if not selected:
        raise ValueError(
            f"Object `{args.object_name}` not found in dataset `{args.dataset}`. "
            f"Available count: {len(all_names)}"
        )

    device = args.device if args.device is not None else ("cuda" if torch.cuda.is_available() else "cpu")

    from utils_model.AdamGrasp import AdamGrasp

    model = AdamGrasp(
        robot_name="inspirehand",
        num_particles=args.num_particles,
        init_rand_scale=0.5,
        max_iter=args.max_iter,
        steps_per_iter=1,
        learning_rate=args.learning_rate,
        device=device,
    )

    top_k = max(1, int(args.save_top_k))
    all_grasps: list[dict[str, Any]] = []
    for sample_idx, data_sample in enumerate(selected):
        object_point_cloud = data_sample["obj_verts"].to(device)
        object_normals = data_sample["obj_vn"].to(device)
        contact_map_value = data_sample["obj_cmap"].to(device)
        contact_partition = data_sample["obj_partition"].to(device)
        _ = data_sample.get("obj_uv", None)

        contact_map_goal = torch.cat([object_point_cloud, object_normals, contact_map_value], dim=1).to(device)
        run_name = f"{args.dataset}_{args.object_name}_{sample_idx}"
        record = model.run_adam(
            contact_map_goal=contact_map_goal,
            contact_part=contact_partition,
            running_name=run_name,
        )
        grasps = _extract_top_k_grasps(
            record=record,
            robot_name="inspirehand",
            object_name=args.object_name,
            dataset_name=args.dataset,
            top_k=top_k,
        )
        all_grasps.extend(grasps)

    if args.output_pt:
        output_pt = os.path.abspath(args.output_pt)
    else:
        pack_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        out_dir = os.path.join(pack_root, "artifacts", "inspire_grasps")
        os.makedirs(out_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = args.object_name.replace("/", "_")
        output_pt = os.path.join(out_dir, f"{safe_name}_inspirehand_top{top_k}_{timestamp}.pt")

    os.makedirs(os.path.dirname(output_pt) or ".", exist_ok=True)
    torch.save(all_grasps, output_pt)

    summary = {
        "cedex_root": cedex_root,
        "dataset": args.dataset,
        "dataset_path": dataset_path,
        "object_name": args.object_name,
        "num_matching_samples": len(selected),
        "num_particles": int(args.num_particles),
        "max_iter": int(args.max_iter),
        "learning_rate": float(args.learning_rate),
        "save_top_k": int(top_k),
        "output_pt": output_pt,
        "num_saved_grasps": len(all_grasps),
        "device": device,
    }

    if args.output_json:
        output_json = os.path.abspath(args.output_json)
    else:
        output_json = os.path.splitext(output_pt)[0] + ".json"
    os.makedirs(os.path.dirname(output_json) or ".", exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[inspire_grasp] grasps saved: {output_pt}")
    print(f"[inspire_grasp] summary saved: {output_json}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
