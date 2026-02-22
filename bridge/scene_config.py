"""Module C: Scene configuration registry for HOI-to-G1 pipeline.

Maps HOI object names → IsaacLab-Arena scene + asset configs.
Centralizes all scene-specific parameters (table height, workspace bounds,
robot placement, object initial pose) so retarget and replay modules
share a single source of truth.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ObjectConfig:
    """Object metadata for IsaacLab-Arena."""
    arena_name: str          # name in IsaacLab-Arena object_library
    initial_pos: tuple[float, float, float] = (0.4, 0.0, 0.1)
    initial_quat_wxyz: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)


@dataclass
class SceneConfig:
    """All parameters needed to place an object + robot in a scene."""
    # IsaacLab-Arena identifiers
    arena_scene_name: str        # CLI subcommand name (e.g. "kitchen_pick_and_place")
    arena_background: str        # background asset name (e.g. "kitchen")

    # Object placement on table
    table_z: float               # table surface Z in world frame
    object_align_pos: tuple[float, float, float]  # where first HOI frame maps to

    # G1 workspace bounds for trajectory clipping
    clip_z: tuple[float, float] = (0.05, 0.40)
    clip_xy_min: tuple[float, float] | None = None
    clip_xy_max: tuple[float, float] | None = None

    # G1 robot defaults
    base_height: float = 0.75
    default_base_pos_w: tuple[float, float, float] = (-0.55, 0.0, 0.0)
    default_base_yaw_deg: float = 0.0
    default_walk_target_offset: tuple[float, float, float] = (-0.35, 0.0, 0.0)

    # Wrist grasp defaults
    right_wrist_pos_obj: tuple[float, float, float] = (-0.20, -0.03, 0.10)
    replay_base_height: float | None = None


# ---------------------------------------------------------------------------
# Scene registry
# ---------------------------------------------------------------------------
SCENES: dict[str, SceneConfig] = {
    "kitchen_pick_and_place": SceneConfig(
        arena_scene_name="kitchen_pick_and_place",
        arena_background="kitchen",
        table_z=0.10,
        object_align_pos=(0.40, 0.00, 0.10),
        clip_z=(0.05, 0.40),
        clip_xy_min=(0.05, -0.45),
        clip_xy_max=(0.65, 0.45),
        base_height=0.75,
        default_base_pos_w=(-0.55, 0.0, 0.0),
        default_walk_target_offset=(-0.35, 0.0, 0.0),
        right_wrist_pos_obj=(-0.16, -0.05, 0.06),
        replay_base_height=0.80,
    ),
    "galileo_g1_locomanip_pick_and_place": SceneConfig(
        arena_scene_name="galileo_g1_locomanip_pick_and_place",
        arena_background="galileo_locomanip",
        table_z=0.07,
        object_align_pos=(0.5785, 0.18, 0.07),
        clip_z=(0.06, 0.40),
        base_height=0.78,
        default_base_pos_w=(-0.80, 0.18, 0.0),
        default_walk_target_offset=(-0.45, 0.0, 0.0),
        replay_base_height=0.78,
    ),
    "packing_table": SceneConfig(
        arena_scene_name="kitchen_pick_and_place",  # reuse kitchen runner
        arena_background="packing_table",
        table_z=0.08,
        object_align_pos=(0.40, 0.00, 0.08),
        clip_z=(0.05, 0.35),
        clip_xy_min=(0.05, -0.40),
        clip_xy_max=(0.60, 0.40),
        base_height=0.75,
    ),
}

# ---------------------------------------------------------------------------
# Object name mapping: HOI model name → IsaacLab-Arena asset name
# ---------------------------------------------------------------------------
OBJECT_MAP: dict[str, ObjectConfig] = {
    # YCB objects (direct match)
    "cracker_box": ObjectConfig("cracker_box"),
    "mustard_bottle": ObjectConfig("mustard_bottle"),
    "sugar_box": ObjectConfig("sugar_box"),
    "tomato_soup_can": ObjectConfig("tomato_soup_can"),
    "power_drill": ObjectConfig("power_drill"),
    # Aliases from HOI datasets
    "003_cracker_box": ObjectConfig("cracker_box"),
    "006_mustard_bottle": ObjectConfig("mustard_bottle"),
    "004_sugar_box": ObjectConfig("sugar_box"),
    "005_tomato_soup_can": ObjectConfig("tomato_soup_can"),
    # Generic fallbacks
    "smallbox": ObjectConfig("cracker_box"),
    "box": ObjectConfig("cracker_box"),
    "brown_box": ObjectConfig("brown_box"),
    "bottle": ObjectConfig("mustard_bottle"),
    "can": ObjectConfig("tomato_soup_can"),
    "drill": ObjectConfig("power_drill"),
}


def get_scene(name: str) -> SceneConfig:
    """Get scene config by name, fallback to kitchen."""
    return SCENES.get(name, SCENES["kitchen_pick_and_place"])


def get_object(name: str) -> ObjectConfig:
    """Get object config by HOI name, fallback to cracker_box."""
    key = name.lower().strip()
    return OBJECT_MAP.get(key, ObjectConfig(key))


def scene_defaults_for_retarget(name: str) -> dict:
    """Return dict compatible with hoi_to_g1_retarget.py SCENE_DEFAULTS format."""
    sc = get_scene(name)
    return dict(
        table_z=sc.table_z,
        align_pos=list(sc.object_align_pos),
        clip_z=sc.clip_z,
        clip_xy=(list(sc.clip_xy_min) if sc.clip_xy_min else None,
                 list(sc.clip_xy_max) if sc.clip_xy_max else None),
        base_height=sc.base_height,
    )


# ---------------------------------------------------------------------------
# Dynamic scene registration (for LW-BenchHub / external scenes)
# ---------------------------------------------------------------------------
def register_scene(name: str, config: SceneConfig) -> None:
    """Register a new scene config at runtime. Does not overwrite existing."""
    if name not in SCENES:
        SCENES[name] = config


def register_scene_from_json(json_path: str) -> str:
    """Register a scene from a lwbench_scene_adapter.py output JSON.

    Returns the scene name.
    """
    import json as _json
    with open(json_path, "r", encoding="utf-8") as f:
        data = _json.load(f)
    sc_dict = data["scene_config"]
    name = sc_dict["arena_scene_name"]
    config = SceneConfig(
        arena_scene_name=sc_dict["arena_scene_name"],
        arena_background=sc_dict["arena_background"],
        table_z=float(sc_dict["table_z"]),
        object_align_pos=tuple(sc_dict["object_align_pos"]),
        clip_z=tuple(sc_dict.get("clip_z", (0.05, 0.40))),
        clip_xy_min=tuple(sc_dict["clip_xy_min"]) if sc_dict.get("clip_xy_min") else None,
        clip_xy_max=tuple(sc_dict["clip_xy_max"]) if sc_dict.get("clip_xy_max") else None,
        base_height=float(sc_dict.get("base_height", 0.75)),
        default_base_pos_w=tuple(sc_dict["default_base_pos_w"]),
        default_base_yaw_deg=float(sc_dict.get("default_base_yaw_deg", 0.0)),
        default_walk_target_offset=tuple(sc_dict["default_walk_target_offset"]),
        right_wrist_pos_obj=tuple(sc_dict.get("right_wrist_pos_obj", (-0.18, -0.04, 0.08))),
        replay_base_height=float(sc_dict["replay_base_height"]) if sc_dict.get("replay_base_height") else None,
    )
    register_scene(name, config)
    return name
