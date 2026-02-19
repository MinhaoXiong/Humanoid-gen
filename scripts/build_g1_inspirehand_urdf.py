#!/usr/bin/env python3
"""Build G1 + InspireHand combined URDF.

Takes the G1 body URDF (without hands or with Unitree hands) and attaches
InspireHand left/right at the wrist_yaw_link via fixed joints.
"""
import argparse
import re
import os


def _strip_robot_wrapper(urdf_text: str) -> str:
    """Remove <robot> and </robot> tags, return inner content."""
    text = re.sub(r'<\?xml[^>]*\?>\s*', '', urdf_text)
    text = re.sub(r'<robot[^>]*>', '', text, count=1)
    text = re.sub(r'</robot>\s*$', '', text)
    # Remove mujoco compiler directives
    text = re.sub(r'<mujoco>.*?</mujoco>', '', text, flags=re.DOTALL)
    return text.strip()


def _extract_g1_body(g1_urdf: str) -> str:
    """Extract G1 body up to and including wrist_yaw_link, removing Unitree hand sections."""
    lines = g1_urdf.split('\n')
    result = []
    skip = False
    for line in lines:
        # Start skipping at Unitree hand joints (left or right)
        if re.search(r'joint name="(left|right)_hand_palm_joint"', line):
            skip = True
        # Stop skipping at right shoulder link (comes before the joint)
        if skip and re.search(r'link name="right_shoulder_pitch_link"', line):
            skip = False
        # Stop skipping at </robot>
        if skip and '</robot>' in line:
            skip = False
        if not skip:
            result.append(line)
    return '\n'.join(result)


def _make_attach_joint(side: str, hand_base_link: str) -> str:
    """Create fixed joint attaching InspireHand to G1 wrist."""
    wrist_link = f"{side}_wrist_yaw_link"
    # Offset from wrist_yaw_link to hand base (approximate, same as Unitree palm joint)
    if side == "left":
        xyz = "0.0415 0.003 0"
    else:
        xyz = "0.0415 -0.003 0"
    return f"""
  <joint name="{side}_inspire_hand_joint" type="fixed">
    <origin xyz="{xyz}" rpy="0 0 0"/>
    <parent link="{wrist_link}"/>
    <child link="{hand_base_link}"/>
  </joint>"""


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--g1-urdf", required=True, help="G1 URDF with Unitree hand")
    p.add_argument("--left-hand-urdf", required=True, help="InspireHand left URDF")
    p.add_argument("--right-hand-urdf", required=True, help="InspireHand right URDF")
    p.add_argument("--output", required=True, help="Output combined URDF path")
    p.add_argument("--hand-mesh-dir", default="inspire_hand_meshes",
                    help="Relative mesh dir for InspireHand STLs in output URDF")
    args = p.parse_args()

    with open(args.g1_urdf) as f:
        g1_text = f.read()
    with open(args.left_hand_urdf) as f:
        left_text = f.read()
    with open(args.right_hand_urdf) as f:
        right_text = f.read()

    # Extract G1 body without Unitree hands
    body = _extract_g1_body(g1_text)
    # Remove </robot> from body
    body = body.replace('</robot>', '').rstrip()

    # Process hand URDFs: strip wrapper, fix mesh paths
    left_inner = _strip_robot_wrapper(left_text)
    right_inner = _strip_robot_wrapper(right_text)

    # Replace mesh paths to point to hand_mesh_dir
    left_inner = re.sub(r'filename="\.?/?meshes/', f'filename="{args.hand_mesh_dir}/', left_inner)
    right_inner = re.sub(r'filename="\.?/?meshes/', f'filename="{args.hand_mesh_dir}/', right_inner)

    # Build combined URDF
    combined = body + "\n"
    combined += "\n  <!-- Left InspireHand -->"
    combined += _make_attach_joint("left", "L_hand_base_link")
    combined += "\n" + left_inner + "\n"
    combined += "\n  <!-- Right InspireHand -->"
    combined += _make_attach_joint("right", "R_hand_base_link")
    combined += "\n" + right_inner + "\n"
    combined += "\n</robot>\n"

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, 'w') as f:
        f.write(combined)
    print(f"[build_urdf] Written: {args.output}")


if __name__ == "__main__":
    main()
