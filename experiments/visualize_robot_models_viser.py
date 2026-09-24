from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rai.scene import RaiScene
from rai.viser_replay import ViserPlanReplayer


GROUND_COLOR = (235, 235, 235)
DEFAULT_ARM_POSES = {
    "a1_": [-0.45, -1.20, 1.35, -1.25, -1.57, 0.0],
    "a2_": [0.45, -1.20, 1.35, -1.25, -1.57, 0.0],
    "h1_a1_": [0.0, -1.15, 1.40, -1.20, -1.57, 0.0],
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Display the dual-arm main robot and one single-arm support "
            "robot side by side in Viser."
        )
    )
    parser.add_argument("--port", type=int, default=8082)
    parser.add_argument(
        "--spacing",
        type=float,
        default=2.0,
        help="Distance in metres between the two mobile bases.",
    )
    return parser.parse_args()


def build_robot_scene(spacing):
    scene = RaiScene()
    scene.import_main_husky(arm_count=2)
    scene.import_support_husky(
        name="h1",
        base_q=(0.0, 0.5 * spacing, 0.0),
    )
    scene.C.setJointState(
        [0.0, -0.5 * spacing, 0.0],
        ["husky_base_XYPhi_joint"],
    )

    joint_names = scene.C.getJointNames()
    for prefix, joint_values in DEFAULT_ARM_POSES.items():
        arm_joint_names = [
            name for name in joint_names if name.startswith(prefix)
        ]
        scene.C.setJointState(joint_values, arm_joint_names)

    return scene.C


def joint_label(joint_name):
    return (
        joint_name
        .removesuffix("_joint")
        .replace("shoulder_", "shoulder ")
        .replace("wrist_", "wrist ")
        .replace("_", " ")
    )


def display_robots(args):
    try:
        import viser
    except ImportError as error:
        raise ImportError(
            "viser is required in the active Python environment."
        ) from error

    config = build_robot_scene(args.spacing)
    server = viser.ViserServer(port=args.port)
    server.scene.set_up_direction("+z")
    server.scene.world_axes.visible = False

    mesh_handles = {}

    for frame in config.getFrames():
        if frame.name == "table":
            continue

        vertices = np.asarray(frame.getMeshPoints(), dtype=np.float32)
        faces = np.asarray(frame.getMeshTriangles(), dtype=np.uint32)

        if vertices.ndim < 2 or faces.ndim < 2:
            continue

        info = frame.info()
        raw_color = info.get("color", [0.7, 0.7, 0.7])
        color = tuple(int(component * 255) for component in raw_color[:3])
        opacity = None

        if len(raw_color) > 3 and raw_color[3] < 1.0:
            opacity = float(raw_color[3])

        base_color = ViserPlanReplayer._robot_base_color(
            frame.name,
            info,
        )
        if base_color is not None:
            color = base_color
            opacity = 0.35

        mesh_handles[frame.name] = server.scene.add_mesh_simple(
            name=f"robots/{frame.name}",
            vertices=vertices,
            faces=faces,
            color=color,
            flat_shading=False,
            opacity=opacity,
            position=np.asarray(frame.getPosition(), dtype=np.float32),
            wxyz=np.asarray(frame.getQuaternion(), dtype=np.float32),
            cast_shadow=True,
            receive_shadow=True,
        )

    def update_robot_poses():
        for frame in config.getFrames():
            handle = mesh_handles.get(frame.name)
            if handle is None:
                continue

            handle.position = np.asarray(
                frame.getPosition(),
                dtype=np.float32,
            )
            handle.wxyz = np.asarray(
                frame.getQuaternion(),
                dtype=np.float32,
            )

    joint_names = config.getJointNames()
    joint_index = {
        name: index for index, name in enumerate(joint_names)
    }
    initial_joint_state = config.getJointState().copy()
    current_joint_state = initial_joint_state.copy()
    initial_joint_values = dict(zip(joint_names, initial_joint_state))
    slider_by_joint = {}
    joint_update_lock = threading.Lock()

    def add_joint_controls(folder_label, prefix):
        arm_joint_names = [
            name for name in joint_names if name.startswith(prefix)
        ]

        with server.gui.add_folder(
            folder_label,
            expand_by_default=False,
        ):
            for joint_name in arm_joint_names:
                slider = server.gui.add_slider(
                    label=joint_label(joint_name.removeprefix(prefix)),
                    min=-3.14,
                    max=3.14,
                    step=0.01,
                    initial_value=float(initial_joint_values[joint_name]),
                )
                slider_by_joint[joint_name] = slider

                def register_callback(slider_handle, active_joint_name):
                    @slider_handle.on_update
                    def _on_slider_update(event):
                        with joint_update_lock:
                            current_joint_state[
                                joint_index[active_joint_name]
                            ] = float(event.target.value)
                            config.setJointState(current_joint_state)
                            update_robot_poses()

                register_callback(slider, joint_name)

    add_joint_controls("Main robot: arm 1", "a1_")
    add_joint_controls("Main robot: arm 2", "a2_")
    add_joint_controls("Support robot", "h1_a1_")

    reset_button = server.gui.add_button("Reset joints")

    @reset_button.on_click
    def _(_event):
        with joint_update_lock:
            current_joint_state[:] = initial_joint_state
            config.setJointState(current_joint_state)
            update_robot_poses()

        for joint_name, slider in slider_by_joint.items():
            slider.value = float(initial_joint_values[joint_name])

    server.scene.add_box(
        "ground",
        color=GROUND_COLOR,
        dimensions=(3.0, args.spacing + 3.0, 0.02),
        position=(0.0, 0.0, -0.015),
        cast_shadow=False,
        receive_shadow=0.55,
    )

    stop_button = server.gui.add_button("Stop")
    state = {"running": True}

    @stop_button.on_click
    def _(_event):
        state["running"] = False

    print(f"[viser] Open http://localhost:{args.port}")
    print("[viser] Press Stop in the GUI or Ctrl-C to exit.")

    try:
        while state["running"]:
            time.sleep(0.2)
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()


def main():
    args = parse_args()
    if args.spacing <= 0:
        raise ValueError("--spacing must be positive.")
    display_robots(args)


if __name__ == "__main__":
    main()
