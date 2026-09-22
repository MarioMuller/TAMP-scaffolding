from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from truss import Truss


SCAFFOLDS = {
    "20": "260804_RobArchDemo_ini.json",
    "34": "260804_RobArchDemo_two_boxes.json",
    "149": "260804_FoC_demo.json",
}
ROD_COLOR = (90, 90, 90)
GROUND_COLOR = (235, 235, 235)
SCALE = 0.001
DEFAULT_ROD_RADIUS = 0.012
DEFAULT_GAP = 0.8


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Display one or all experiment scaffolds in Viser for screenshots."
        )
    )
    parser.add_argument(
        "scaffold",
        choices=(*SCAFFOLDS, "all"),
        help="Scaffold size in rods, or 'all' for a side-by-side view.",
    )
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument(
        "--radius",
        type=float,
        default=DEFAULT_ROD_RADIUS,
        help="Display cylinder radius in metres (default: 0.012).",
    )
    parser.add_argument(
        "--gap",
        type=float,
        default=DEFAULT_GAP,
        help="Gap in metres between scaffolds in the 'all' view.",
    )
    return parser.parse_args()


def quaternion_from_z_to_vector(direction):
    direction = np.asarray(direction, dtype=float)
    direction /= np.linalg.norm(direction)
    z_axis = np.array([0.0, 0.0, 1.0])
    dot = float(np.dot(z_axis, direction))

    if dot > 1.0 - 1e-8:
        return np.array([1.0, 0.0, 0.0, 0.0])
    if dot < -1.0 + 1e-8:
        return np.array([0.0, 1.0, 0.0, 0.0])

    quaternion = np.concatenate(
        ([1.0 + dot], np.cross(z_axis, direction))
    )
    return quaternion / np.linalg.norm(quaternion)


def rod_segments(truss):
    rod_ids = sorted(truss.elements)
    segments = np.asarray(
        [
            [truss.nodes[start], truss.nodes[end]]
            for start, end in (
                truss.elements[rod_id] for rod_id in rod_ids
            )
        ],
        dtype=np.float32,
    )
    return rod_ids, segments * SCALE


def centre_on_ground(segments):
    minimum = segments.min(axis=(0, 1))
    maximum = segments.max(axis=(0, 1))
    centre = 0.5 * (minimum + maximum)
    offset = np.asarray(
        [centre[0], centre[1], minimum[2]],
        dtype=np.float32,
    )
    return segments - offset


def load_scaffold(scaffold_key):
    scaffold_path = (
        PROJECT_ROOT
        / "JSON"
        / "own_examples"
        / SCAFFOLDS[scaffold_key]
    )
    truss = Truss.from_json(scaffold_path)
    rod_ids, segments = rod_segments(truss)
    return scaffold_path, rod_ids, centre_on_ground(segments)


def scaffold_layout(selection, gap):
    scaffold_keys = (
        tuple(reversed(SCAFFOLDS))
        if selection == "all"
        else (selection,)
    )
    layout = []
    x_cursor = 0.0

    for scaffold_key in scaffold_keys:
        scaffold_path, rod_ids, segments = load_scaffold(scaffold_key)
        minimum = segments.min(axis=(0, 1))
        maximum = segments.max(axis=(0, 1))

        if selection == "all":
            segments = segments + np.asarray(
                [x_cursor - minimum[0], 0.0, 0.0],
                dtype=np.float32,
            )
            x_cursor += float(maximum[0] - minimum[0]) + gap

        layout.append((scaffold_key, scaffold_path, rod_ids, segments))

    combined = np.concatenate(
        [segments for _, _, _, segments in layout],
        axis=0,
    )
    minimum = combined.min(axis=(0, 1))
    maximum = combined.max(axis=(0, 1))
    horizontal_centre = 0.5 * (minimum[:2] + maximum[:2])
    offset = np.asarray(
        [horizontal_centre[0], horizontal_centre[1], 0.0],
        dtype=np.float32,
    )

    return [
        (scaffold_key, scaffold_path, rod_ids, segments - offset)
        for scaffold_key, scaffold_path, rod_ids, segments in layout
    ]


def display_scaffold(args):
    try:
        import viser
    except ImportError as error:
        raise ImportError(
            "viser is required in the active Python environment."
        ) from error

    layout = scaffold_layout(args.scaffold, args.gap)
    combined = np.concatenate(
        [segments for _, _, _, segments in layout],
        axis=0,
    )

    minimum = combined.min(axis=(0, 1))
    maximum = combined.max(axis=(0, 1))
    centre = 0.5 * (minimum + maximum)

    server = viser.ViserServer(port=args.port)
    server.scene.set_up_direction("+z")
    server.scene.world_axes.visible = False

    for scaffold_key, _, rod_ids, segments in layout:
        for rod_id, segment in zip(rod_ids, segments):
            start, end = segment
            direction = end - start
            server.scene.add_cylinder(
                f"scaffolds/{scaffold_key}/rods/rod_{rod_id}",
                radius=args.radius,
                height=float(np.linalg.norm(direction)),
                color=ROD_COLOR,
                radial_segments=16,
                material="standard",
                cast_shadow=True,
                receive_shadow=True,
                wxyz=quaternion_from_z_to_vector(direction),
                position=0.5 * (start + end),
            )

    margin = 0.12 * max(
        maximum[0] - minimum[0],
        maximum[1] - minimum[1],
    )
    server.scene.add_box(
        "scaffold/ground",
        color=GROUND_COLOR,
        dimensions=(
            float(maximum[0] - minimum[0] + 2.0 * margin),
            float(maximum[1] - minimum[1] + 2.0 * margin),
            0.02,
        ),
        position=(float(centre[0]), float(centre[1]), -0.015),
        cast_shadow=False,
        receive_shadow=0.55,
    )

    stop_button = server.gui.add_button("Stop")
    state = {"running": True}

    @stop_button.on_click
    def _(_event):
        state["running"] = False

    if args.scaffold == "all":
        print("Showing 149-, 34-, and 20-rod scaffolds from left to right.")
    else:
        scaffold_path = layout[0][1]
        print(
            f"Showing {args.scaffold}-rod scaffold from "
            f"{scaffold_path.name}."
        )
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
    if args.radius <= 0:
        raise ValueError("--radius must be positive.")
    if args.gap < 0:
        raise ValueError("--gap must be non-negative.")
    display_scaffold(args)


if __name__ == "__main__":
    main()
