from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.structural_experiment_utils import load_filtered_truss


COLORS = {
    # The unused-rod color includes Matplotlib's 0.18 alpha over white.
    "not_installed": (244, 244, 244),
    "installed": (90, 90, 90),
    "added": (255, 106, 0),
    "supported": (255, 0, 255),
    "grounded": (90, 90, 90),
}
DEFAULT_ROD_RADIUS = 0.010


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Replay a saved rigidity-only structural plan in Viser. "
            "This shows rods and support assignments, not robot motion."
        )
    )
    parser.add_argument("replay_jsonl", type=Path)
    parser.add_argument("--repetition", type=int)
    parser.add_argument("--run-index", type=int)
    parser.add_argument(
        "--list",
        action="store_true",
        help="List saved runs without opening Viser.",
    )
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--pause-time", type=float, default=0.8)
    parser.add_argument(
        "--rod-radius",
        type=float,
        default=DEFAULT_ROD_RADIUS,
        help="Display-only cylinder radius in metres.",
    )
    parser.add_argument(
        "--show-not-placed-rods",
        action="store_true",
        help="Show rods that have not yet been placed in faded grey.",
    )
    return parser.parse_args()


def load_replay_file(path):
    metadata = None
    runs = []

    with path.open(encoding="utf-8") as replay_file:
        for line_number, line in enumerate(replay_file, start=1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON on line {line_number} of {path}."
                ) from error

            record_type = record.get("record_type")
            if record_type == "metadata":
                metadata = record
            elif record_type == "run":
                runs.append(record)

    if metadata is None:
        raise ValueError(f"Replay metadata is missing from {path}.")
    if metadata.get("version") != 1:
        raise ValueError(
            f"Unsupported replay version: {metadata.get('version')!r}."
        )

    return metadata, runs


def list_runs(runs):
    if not runs:
        print("No successful runs were saved.")
        return

    print("repetition  run  seed    rods  removed-prefix  steps")
    for run in runs:
        print(
            f"{run['repetition']:>10}  "
            f"{run['run_index']:>3}  "
            f"{run['seed']:>6}  "
            f"{len(run['included_rods']):>4}  "
            f"{run['removed_prefix_count']:>14}  "
            f"{len(run['structural_steps']):>5}"
        )


def select_run(runs, repetition, run_index):
    matches = [
        run
        for run in runs
        if (
            repetition is None
            or run["repetition"] == repetition
        )
        and (
            run_index is None
            or run["run_index"] == run_index
        )
    ]

    if not matches:
        raise ValueError(
            "No successful replay matches the requested repetition/run. "
            "Use --list to see the available runs."
        )

    if repetition is None and run_index is None:
        full_runs = [
            run for run in matches if run["removed_prefix_count"] == 0
        ]
        return full_runs[0] if full_runs else matches[0]

    if len(matches) != 1:
        raise ValueError(
            f"The selection matches {len(matches)} runs. Specify both "
            "--repetition and --run-index."
        )

    return matches[0]


def assembly_frames(run):
    steps = run["structural_steps"]
    active = set(run["included_rods"])
    active.difference_update(step["rod_id"] for step in steps)

    frames = [
        {
            "active": set(active),
            "supports": (
                dict(steps[-1]["supports_after"])
                if steps
                else dict(run["initial_supports"])
            ),
            "added_rod": None,
        }
    ]

    for step in reversed(steps):
        active.add(step["rod_id"])
        frames.append(
            {
                "active": set(active),
                "supports": dict(step["supports_before"]),
                "added_rod": step["rod_id"],
            }
        )

    return frames


def rod_segments(truss, scale):
    rod_ids = sorted(truss.elements)
    segments = []

    for rod_id in rod_ids:
        node_1, node_2 = truss.elements[rod_id]
        segments.append(
            [
                np.asarray(truss.nodes[node_1], dtype=np.float32) * scale,
                np.asarray(truss.nodes[node_2], dtype=np.float32) * scale,
            ]
        )

    return rod_ids, np.asarray(segments, dtype=np.float32)


def rod_visual_state(frame, rod_id, grounded_rods):
    active = set(frame["active"])
    supported_rods = set(frame["supports"].values())
    added_rod = frame["added_rod"]

    if rod_id not in active:
        return "not_installed"
    if rod_id in supported_rods:
        return "supported"
    if rod_id == added_rod:
        return "added"
    if rod_id in grounded_rods:
        return "grounded"
    return "installed"


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


def add_ground_plane(server, segments):
    minimum = segments.min(axis=(0, 1))
    maximum = segments.max(axis=(0, 1))
    centre = 0.5 * (minimum + maximum)
    margin = 0.1 * max(maximum[0] - minimum[0], maximum[1] - minimum[1])

    plane_thickness = 0.01
    server.scene.add_box(
        "scaffold/ground",
        color=(190, 190, 190),
        dimensions=(
            max(maximum[0] - minimum[0] + 2.0 * margin, 1.0),
            max(maximum[1] - minimum[1] + 2.0 * margin, 1.0),
            plane_thickness,
        ),
        position=(
            float(centre[0]),
            float(centre[1]),
            float(minimum[2]) - 0.002 - 0.5 * plane_thickness,
        ),
        cast_shadow=False,
        receive_shadow=0.55,
    )


def display_replay(metadata, run, args):
    try:
        import viser
    except ImportError as error:
        raise ImportError(
            "viser is required. Install it in the active environment."
        ) from error

    truss = load_filtered_truss(
        metadata["truss"],
        included_rods=run["included_rods"],
    )
    scale = float(metadata.get("scale", 0.001))
    rod_ids, segments = rod_segments(truss, scale)
    frames = assembly_frames(run)

    server = viser.ViserServer(port=args.port)
    server.scene.set_up_direction("+z")
    server.scene.world_axes.visible = False

    add_ground_plane(server, segments)
    rod_handles = {}

    for rod_id, segment in zip(rod_ids, segments):
        initial_state = rod_visual_state(
            frames[0],
            rod_id,
            truss.grounded_rods,
        )
        start, end = segment
        direction = end - start
        rod_handles[rod_id] = {}

        for variant_state, color in COLORS.items():
            rod_handles[rod_id][variant_state] = server.scene.add_cylinder(
                f"scaffold/rods/rod_{rod_id}/{variant_state}",
                radius=args.rod_radius,
                height=float(np.linalg.norm(direction)),
                color=color,
                radial_segments=16,
                material="standard",
                cast_shadow=True,
                receive_shadow=True,
                wxyz=quaternion_from_z_to_vector(direction),
                position=0.5 * (start + end),
                visible=(
                    variant_state == initial_state
                    and (
                        args.show_not_placed_rods
                        or initial_state != "not_installed"
                    )
                ),
            )

    step_slider = server.gui.add_slider(
        "Assembly step",
        min=0,
        max=len(frames) - 1,
        step=1,
        initial_value=0,
    )
    play_checkbox = server.gui.add_checkbox("Play", initial_value=False)
    pause_field = server.gui.add_number(
        "Pause time (s)",
        initial_value=args.pause_time,
        min=0.01,
        step=0.05,
    )
    previous_button = server.gui.add_button("Previous")
    next_button = server.gui.add_button("Next")
    stop_button = server.gui.add_button("Stop")
    status = server.gui.add_markdown("")

    stopped = False

    def show_frame(frame_index):
        frame_index = max(0, min(int(frame_index), len(frames) - 1))
        frame = frames[frame_index]

        with server.atomic():
            for rod_id, variants in rod_handles.items():
                state = rod_visual_state(
                    frame,
                    rod_id,
                    truss.grounded_rods,
                )
                for variant_state, handle in variants.items():
                    handle.visible = (
                        variant_state == state
                        and (
                            args.show_not_placed_rods
                            or state != "not_installed"
                        )
                    )

        supports = ", ".join(
            f"{support}: rod {rod_id}"
            for support, rod_id in sorted(frame["supports"].items())
        ) or "none"
        added_rod = frame["added_rod"]
        unused_description = (
            ", grey not installed"
            if args.show_not_placed_rods
            else ""
        )
        status.content = (
            f"**Repetition:** {run['repetition']}  \n"
            f"**Run:** {run['run_index']}  \n"
            f"**Assembly step:** {frame_index}/{len(frames) - 1}  \n"
            f"**Added rod:** {added_rod if added_rod is not None else 'none'}  \n"
            f"**Active rods:** {len(frame['active'])}  \n"
            f"**Supports:** {supports}  \n"
            "**Colors:** grey installed/grounded, orange added, "
            f"magenta supported{unused_description}"
        )

    @previous_button.on_click
    def _(_event):
        play_checkbox.value = False
        step_slider.value = max(0, int(step_slider.value) - 1)
        show_frame(step_slider.value)

    @next_button.on_click
    def _(_event):
        play_checkbox.value = False
        step_slider.value = min(
            len(frames) - 1,
            int(step_slider.value) + 1,
        )
        show_frame(step_slider.value)

    @step_slider.on_update
    def _(event):
        if not play_checkbox.value:
            show_frame(event.target.value)

    @stop_button.on_click
    def _(_event):
        nonlocal stopped
        stopped = True

    show_frame(0)
    print(
        f"Selected repetition {run['repetition']}, run {run['run_index']}, "
        f"{len(run['included_rods'])} rods."
    )
    print(f"[viser] Open http://localhost:{args.port}")
    print("[viser] Press Stop in the GUI or Ctrl-C to exit.")

    try:
        while not stopped:
            if play_checkbox.value:
                next_step = (int(step_slider.value) + 1) % len(frames)
                step_slider.value = next_step
                show_frame(next_step)

            time.sleep(float(pause_field.value))
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()


def main():
    args = parse_args()
    if args.rod_radius <= 0:
        raise ValueError("--rod-radius must be positive.")

    metadata, runs = load_replay_file(args.replay_jsonl)

    if args.list:
        list_runs(runs)
        return

    run = select_run(runs, args.repetition, args.run_index)
    display_replay(metadata, run, args)


if __name__ == "__main__":
    main()
