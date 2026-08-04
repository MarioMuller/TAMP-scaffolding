import numpy as np
import matplotlib.pyplot as plt

from matplotlib.lines import Line2D
from matplotlib.widgets import Button, Slider
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.animation import FFMpegWriter


def display_structural_assembly(
    truss,
    removal_steps,
    scale=1.0,
    label_rods=False,
    video_path=None,
    seconds_per_step=2,
    fps=30,
):
    if not removal_steps:
        raise ValueError("The structural plan contains no steps.")

    # The final removal state is the initial assembly state.
    last_step = removal_steps[-1]

    frames = [
        {
            "active": last_step.rods_after,
            "supports": last_step.supports_after,
            "added_rod": None,
        }
    ]

    # Reverse removal transitions to obtain assembly transitions.
    for step in reversed(removal_steps):
        frames.append(
            {
                "active": step.rods_before,
                "supports": step.supports_before,
                "added_rod": step.rod_id,
            }
        )

    rod_ids = sorted(truss.elements)

    segments = []

    for rod_id in rod_ids:
        node_1, node_2 = truss.elements[rod_id]

        point_1 = np.asarray(
            truss.nodes[node_1],
            dtype=float,
        ) * scale

        point_2 = np.asarray(
            truss.nodes[node_2],
            dtype=float,
        ) * scale

        segments.append([point_1, point_2])

    all_points = np.asarray(
        [
            point
            for segment in segments
            for point in segment
        ]
    )

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    plt.subplots_adjust(bottom=0.18)

    rod_collection = Line3DCollection(segments)
    ax.add_collection3d(rod_collection)

    if label_rods:
        for rod_id, segment in zip(rod_ids, segments):
            midpoint = 0.5 * (
                np.asarray(segment[0])
                + np.asarray(segment[1])
            )

            ax.text(
                midpoint[0],
                midpoint[1],
                midpoint[2],
                str(rod_id),
                fontsize=7,
            )

    minimum = all_points.min(axis=0)
    maximum = all_points.max(axis=0)
    centre = 0.5 * (minimum + maximum)
    radius = max(0.5 * np.max(maximum - minimum), 1e-6)

    ax.set_xlim(centre[0] - radius, centre[0] + radius)
    ax.set_ylim(centre[1] - radius, centre[1] + radius)
    ax.set_zlim(centre[2] - radius, centre[2] + radius)
    ax.set_box_aspect((1, 1, 1))

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    ax.legend(
        handles=[
            Line2D([0], [0], color="yellowgreen", lw=3, label="installed"),
            Line2D([0], [0], color="orange", lw=4, label="just added"),
            Line2D([0], [0], color="magenta", lw=4, label="supported"),
            Line2D([0], [0], color="cornflowerblue", lw=3, label="grounded"),
            Line2D([0], [0], color="0.75", lw=1, label="not installed"),
        ],
        loc="upper right",
    )

    slider_ax = fig.add_axes([0.22, 0.08, 0.56, 0.03])

    slider = Slider(
        slider_ax,
        "Assembly step",
        0,
        len(frames) - 1,
        valinit=0,
        valstep=1,
    )

    previous_ax = fig.add_axes([0.05, 0.065, 0.12, 0.05])
    next_ax = fig.add_axes([0.83, 0.065, 0.12, 0.05])

    previous_button = Button(previous_ax, "Previous")
    next_button = Button(next_ax, "Next")

    def draw_frame(frame_index):
        frame = frames[int(frame_index)]

        active = set(frame["active"])
        supports = dict(frame["supports"])
        supported_rods = set(supports.values())
        added_rod = frame["added_rod"]

        colors = []
        widths = []

        for rod_id in rod_ids:
            if rod_id not in active:
                colors.append((0.75, 0.75, 0.75, 0.18))
                widths.append(1.0)

            elif rod_id in supported_rods:
                colors.append("magenta")
                widths.append(4.0)

            elif rod_id == added_rod:
                colors.append("orange")
                widths.append(4.0)

            elif rod_id in truss.grounded_rods:
                colors.append("cornflowerblue")
                widths.append(3.0)

            else:
                colors.append("yellowgreen")
                widths.append(2.5)

        rod_collection.set_colors(colors)
        rod_collection.set_linewidths(widths)

        support_text = ", ".join(
            f"{support} → rod {rod_id}"
            for support, rod_id in supports.items()
        )

        if not support_text:
            support_text = "none"

        ax.set_title(
            f"Assembly step {int(frame_index)}/{len(frames) - 1}\n"
            f"added rod: {added_rod if added_rod is not None else 'none'} | "
            f"supports: {support_text}"
        )

        fig.canvas.draw_idle()

    def previous(_event):
        slider.set_val(max(0, int(slider.val) - 1))

    def next_step(_event):
        slider.set_val(
            min(len(frames) - 1, int(slider.val) + 1)
        )

    def key_press(event):
        if event.key in {"left", "p"}:
            previous(event)
        elif event.key in {"right", "n", " "}:
            next_step(event)

    slider.on_changed(draw_frame)
    previous_button.on_clicked(previous)
    next_button.on_clicked(next_step)
    fig.canvas.mpl_connect("key_press_event", key_press)

    draw_frame(0)

    if video_path is not None:
        writer = FFMpegWriter(
            fps=fps,
            codec="libx264",
            extra_args=["-pix_fmt", "yuv420p"],
            metadata={"title": "Structural assembly sequence"},
        )

        # Hide interactive controls in the exported video.
        slider_ax.set_visible(False)
        previous_ax.set_visible(False)
        next_ax.set_visible(False)

        frames_per_step = max(1, round(seconds_per_step * fps))

        print(f"Exporting video to {video_path}...")

        with writer.saving(fig, video_path, dpi=120):
            for frame_index in range(len(frames)):
                draw_frame(frame_index)

                # Hold each construction state for the requested duration.
                for _ in range(frames_per_step):
                    writer.grab_frame()

        print("Video export completed.")

        # Restore controls for the interactive window.
        slider_ax.set_visible(True)
        previous_ax.set_visible(True)
        next_ax.set_visible(True)
        draw_frame(0)

    plt.show()