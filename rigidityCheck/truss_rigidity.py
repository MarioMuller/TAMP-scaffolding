from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import time
from time import perf_counter

from mpl_toolkits.mplot3d.art3d import Line3DCollection

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from .Datastructures import ElementObject, ElementStatus
    from .rigiditycheck import AlgebraicChecker
    from .utils import closest_points_between_segments
except ImportError:
    from Datastructures import ElementObject, ElementStatus
    from rigiditycheck import AlgebraicChecker
    from utils import closest_points_between_segments


@dataclass(frozen=True)
class RigidityResult:
    is_rigid: bool
    rank: int
    dof: int
    rows: int
    statuses: dict[int, ElementStatus] = field(default_factory=dict)

    failure_modes: tuple[np.ndarray, ...] = ()
    failure_vertices: tuple = ()
    failure_elements: dict = field(default_factory=dict)
    
    # IDs of the artificial off-axis orientation vertices Q.
    failure_orientation_vertex_ids: frozenset[int] = frozenset()
    
    @property
    def nullity(self) -> int:
        return self.dof - self.rank

    @property
    def status_names(self) -> dict[int, str]:
        return {
            rod_id: status.name
            for rod_id, status in self.statuses.items()
        }


class TrussRigidityChecker:
    """
    Adapter from the JSON-backed Truss model to rigiditycheck.py.

    follows the original checker implementation:
    - rods are converted to ElementObject instances
    - rods sharing a JSON node become coupled/assembled elements
    - grounded rods use ElementObject.is_grounded
    - rigidity is evaluated through AlgebraicChecker.Check
    """

    def __init__(self, truss):
        self.truss = truss

    def check(
        self,
        active_rods: Iterable[int],
        supported_rods: Iterable[int] | None = None,
    ) -> RigidityResult:
        total_start = perf_counter()
        active = set(active_rods)
        if not active:
            return RigidityResult(
                is_rigid=True,
                rank=0,
                dof=0,
                rows=0,
                statuses={},
                failure_modes=(),
                failure_vertices=(),
                failure_elements={},
                failure_orientation_vertex_ids=frozenset(),
            )

        build_elements_start = perf_counter()

        element_objects, rod_to_index, index_to_rod = self._build_element_objects(
            active,
            set(supported_rods or ()),
        )

        build_elements_end = perf_counter()
        
        assembled_indices = [rod_to_index[rod_id] for rod_id in sorted(active)]
        
        build_matrix_start = perf_counter()
        
        matrix_result = AlgebraicChecker.BuildRigidityMatrix(
            assembled_indices,
            element_objects,
        )
        
        build_matrix_end = perf_counter()
        
        K = matrix_result.matrix
        matrix_dof = K.shape[1]
        
        analysis_start = perf_counter()

        matrix_rank, failure_modes = AlgebraicChecker.AnalyzeSparseQR(K)
        # matrix_rank = np.linalg.matrix_rank(K)
        # failure_modes = None    
        print("QR")

        analysis_end = perf_counter()

        if matrix_rank == matrix_dof:
            statuses = {
                rod_id: ElementStatus.fixed
                for rod_id in active
            }
        else:
            statuses = self._statuses_from_nullspace(
                failure_modes=failure_modes,
                matrix_result=matrix_result,
                index_to_rod=index_to_rod,
                active_rods=active,
                tolerance=1e-7,
            )
        
        total_end = perf_counter()

        # print("\nRigidity timing breakdown")
        # print("-------------------------")
        # print(
        #     f"Build element objects: "
        #     f"{build_elements_end - build_elements_start:.6f} s"
        # )
        # print(
        #     f"Build rigidity matrix: "
        #     f"{build_matrix_end - build_matrix_start:.6f} s"
        # )
        # print(
        #     f"Analysis:           "
        #     f"{analysis_end - analysis_start:.6f} s"
        # )
        # print(
        #     f"Total check:           "
        #     f"{total_end - total_start:.6f} s"
        # )
        # print(f"Matrix shape:          {K.shape}")

        return RigidityResult(
            is_rigid=matrix_rank == matrix_dof,
            rank=matrix_rank,
            dof=matrix_dof,
            rows=K.shape[0],
            statuses=statuses,
            failure_modes=failure_modes,
            failure_vertices=tuple(matrix_result.vertex_list),
            failure_elements=matrix_result.elements_dict,
            failure_orientation_vertex_ids=frozenset(
                vertex.id
                for orientation_pair in matrix_result.orientation_vertices.values()
                for vertex in orientation_pair
            ),
        )

    def is_rigid(
        self,
        active_rods: Iterable[int],
        supported_rods: Iterable[int] | None = None,
    ) -> bool:
        return self.check(active_rods, supported_rods=supported_rods).is_rigid

    def choose_support_targets(
        self,
        active_rods: Iterable[int],
        already_supported: Iterable[int] | None = None,
        max_targets: int = 2,
        key=None,
    ) -> list[int]:
        active = set(active_rods)
        supported = set(already_supported or ())
        chosen: list[int] = []

        # Calculate the initial state only once.
        current_result = self.check(
            active,
            supported_rods=supported,
        )

        while (
            not current_result.is_rigid
            and len(chosen) < max_targets
        ):
            # Supporting an already fixed rod cannot remove a failure mode.
            candidates = [
                rod
                for rod in active
                if (
                    rod not in supported
                    and current_result.statuses[rod]
                    != ElementStatus.fixed
                )
            ]

            # Highest non-fixed rod is tested first.
            if key is not None:
                candidates.sort(key=key, reverse=True)

            best_rod = None
            best_result = current_result

            for rod in candidates:
                result = self.check(
                    active,
                    supported_rods=supported | {rod},
                )

                # Full rank is the best possible result.
                # Because candidates are height-sorted, this is also the
                # highest candidate that makes the structure rigid.
                if result.is_rigid:
                    chosen.append(rod)
                    return chosen

                if result.rank > best_result.rank:
                    best_rod = rod
                    best_result = result

            # No tested support increased the rank.
            if best_rod is None:
                break

            chosen.append(best_rod)
            supported.add(best_rod)

            # Reuse the result instead of running the same QR again.
            current_result = best_result

        return chosen

    def _build_element_objects(
        self,
        active_rods: set[int],
        supported_rods: set[int],
    ):
        rod_ids = sorted(self.truss.elements)
        rod_to_index = {
            rod_id: index
            for index, rod_id in enumerate(rod_ids)
        }
        index_to_rod = {
            index: rod_id
            for rod_id, index in rod_to_index.items()
        }

        coupled_rods = self._coupled_rods()
        element_objects = []

        for rod_id in rod_ids:
            n1, n2 = self.truss.elements[rod_id]
            vertices = [
                np.asarray(self.truss.nodes[n1], dtype=float),
                np.asarray(self.truss.nodes[n2], dtype=float),
            ]
            is_grounded = (
                rod_id in self.truss.grounded_rods
                or rod_id in supported_rods
            )
            coupled_elements = [
                rod_to_index[coupled_rod]
                for coupled_rod in sorted(coupled_rods[rod_id])
            ]

            element = ElementObject(
                index=rod_to_index[rod_id],
                body=None,
                init_pose=None,
                goal_pose=None,
                vertices=vertices,
                coupled_elements=coupled_elements,
                checker="algebraic",
                is_grounded=is_grounded,
            )

            if rod_id in active_rods:
                element.status = ElementStatus.float
                element.assembled_elements = [
                    rod_to_index[coupled_rod]
                    for coupled_rod in sorted(coupled_rods[rod_id] & active_rods)
                ]
            else:
                element.status = ElementStatus.unassembled
                element.assembled_elements = []

            element_objects.append(element)

        return element_objects, rod_to_index, index_to_rod

    def _coupled_rods(self) -> dict[int, set[int]]:
        coupled_rods = {
            rod_id: set()
            for rod_id in self.truss.elements
        }

        for rod_1, rod_2 in self.truss.couplers:
            coupled_rods[rod_1].add(rod_2)
            coupled_rods[rod_2].add(rod_1)
            
        # print (f"Coupled rods: {coupled_rods}")  # Debugging statement
        return coupled_rods
    
    def _statuses_from_nullspace(
        self,
        failure_modes: tuple[np.ndarray, ...],
        matrix_result,
        index_to_rod: dict[int, int],
        active_rods: set[int],
        tolerance: float = 1e-7,
    ) -> dict[int, ElementStatus]:
        statuses = {
            rod_id: ElementStatus.fixed
            for rod_id in active_rods
        }

        if not failure_modes:
            return statuses

        modes = np.column_stack(failure_modes)
        mode_count = modes.shape[1]

        vertex_displacements = modes.reshape(
            (-1, 3, mode_count)
        )

        vertex_motion = np.linalg.norm(
            vertex_displacements,
            axis=(1, 2),
        )

        element_vertices: dict[int, set[int]] = defaultdict(set)

        # Rod endpoint vertices.
        for element_index, vertices in matrix_result.elements_dict.items():
            for vertex in vertices:
                element_vertices[element_index].add(vertex.id)

        # Coupler and orientation vertices that belong to a rod.
        for vertex in matrix_result.vertex_list:
            if vertex.element_index >= 0:
                element_vertices[vertex.element_index].add(vertex.id)

        for element_index, vertex_ids in element_vertices.items():
            rod_id = index_to_rod[element_index]

            if rod_id not in active_rods:
                continue

            maximum_motion = max(
                vertex_motion[vertex_id]
                for vertex_id in vertex_ids
            )

            if maximum_motion > tolerance:
                statuses[rod_id] = ElementStatus.rotate

        return statuses


def _rod_height_key(truss):
    def key(rod_id: int) -> float:
        n1, n2 = truss.elements[rod_id]
        return 0.5 * (truss.nodes[n1][2] + truss.nodes[n2][2])

    return key


def _set_axes_equal(ax, points: np.ndarray) -> None:
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    centers = 0.5 * (mins + maxs)
    radius = 0.5 * max(maxs - mins)
    if radius == 0:
        radius = 1.0

    ax.set_xlim(centers[0] - radius, centers[0] + radius)
    ax.set_ylim(centers[1] - radius, centers[1] + radius)
    ax.set_zlim(centers[2] - radius, centers[2] + radius)
    
def _plot_failure_mode(
    ax,
    result: RigidityResult,
    mode_index: int = 0,
    relative_scale: float = 0.15,
) -> None:
    if not result.failure_modes:
        return

    if mode_index >= len(result.failure_modes):
        raise ValueError(
            f"Failure mode {mode_index} does not exist. "
            f"Available modes: {len(result.failure_modes)}"
        )

    mode = result.failure_modes[mode_index].reshape((-1, 3))

    points = np.asarray(
        [
            vertex.point
            for vertex in result.failure_vertices
        ],
        dtype=float,
    )

    maximum_motion = np.linalg.norm(
        mode,
        axis=1,
    ).max()

    if maximum_motion == 0:
        return

    structure_size = np.linalg.norm(
        points.max(axis=0) - points.min(axis=0)
    )

    if structure_size == 0:
        structure_size = 1.0

    displacements = (
        mode
        / maximum_motion
        * relative_scale
        * structure_size
    )

    # Do not visualize the artificial orientation vertices Q.
    visible_vertex_mask = np.asarray(
        [
            vertex.id not in result.failure_orientation_vertex_ids
            for vertex in result.failure_vertices
        ],
        dtype=bool,
    )

    visible_points = points[visible_vertex_mask]
    visible_displacements = displacements[visible_vertex_mask]

    # Draw displacement arrows only for physical vertices.
    ax.quiver(
        visible_points[:, 0],
        visible_points[:, 1],
        visible_points[:, 2],
        visible_displacements[:, 0],
        visible_displacements[:, 1],
        visible_displacements[:, 2],
        color="cyan",
        linewidth=1.2,
        arrow_length_ratio=0.1,
    )

    # Draw displaced rods as dashed lines.
    for vertex_1, vertex_2 in result.failure_elements.values():
        p1 = np.asarray(vertex_1.point, dtype=float)
        p2 = np.asarray(vertex_2.point, dtype=float)

        displaced_p1 = (
            p1 + displacements[vertex_1.id]
        )
        displaced_p2 = (
            p2 + displacements[vertex_2.id]
        )

        ax.plot(
            [displaced_p1[0], displaced_p2[0]],
            [displaced_p1[1], displaced_p2[1]],
            [displaced_p1[2], displaced_p2[2]],
            color="darkgoldenrod",
            linewidth=2.0,
            linestyle="-",
        )

def _iter_couplers(truss):
    """
    Yield explicit couplers as pairs of rod IDs.

    Supports:
    - truss.couplers = [(0, 1), (1, 2)]
    - truss.couplers = [[0, 1], [1, 2]]
    - truss.couplers = [{"rod_ids": [0, 1]}, ...]
    """

    couplers = getattr(truss, "couplers", [])

    if isinstance(couplers, dict):
        couplers = couplers.keys()

    for coupler in couplers:
        if isinstance(coupler, dict):
            rod_ids = coupler.get("rod_ids")
        else:
            rod_ids = coupler

        if rod_ids is None or len(rod_ids) != 2:
            continue

        yield int(rod_ids[0]), int(rod_ids[1])

def plot_scaffold(
    truss,
    active_rods: set[int],
    removed_rods: set[int],
    supported_rods: set[int],
    result: RigidityResult,
    label_rods: bool = False,
    label_non_fixed_only: bool = False,
    show_nodes: bool = True,
    fast_mode: bool = False,
    save_path: str | None = None,
    show_couplers: bool = True,
    show_failure_mode: bool = False,
    failure_mode_index: int = 0,
    failure_mode_scale: float = 0.15,
    show: bool = True,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from mpl_toolkits.mplot3d.art3d import Line3DCollection


    # floating and rotating is not implemented in current version. Both are combined as "moving" in the plot.
    status_colors = {
        "fixed": "yellowgreen",
        "rotate": "tab:orange",
        # "float": "tab:red",
        "unassembled": "0.65",
    }

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    # For window maximising and placement
    manager = fig.canvas.manager
    window = manager.window

    window.geometry("+0+0")
    window.after(
        100,
        lambda: window.attributes("-zoomed", True),
    )

    # 
    def close_plot(event=None):
        plt.close(fig)

    fig.canvas.mpl_connect(
        "key_press_event",
        lambda event: close_plot()
        if event.key in {"escape", "q"}
        else None,
    )
    
    ax.set_title(
        "Rigidity check scaffold "
        f"({'rigid' if result.is_rigid else 'not rigid'}, "
        f"{result.rank}/{result.dof} fixed rods)"
    )
    
    # zoom by scrolling
    def on_scroll(event):
        if event.inaxes != ax:
            return

        zoom_factor = 0.85 if event.button == "up" else 1.18

        x_min, x_max = ax.get_xlim3d()
        y_min, y_max = ax.get_ylim3d()
        z_min, z_max = ax.get_zlim3d()

        center = np.array([
            0.5 * (x_min + x_max),
            0.5 * (y_min + y_max),
            0.5 * (z_min + z_max),
        ])

        spans = np.array([
            x_max - x_min,
            y_max - y_min,
            z_max - z_min,
        ])

        # Mouse position inside the axes, normalized to [-0.5, 0.5].
        bbox = ax.get_window_extent()

        mouse_x = (event.x - bbox.x0) / bbox.width - 0.5
        mouse_y = (event.y - bbox.y0) / bbox.height - 0.5

        azimuth = np.deg2rad(ax.azim)
        elevation = np.deg2rad(ax.elev)

        # Approximate screen-right direction in world coordinates.
        right = np.array([
            -np.sin(azimuth),
            np.cos(azimuth),
            0.0,
        ])

        # Approximate screen-up direction in world coordinates.
        up = np.array([
            -np.sin(elevation) * np.cos(azimuth),
            -np.sin(elevation) * np.sin(azimuth),
            np.cos(elevation),
        ])

        scene_scale = float(np.max(spans))

        target = (
            center
            + mouse_x * scene_scale * right
            + mouse_y * scene_scale * up
        )

        old_min = np.array([x_min, y_min, z_min])
        old_max = np.array([x_max, y_max, z_max])

        # Scale all limits around the cursor target.
        new_min = target + (old_min - target) * zoom_factor
        new_max = target + (old_max - target) * zoom_factor

        ax.set_xlim3d(new_min[0], new_max[0])
        ax.set_ylim3d(new_min[1], new_max[1])
        ax.set_zlim3d(new_min[2], new_max[2])

        fig.canvas.draw_idle()


    fig.canvas.mpl_connect("scroll_event", on_scroll)

    # Store rods in groups so that each group becomes only one artist.
    fixed_segments = []
    rotating_segments = []
    # floating_segments = []
    supported_segments = []
    removed_segments = []
    inactive_segments = []
    grounded_segments = []

    label_data = []

    for rod_id, (n1, n2) in truss.elements.items():
        p1 = np.asarray(truss.nodes[n1], dtype=float)
        p2 = np.asarray(truss.nodes[n2], dtype=float)
        segment = [p1, p2]

        if rod_id in removed_rods:
            removed_segments.append(segment)

        elif rod_id in supported_rods:
            supported_segments.append(segment)

        elif rod_id in truss.grounded_rods:
            grounded_segments.append(segment)

        elif rod_id not in active_rods:
            inactive_segments.append(segment)

        else:
            status = result.statuses.get(
                rod_id,
                ElementStatus.unassembled,
            )

            if status == ElementStatus.fixed:
                fixed_segments.append(segment)
            elif status == ElementStatus.rotate:
                rotating_segments.append(segment)
            # elif status == ElementStatus.float:
            #     floating_segments.append(segment)
            else:
                inactive_segments.append(segment)

        should_label = label_rods

        if label_non_fixed_only:
            status = result.statuses.get(
                rod_id,
                ElementStatus.unassembled,
            )
            should_label = (
                rod_id in supported_rods
                or status != ElementStatus.fixed
            )

        if should_label:
            midpoint = 0.5 * (p1 + p2)
            label_data.append((midpoint, rod_id))

    def add_collection(
        segments,
        color,
        linewidth,
        linestyle="-",
    ):
        if not segments:
            return

        collection = Line3DCollection(
            segments,
            colors=color,
            linewidths=linewidth,
            linestyles=linestyle,
        )
        ax.add_collection3d(collection)

    # One artist per category instead of one artist per rod.
        
    add_collection(
        grounded_segments,
        color="cornflowerblue",
        linewidth=2.5,
    )
    
    add_collection(
        fixed_segments,
        color=status_colors["fixed"],
        linewidth=2.0,
    )

    add_collection(
        rotating_segments,
        color=status_colors["rotate"],
        linewidth=2.5,
    )

    # add_collection(
    #     floating_segments,
    #     color=status_colors["float"],
    #     linewidth=2.5,
    # )

    add_collection(
        supported_segments,
        color="magenta",
        linewidth=3.5,
    )

    # Dashed rendering costs more while moving.
    # Fast mode uses a normal solid line.
    add_collection(
        removed_segments,
        color="0.78",
        linewidth=1.0,
        linestyle="-" if fast_mode else "--",
    )

    add_collection(
        inactive_segments,
        color="0.7",
        linewidth=1.0,
    )

    node_points = np.asarray(
        [
            truss.nodes[node_id]
            for node_id in sorted(truss.nodes)
        ],
        dtype=float,
    )

    if show_nodes:
        ax.scatter(
            node_points[:, 0],
            node_points[:, 1],
            node_points[:, 2],
            color="black",
            s=6 if fast_mode else 12,
            depthshade=False,
        )

    # Text labels are expensive in Matplotlib 3D.
    # They are created only when explicitly requested.
    for midpoint, rod_id in label_data:
        ax.text(
            midpoint[0],
            midpoint[1],
            midpoint[2],
            str(rod_id),
            fontsize=7,
        )
        
    coupler_segments = []

    if show_couplers:
        for rod_id_1, rod_id_2 in _iter_couplers(truss):
            if (
                rod_id_1 not in active_rods
                or rod_id_2 not in active_rods
            ):
                continue

            if (
                rod_id_1 not in truss.elements
                or rod_id_2 not in truss.elements
            ):
                continue

            n11, n12 = truss.elements[rod_id_1]
            n21, n22 = truss.elements[rod_id_2]

            rod_1_segment = [
                np.asarray(truss.nodes[n11], dtype=float),
                np.asarray(truss.nodes[n12], dtype=float),
            ]

            rod_2_segment = [
                np.asarray(truss.nodes[n21], dtype=float),
                np.asarray(truss.nodes[n22], dtype=float),
            ]

            try:
                point_1, point_2 = closest_points_between_segments(
                    rod_1_segment,
                    rod_2_segment,
                )
            except ValueError:
                continue

            coupler_segments.append(
                [
                    np.asarray(point_1, dtype=float),
                    np.asarray(point_2, dtype=float),
                ]
            )

    if coupler_segments:
        coupler_collection = Line3DCollection(
            coupler_segments,
            colors="darkviolet",
            linewidths=3.0,
            linestyles="-",
        )
        ax.add_collection3d(coupler_collection)
        
    if show_failure_mode:
        if not result.failure_modes:
            print("No failure modes are available to plot.")
        else:
            _plot_failure_mode(
                ax=ax,
                result=result,
                mode_index=failure_mode_index,
                relative_scale=failure_mode_scale,
            )

    _set_axes_equal(ax, node_points)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    if fast_mode:
        # Grid panes add redraw work and visual clutter.
        ax.grid(False)

        ax.xaxis.pane.set_visible(False)
        ax.yaxis.pane.set_visible(False)
        ax.zaxis.pane.set_visible(False)

    legend_items = [
        Line2D(
            [0],
            [0],
            color="yellowgreen",
            lw=2,
            label="fixed",
        ),
        Line2D(
            [0],
            [0],
            color="tab:orange",
            lw=2,
            label="moving", # "moving" combines both "rotate" and "float" in the current implementation
        ),
        # Line2D(
        #     [0],
        #     [0],
        #     color="tab:red",
        #     lw=2,
        #     label="float",
        # ),
        Line2D(
            [0],
            [0],
            color="magenta",
            lw=3,
            label="supported",
        ),
        Line2D(
            [0],
            [0],
            color="0.78",
            lw=1,
            ls="-" if fast_mode else "--",
            label="removed",
        ),
        Line2D(
            [0],
            [0],
            color="cornflowerblue",
            lw=2,
            label="grounded rod",
        ),
        Line2D(
            [0],
            [0],
            color="darkviolet",
            lw=2,
            label="coupler",
        ),
        Line2D(
            [0],
            [0],
            color="darkgoldenrod",
            lw=2,
            ls="-",
            label="failure mode",
        ),
    ]

    ax.legend(
        handles=legend_items,
        loc="upper right",
    )

    # tight_layout is useful for saved figures but unnecessary
    # for a fast interactive window.
    if not fast_mode or save_path:
        fig.tight_layout()

    if save_path:
        fig.savefig(
            save_path,
            dpi=180,
            bbox_inches="tight",
        )
        print(f"plot saved to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the rigiditycheck.py-backed rigidity check for a truss JSON file."
    )
    parser.add_argument(
        "json_path",
        nargs="?",
        default="JSON/own_examples/260804_FoC_demo.json",
        # default="JSON/own_examples/260724_stability_ini.json",
        # default="JSON/own_examples/diy_proper_full.json",
        help="Path to the truss JSON file.",
    )
    parser.add_argument(
        "--remove",
        type=int,
        nargs="*",
        default=[],
        help="Rod ids to remove before checking.",
    )
    parser.add_argument(
        "--supported",
        type=int,
        nargs="*",
        default=[],
        help="Rod ids treated as grounded/supported ElementObjects.",
    )
    parser.add_argument(
        "--suggest-supports",
        type=int,
        default=2,
        help="Maximum number of additional support rods to suggest.",
    )
    parser.add_argument(
        "--show-statuses",
        action="store_true",
        help="Print every active rod's ElementStatus.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Show a matplotlib 3D plot of the checked scaffold.",
    )
    parser.add_argument(
        "--save-plot",
        help="Save the scaffold plot to an image file instead of only displaying it.",
    )
    parser.add_argument(
        "--label-rods",
        action="store_true",
        help="Label rods with their element ids in the plot.",
    )
    parser.add_argument(
        "--label-non-fixed-only",
        action="store_true",
        help="Only label rods that are not fixed or are externally supported.",
    )

    parser.add_argument(
        "--hide-nodes",
        action="store_true",
        help="Do not draw scaffold node markers.",
    )

    parser.add_argument(
        "--fast-plot",
        action="store_true",
        help="Use simplified rendering for smoother interactive movement.",
    )
    parser.add_argument(
        "--hide-couplers",
        action="store_true",
        help="Do not display coupler segments.",
    )
    
    parser.add_argument(
        "--failure-mode",
        type=int,
        metavar="INDEX",
        help=(
            "Plot the selected nullspace failure mode. "
            "Failure modes are indexed from 0."
        ),
    )

    parser.add_argument(
        "--failure-scale",
        type=float,
        default=0.15,
        help="Relative visual scale of the plotted failure mode.",
    )
    args = parser.parse_args()

    from truss import Truss

    truss = Truss.from_json(args.json_path)
    
    checker = TrussRigidityChecker(truss)

    start_time = time.time()
    active_rods = set(truss.elements) - set(args.remove)
    supported_rods = set(args.supported) & active_rods
    try:
        result = checker.check(
            active_rods,
            supported_rods=supported_rods,
        )

    except ValueError as error:
        print()
        print("Scaffold validation failed")
        print("--------------------------")
        print(error)

        if args.plot or args.save_plot:
            empty_result = RigidityResult(
                is_rigid=False,
                rank=0,
                dof=len(active_rods),
                rows=0,
                statuses={
                    rod_id: ElementStatus.unassembled
                    for rod_id in active_rods
                },
                failure_modes=(),
                failure_vertices=(),
                failure_elements={},
                failure_orientation_vertex_ids=frozenset(),
            )

            plot_scaffold(
                truss=truss,
                active_rods=active_rods,
                removed_rods=set(args.remove),
                supported_rods=supported_rods,
                result=empty_result,
                label_rods=args.label_rods,
                label_non_fixed_only=args.label_non_fixed_only,
                show_nodes=not args.hide_nodes,
                show_couplers=not args.hide_couplers,
                fast_mode=args.fast_plot,
                save_path=args.save_plot,
                show_failure_mode=False,
                show=args.plot,
            )

        sys.exit(1)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Rigidity check completed in {elapsed_time:.4f} seconds.")
    
    fixed_rods = sum(
        status == ElementStatus.fixed
        for status in result.statuses.values()
    )

    print(f"JSON: {args.json_path}")
    print(f"active rods: {len(active_rods)}")
    print(f"removed rods: {sorted(args.remove)}")
    print(f"supported rods: {sorted(supported_rods)}")
    print(f"is rigid: {result.is_rigid}")
    print(f"fixed rods: {fixed_rods}/{len(result.statuses)}")
    print(f"non-fixed rods: {result.nullity}")
    print(f"failure modes: {len(result.failure_modes)}")

    if args.show_statuses:
        for rod_id, status_name in result.status_names.items():
            print(f"rod {rod_id}: {status_name}")

    # if not result.is_rigid and args.suggest_supports > 0:
    #     suggestions = checker.choose_support_targets(
    #         active_rods,
    #         already_supported=supported_rods,
    #         max_targets=args.suggest_supports,
    #         key=_rod_height_key(truss),
    #     )
    #     if suggestions:
    #         supported_result = checker.check(
    #             active_rods,
    #             supported_rods=supported_rods | set(suggestions),
    #         )
            
    #         fixed_rods = sum(
    #             status == ElementStatus.fixed
    #             for status in result.statuses.values()
    #         )
             
    #         print(f"suggested supports: {suggestions}")
    #         print(
    #             "with suggested supports: "
    #             f"{supported_result.is_rigid}, "
    #             f"fixed rods {fixed_rods}/{len(supported_result.statuses)}, "
    #         )
    #     else:
    #         print("suggested supports: none found")

    if args.plot or args.save_plot:
        plot_scaffold(
            truss=truss,
            active_rods=active_rods,
            removed_rods=set(args.remove),
            supported_rods=supported_rods,
            result=result,
            label_rods=args.label_rods,
            label_non_fixed_only=args.label_non_fixed_only,
            show_nodes=not args.hide_nodes,
            show_couplers=not args.hide_couplers,
            fast_mode=args.fast_plot,
            save_path=args.save_plot,
            show_failure_mode=args.failure_mode is not None,
            failure_mode_index=(
                args.failure_mode
                if args.failure_mode is not None
                else 0
            ),
            failure_mode_scale=args.failure_scale,
            show=args.plot,
        )


if __name__ == "__main__":
    main()
