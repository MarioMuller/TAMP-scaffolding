from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from .Datastructures import ElementObject, ElementStatus
    from .rigiditycheck import AlgebraicChecker
except ImportError:
    from Datastructures import ElementObject, ElementStatus
    from rigiditycheck import AlgebraicChecker


@dataclass(frozen=True)
class RigidityResult:
    is_rigid: bool
    rank: int
    dof: int
    rows: int
    statuses: dict[int, ElementStatus] = field(default_factory=dict)

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
        active = set(active_rods)
        if not active:
            return RigidityResult(
                is_rigid=True,
                rank=0,
                dof=0,
                rows=0,
                statuses={},
            )

        element_objects, rod_to_index, index_to_rod = self._build_element_objects(
            active,
            set(supported_rods or ()),
        )
        assembled_indices = [rod_to_index[rod_id] for rod_id in sorted(active)]

        statuses = {}
        fixed_count = 0

        for rod_id in sorted(active):
            index = rod_to_index[rod_id]
            status = AlgebraicChecker.Check(
                index,
                assembled_indices.copy(),
                element_objects,
            )
            statuses[rod_id] = status
            if status == ElementStatus.fixed:
                fixed_count += 1

        return RigidityResult(
            is_rigid=fixed_count == len(active),
            rank=fixed_count,
            dof=len(active),
            rows=0,
            statuses=statuses,
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

        if self.is_rigid(active, supported_rods=supported):
            return chosen

        while len(chosen) < max_targets:
            base = self.check(active, supported_rods=supported)
            best_rod = None
            best_result = base

            candidates = [rod for rod in active if rod not in supported]
            if key is not None:
                candidates.sort(key=key, reverse=True)

            for rod in candidates:
                result = self.check(active, supported_rods=supported | {rod})
                if result.rank > best_result.rank:
                    best_rod = rod
                    best_result = result

            if best_rod is None or best_result.rank <= base.rank:
                break

            chosen.append(best_rod)
            supported.add(best_rod)

            if best_result.is_rigid:
                break

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

        coupled_rods = self._coupled_rods_by_shared_node()
        element_objects = []

        for rod_id in rod_ids:
            n1, n2 = self.truss.elements[rod_id]
            vertices = [
                np.asarray(self.truss.nodes[n1], dtype=float),
                np.asarray(self.truss.nodes[n2], dtype=float),
            ]
            is_grounded = (
                n1 in self.truss.grounded_nodes
                or n2 in self.truss.grounded_nodes
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

    def _coupled_rods_by_shared_node(self) -> dict[int, set[int]]:
        node_to_rods = defaultdict(set)
        for rod_id, (n1, n2) in self.truss.elements.items():
            node_to_rods[n1].add(rod_id)
            node_to_rods[n2].add(rod_id)

        coupled_rods = {
            rod_id: set()
            for rod_id in self.truss.elements
        }

        for rods in node_to_rods.values():
            for rod_id in rods:
                coupled_rods[rod_id].update(rods - {rod_id})

        return coupled_rods


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


def plot_scaffold(
    truss,
    active_rods: set[int],
    removed_rods: set[int],
    supported_rods: set[int],
    result: RigidityResult,
    label_rods: bool = True,
    save_path: str | None = None,
    show: bool = True,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    status_colors = {
        "fixed": "tab:green",
        "rotate": "tab:orange",
        "float": "tab:red",
        "unassembled": "0.65",
    }

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title(
        "Rigidity check scaffold "
        f"({'rigid' if result.is_rigid else 'not rigid'}, "
        f"{result.rank}/{result.dof} fixed rods)"
    )

    for rod_id, (n1, n2) in truss.elements.items():
        p1 = np.asarray(truss.nodes[n1], dtype=float)
        p2 = np.asarray(truss.nodes[n2], dtype=float)
        xs, ys, zs = zip(p1, p2)

        if rod_id in removed_rods:
            color = "0.82"
            linewidth = 1.0
            linestyle = "--"
            alpha = 0.45
        elif rod_id in supported_rods:
            color = "magenta"
            linewidth = 3.0
            linestyle = "-"
            alpha = 1.0
        else:
            status = result.statuses.get(rod_id, ElementStatus.unassembled).name
            color = status_colors[status]
            linewidth = 2.0
            linestyle = "-"
            alpha = 1.0 if rod_id in active_rods else 0.35

        ax.plot(xs, ys, zs, color=color, linewidth=linewidth, linestyle=linestyle, alpha=alpha)

        if label_rods:
            midpoint = 0.5 * (p1 + p2)
            ax.text(midpoint[0], midpoint[1], midpoint[2], str(rod_id), fontsize=8)

    node_points = np.asarray(
        [truss.nodes[node_id] for node_id in sorted(truss.nodes)],
        dtype=float,
    )
    ax.scatter(
        node_points[:, 0],
        node_points[:, 1],
        node_points[:, 2],
        color="black",
        s=12,
        alpha=0.65,
    )

    grounded_points = np.asarray(
        [truss.nodes[node_id] for node_id in sorted(truss.grounded_nodes)],
        dtype=float,
    )
    if len(grounded_points) > 0:
        ax.scatter(
            grounded_points[:, 0],
            grounded_points[:, 1],
            grounded_points[:, 2],
            color="black",
            marker="s",
            s=45,
            label="grounded nodes",
        )

    _set_axes_equal(ax, node_points)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    legend_items = [
        Line2D([0], [0], color="tab:green", lw=2, label="fixed"),
        Line2D([0], [0], color="tab:orange", lw=2, label="rotate"),
        Line2D([0], [0], color="tab:red", lw=2, label="float"),
        Line2D([0], [0], color="magenta", lw=3, label="supported"),
        Line2D([0], [0], color="0.82", lw=1, ls="--", label="removed"),
        Line2D([0], [0], marker="s", color="black", lw=0, label="grounded node"),
    ]
    ax.legend(handles=legend_items, loc="upper right")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=180)
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
        # default="JSON/scaffold_two_floors.json",
        default="JSON/scaffold_test.json",
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
    args = parser.parse_args()

    from truss import Truss

    truss = Truss.from_json(args.json_path)
    checker = TrussRigidityChecker(truss)

    active_rods = set(truss.elements) - set(args.remove)
    supported_rods = set(args.supported) & active_rods
    result = checker.check(active_rods, supported_rods=supported_rods)

    print(f"JSON: {args.json_path}")
    print(f"active rods: {len(active_rods)}")
    print(f"removed rods: {sorted(args.remove)}")
    print(f"supported rods: {sorted(supported_rods)}")
    print(f"is rigid: {result.is_rigid}")
    print(f"fixed rods: {result.rank}/{result.dof}")
    print(f"non-fixed rods: {result.nullity}")

    if args.show_statuses:
        for rod_id, status_name in result.status_names.items():
            print(f"rod {rod_id}: {status_name}")

    if not result.is_rigid and args.suggest_supports > 0:
        suggestions = checker.choose_support_targets(
            active_rods,
            already_supported=supported_rods,
            max_targets=args.suggest_supports,
            key=_rod_height_key(truss),
        )
        if suggestions:
            supported_result = checker.check(
                active_rods,
                supported_rods=supported_rods | set(suggestions),
            )
            print(f"suggested supports: {suggestions}")
            print(
                "with suggested supports: "
                f"{supported_result.is_rigid}, "
                f"fixed rods {supported_result.rank}/{supported_result.dof}"
            )
        else:
            print("suggested supports: none found")

    if args.plot or args.save_plot:
        plot_scaffold(
            truss=truss,
            active_rods=active_rods,
            removed_rods=set(args.remove),
            supported_rods=supported_rods,
            result=result,
            label_rods=args.label_rods,
            save_path=args.save_plot,
            show=args.plot,
        )


if __name__ == "__main__":
    main()
