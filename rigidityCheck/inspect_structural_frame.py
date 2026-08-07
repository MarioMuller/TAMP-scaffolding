import argparse
import pickle
from pathlib import Path

from truss_rigidity import (
    TrussRigidityChecker,
    plot_scaffold,
)


def list_frames(steps):
    print(
        "Frame | removed rod | active before -> after | "
        "supports after | rank/dof"
    )
    print("-" * 80)

    for frame, step in enumerate(steps):
        supports_after = sorted(
            set(step.supports_after.values())
        )

        print(
            f"{frame:5d} | "
            f"{step.rod_id:11d} | "
            f"{len(step.rods_before):6d} -> "
            f"{len(step.rods_after):5d} | "
            f"{str(supports_after):14s} | "
            f"{step.rank_after}/{step.dof_after}"
        )


def validate_rods(name, rods, all_rods):
    unknown = set(rods) - all_rods

    if unknown:
        raise ValueError(
            f"{name} contains unknown rod IDs: {sorted(unknown)}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Inspect one configuration from a structural capture."
    )

    parser.add_argument(
        "capture_path",
        nargs="?",
        default="debug_captures/structural_sequence.pkl",
        help="Pickle file created by main.py.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all recorded frames.",
    )
    parser.add_argument(
        "--frame",
        type=int,
        help="Zero-based StructuralRemovalStep index.",
    )
    parser.add_argument(
        "--state",
        choices=("before", "after"),
        default="after",
        help="Plot the configuration before or after the removal.",
    )

    # Completely replace the saved sets.
    parser.add_argument(
        "--active",
        type=int,
        nargs="*",
        default=None,
        help="Replace the complete active-rod set.",
    )
    parser.add_argument(
        "--supported",
        type=int,
        nargs="*",
        default=None,
        help="Replace the complete supported-rod set.",
    )

    # Modify individual rods relative to the saved frame.
    parser.add_argument(
        "--add-rods",
        type=int,
        nargs="*",
        default=[],
    )
    parser.add_argument(
        "--remove-rods",
        type=int,
        nargs="*",
        default=[],
    )
    parser.add_argument(
        "--add-supports",
        type=int,
        nargs="*",
        default=[],
    )
    parser.add_argument(
        "--remove-supports",
        type=int,
        nargs="*",
        default=[],
    )

    parser.add_argument(
        "--failure-mode",
        type=int,
        default=None,
        help="Failure-mode index to plot, for example 0.",
    )
    parser.add_argument(
        "--failure-scale",
        type=float,
        default=0.15,
    )
    parser.add_argument(
        "--label-rods",
        action="store_true",
    )
    parser.add_argument(
        "--save-plot",
        help="Optional output image path.",
    )

    args = parser.parse_args()

    capture_path = Path(args.capture_path)

    # Only load pickle files you created yourself.
    with capture_path.open("rb") as file:
        capture = pickle.load(file)

    truss = capture["truss"]
    steps = capture["structural_steps"]

    if not steps:
        raise RuntimeError("The capture contains no structural frames.")

    if args.list:
        list_frames(steps)
        return

    list_frames(steps)

    frame = args.frame

    if frame is None:
        frame = int(input("\nSelect frame: "))

    if not 0 <= frame < len(steps):
        raise IndexError(
            f"Frame must be between 0 and {len(steps) - 1}."
        )

    step = steps[frame]

    if args.state == "before":
        active_rods = set(step.rods_before)
        supported_rods = set(step.supports_before.values())
    else:
        active_rods = set(step.rods_after)
        supported_rods = set(step.supports_after.values())

    all_rods = set(truss.elements)

    # Complete replacements.
    if args.active is not None:
        active_rods = set(args.active)

    if args.supported is not None:
        supported_rods = set(args.supported)

    # Individual modifications.
    active_rods.update(args.add_rods)
    active_rods.difference_update(args.remove_rods)

    supported_rods.update(args.add_supports)
    supported_rods.difference_update(args.remove_supports)

    validate_rods("Active rods", active_rods, all_rods)
    validate_rods("Supported rods", supported_rods, all_rods)

    inactive_supports = supported_rods - active_rods

    if inactive_supports:
        raise ValueError(
            "These supported rods are not active: "
            f"{sorted(inactive_supports)}"
        )

    removed_rods = all_rods - active_rods

    checker = TrussRigidityChecker(truss)
    result = checker.check(
        active_rods,
        supported_rods=supported_rods,
    )

    print()
    print(f"Frame:          {frame} ({args.state})")
    print(f"Removal step:   rod {step.rod_id}")
    print(f"Active rods:    {sorted(active_rods)}")
    print(f"Removed rods:   {sorted(removed_rods)}")
    print(f"Supported rods: {sorted(supported_rods)}")
    print(f"Rigid:          {result.is_rigid}")
    print(f"Rank/DOF:       {result.rank}/{result.dof}")
    print(f"Failure modes:  {len(result.failure_modes)}")

    if (
        args.failure_mode is not None
        and args.failure_mode >= len(result.failure_modes)
    ):
        raise ValueError(
            f"Failure mode {args.failure_mode} does not exist. "
            f"Available modes: {len(result.failure_modes)}"
        )

    plot_scaffold(
        truss=truss,
        active_rods=active_rods,
        removed_rods=removed_rods,
        supported_rods=supported_rods,
        result=result,
        label_rods=args.label_rods,
        show_failure_mode=args.failure_mode is not None,
        failure_mode_index=(
            args.failure_mode
            if args.failure_mode is not None
            else 0
        ),
        failure_mode_scale=args.failure_scale,
        save_path=args.save_plot,
        show=True,
    )


if __name__ == "__main__":
    main()