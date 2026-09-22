# used for debugging and replaying/video generating of the structural 
# assembly sequence found by backward search

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

MPLCONFIGDIR = Path("/tmp/tamp_scaffolding_matplotlib")
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))

from backward_search import AssemblyPlanner
from experiment_metrics import support_summary
from experiments.structural_experiment_utils import (
    load_filtered_truss,
    parse_rod_ids,
)
from rigidityCheck.structural_replay import display_structural_assembly

DEFAULT_TRUSS_PATH = (PROJECT_ROOT/"JSON/own_examples/260804_FoC_demo.json")

SUPPORT_CSV_FIELDS = [
    "truss",
    "strategy_name",
    "support_target_order",
    "require_connected_supports",
    "max_supports",
    "seed",
    "runtime_s",
    "runtime_ns",
    "elapsed_s",
    "removal_steps",
    "assembly_sequence",
    "search_backtracks",
    "peak_supports",
    "support_steps",
    "support_moves",
    "supported_rod_steps",
    "support_assignment_episodes",
    "supported_rod_episodes",
    "support_additions",
    "support_releases",
]

# allows to choose parameters for the backward search and the structural replay in command line
def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run rigidity-only backward search and replay/export the "
            "resulting structural assembly sequence."
        )
    )
    parser.add_argument("--truss", type=Path, default=DEFAULT_TRUSS_PATH)
    parser.add_argument(
        "--include-rods",
        default=None,
        help=(
            "Optional comma-separated ids or JSON list, for example "
            "'0,1,2' or '[0, 1, 2]'."
        ),
    )
    parser.add_argument("--max-supports", type=int, default=2)
    parser.add_argument(
        "--strategy-name",
        choices=AssemblyPlanner.STRATEGY_NAMES,
        default="fast_reduce_support",
    )
    parser.add_argument(
        "--support-target-order",
        choices=AssemblyPlanner.SUPPORT_TARGET_ORDERS,
        default="lowest_first",
        help=(
            "Ordering used when choosing structural support rods; distance "
            "modes use Euclidean distance between rod centers."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-runtime", type=float, default=1800.0)
    parser.add_argument(
        "--capture-key",
        default="v",
        help="Set to an empty string to disable the debug-capture hotkey.",
    )
    parser.add_argument("--support-csv", type=Path, default=None)
    parser.add_argument(
        "--skip-support-csv",
        action="store_true",
        help="Do not write the structural support summary CSV.",
    )
    parser.add_argument("--scale", type=float, default=0.001)
    parser.add_argument("--hide-rod-labels", action="store_true")
    parser.add_argument("--video-path", type=Path, default=None)
    parser.add_argument(
        "--skip-video",
        action="store_true",
        help="Do not export the structural replay video.",
    )
    parser.add_argument("--seconds-per-step", type=float, default=0.8)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Export/save without opening the interactive replay window.",
    )
    return parser.parse_args()


def default_output_path(args, folder, suffix):
    stem = Path(args.truss).stem
    return PROJECT_ROOT / folder / f"{stem}_{suffix}"


def save_support_summary(
    path,
    args,
    searcher,
    removal_sequence,
    runtime_ns,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    runtime_s = runtime_ns / 1_000_000_000
    removal_sequence = list(removal_sequence)
    assembly_sequence = list(reversed(removal_sequence))

    row = {
        "truss": str(args.truss),
        "strategy_name": args.strategy_name,
        "support_target_order": args.support_target_order,
        "require_connected_supports": True,
        "max_supports": args.max_supports,
        "seed": args.seed,
        "runtime_s": f"{runtime_s:.9f}",
        "runtime_ns": runtime_ns,
        "elapsed_s": f"{runtime_s:.9f}",
        "removal_steps": len(removal_sequence),
        "assembly_sequence": json.dumps(assembly_sequence),
        "search_backtracks": getattr(searcher, "search_backtracks", 0),
    }
    row.update(support_summary(searcher.final_node.structural_steps))

    with path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=SUPPORT_CSV_FIELDS)
        writer.writeheader()
        writer.writerow(row)


def main():
    args = parse_args()
    support_csv_path = (
        args.support_csv
        or default_output_path(args, "experiments/results/csv", "support_summary.csv")
    )
    video_path = (
        None
        if args.skip_video
        else (
            args.video_path
            or default_output_path(args, "experiments/results/videos", "structural_assembly.mp4")
        )
    )

    included_rods = parse_rod_ids(args.include_rods)
    truss = load_filtered_truss(args.truss, included_rods)

    searcher = AssemblyPlanner(
        truss=truss,
        builder=None,
        max_supports=args.max_supports,
        strategy_name=args.strategy_name,
        random_seed=args.seed,
        support_target_order=args.support_target_order,
    )

    start_ns = time.perf_counter_ns()
    removal_sequence = searcher.backward_search(
        capture_key=args.capture_key or None,
        max_runtime=args.max_runtime,
    )
    runtime_ns = time.perf_counter_ns() - start_ns
    runtime_s = runtime_ns / 1_000_000_000

    print(f"Backward search took {runtime_s:.2f} seconds.")

    if removal_sequence is None:
        raise RuntimeError("No structurally feasible sequence found.")

    assembly_sequence = list(reversed(removal_sequence))

    if not args.skip_support_csv:
        save_support_summary(
            support_csv_path,
            args,
            searcher,
            removal_sequence,
            runtime_ns,
        )
        print(f"Saved support summary to: {support_csv_path}")

    print("Removal:", list(removal_sequence))
    print("Assembly:", assembly_sequence)

    display_structural_assembly(
        truss=truss,
        removal_steps=searcher.final_node.structural_steps,
        scale=args.scale,
        label_rods= args.hide_rod_labels,
        video_path=video_path,
        seconds_per_step=args.seconds_per_step,
        fps=args.fps,
        show=not args.no_display,
    )


if __name__ == "__main__":
    main()
