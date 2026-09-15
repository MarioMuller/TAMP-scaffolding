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
from experiment_metrics import structural_summary
from rigidityCheck.structural_replay import display_structural_assembly
from truss import Truss


DEFAULT_TRUSS_PATH = (
    PROJECT_ROOT
    / "JSON"
    / "own_examples"
    / "260804_FoC_demo.json"
)
DEFAULT_OUTPUT_CSV = (
    PROJECT_ROOT
    / "experiments"
    / "results"
    / "rigidity_check_scaling.csv"
)


FIELDNAMES = [
    "repetition",
    "seed",
    "run_index",
    "removed_prefix_count",
    "included_rod_count",
    "excluded_rods",
    "included_rods",
    "default_order",
    "success",
    "elapsed_ns",
    "elapsed_s",
    "removal_sequence",
    "assembly_sequence",
    "search_stop_reason",
    "search_expansions",
    "search_attempted_transitions",
    "search_enqueued_candidates",
    "rigidity_check_calls",
    "rigidity_cache_hits",
    "rigidity_cache_misses",
    "rigidity_cached_entries",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Measure rigidity-only backward-search scaling while removing "
            "prefixes of each repetition's full-truss removal order."
        )
    )
    parser.add_argument("--repetitions", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-supports", type=int, default=2)
    parser.add_argument("--strategy-name", default="reduced_supports")
    parser.add_argument("--max-runtime", type=float, default=1800.0)
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Show an interactive structural assembly replay after each successful run.",
    )
    parser.add_argument("--visualization-scale", type=float, default=0.0011)
    parser.add_argument(
        "--visualization-seconds-per-step",
        type=float,
        default=0.8,
    )
    parser.add_argument(
        "--hide-rod-labels",
        action="store_true",
        help="Hide rod labels in the structural replay.",
    )
    parser.add_argument(
        "--truss",
        type=Path,
        default=DEFAULT_TRUSS_PATH,
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_OUTPUT_CSV,
    )
    return parser.parse_args()


def require_positive_repetitions(repetitions):
    if repetitions <= 0:
        raise ValueError("repetitions must be positive.")


def load_filtered_truss(truss_path, included_rods=None):
    truss = Truss.from_json(truss_path)

    if included_rods is None:
        return truss

    included_rods = set(included_rods)
    unknown_rods = included_rods - set(truss.elements)
    if unknown_rods:
        raise ValueError(
            f"Included rods do not exist: {sorted(unknown_rods)}"
        )

    truss.elements = {
        rod_id: endpoints
        for rod_id, endpoints in truss.elements.items()
        if rod_id in included_rods
    }
    truss.grounded_rods &= included_rods
    truss.couplers = {
        (rod_1, rod_2)
        for rod_1, rod_2 in truss.couplers
        if rod_1 in included_rods and rod_2 in included_rods
    }

    return truss


def as_json_list(values):
    return json.dumps([int(value) for value in values])


def run_backward_search(args, included_rods, seed):
    truss = load_filtered_truss(args.truss, included_rods)

    searcher = AssemblyPlanner(
        truss=truss,
        builder=None,
        max_supports=args.max_supports,
        strategy_name=args.strategy_name,
        random_seed=seed,
    )

    start_ns = time.perf_counter_ns()
    removal_sequence = searcher.backward_search(
        capture_key=None,
        max_runtime=args.max_runtime,
    )
    elapsed_ns = time.perf_counter_ns() - start_ns

    return searcher, removal_sequence, elapsed_ns


def make_row(
    *,
    repetition,
    seed,
    run_index,
    removed_prefix_count,
    included_rods,
    excluded_rods,
    default_order,
    searcher,
    removal_sequence,
    elapsed_ns,
):
    success = removal_sequence is not None
    removal_sequence = list(removal_sequence or [])
    assembly_sequence = list(reversed(removal_sequence))

    row = {
        "repetition": repetition,
        "seed": seed,
        "run_index": run_index,
        "removed_prefix_count": removed_prefix_count,
        "included_rod_count": len(included_rods),
        "excluded_rods": as_json_list(sorted(excluded_rods)),
        "included_rods": as_json_list(sorted(included_rods)),
        "default_order": as_json_list(default_order),
        "success": success,
        "elapsed_ns": elapsed_ns,
        "elapsed_s": f"{elapsed_ns / 1_000_000_000:.9f}",
        "removal_sequence": as_json_list(removal_sequence),
        "assembly_sequence": as_json_list(assembly_sequence),
    }
    row.update(structural_summary(searcher))
    return row


def write_row(writer, csv_file, row):
    writer.writerow(row)
    csv_file.flush()


def visualize_run(args, run_index, searcher, removal_sequence):
    if not args.visualize or removal_sequence is None:
        return

    print(
        f"Opening visualization for run {run_index}. "
        "Close the window to continue."
    )

    display_structural_assembly(
        truss=searcher.truss,
        removal_steps=searcher.final_node.structural_steps,
        scale=args.visualization_scale,
        label_rods=not args.hide_rod_labels,
        video_path=None,
        seconds_per_step=args.visualization_seconds_per_step,
        fps=30,
    )


def main():
    args = parse_args()
    require_positive_repetitions(args.repetitions)

    all_rods = sorted(load_filtered_truss(args.truss).elements)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)

    with args.output_csv.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=FIELDNAMES)
        writer.writeheader()

        for repetition_index in range(args.repetitions):
            repetition = repetition_index + 1
            seed = args.seed + repetition_index

            print(
                "\n"
                + "=" * 70
                + f"\nRepetition {repetition}/{args.repetitions}"
                + f"\nSeed: {seed}"
                + "\n"
                + "=" * 70
            )

            searcher, removal_sequence, elapsed_ns = run_backward_search(
                args=args,
                included_rods=all_rods,
                seed=seed,
            )

            if removal_sequence is None:
                row = make_row(
                    repetition=repetition,
                    seed=seed,
                    run_index=1,
                    removed_prefix_count=0,
                    included_rods=all_rods,
                    excluded_rods=[],
                    default_order=[],
                    searcher=searcher,
                    removal_sequence=removal_sequence,
                    elapsed_ns=elapsed_ns,
                )
                write_row(writer, csv_file, row)
                raise RuntimeError(
                    "The full-truss run failed, so no default order could "
                    f"be generated for repetition {repetition}."
                )

            default_order = list(removal_sequence)

            row = make_row(
                repetition=repetition,
                seed=seed,
                run_index=1,
                removed_prefix_count=0,
                included_rods=all_rods,
                excluded_rods=[],
                default_order=default_order,
                searcher=searcher,
                removal_sequence=removal_sequence,
                elapsed_ns=elapsed_ns,
            )
            write_row(writer, csv_file, row)
            print(
                f"Run 1/{len(default_order)}: "
                f"{len(all_rods)} rods, {elapsed_ns / 1_000_000_000:.3f}s"
            )
            visualize_run(
                args=args,
                run_index=1,
                searcher=searcher,
                removal_sequence=removal_sequence,
            )

            for removed_prefix_count in range(1, len(default_order)):
                excluded_rods = default_order[:removed_prefix_count]
                included_rods = [
                    rod_id
                    for rod_id in all_rods
                    if rod_id not in excluded_rods
                ]

                searcher, removal_sequence, elapsed_ns = run_backward_search(
                    args=args,
                    included_rods=included_rods,
                    seed=seed,
                )

                row = make_row(
                    repetition=repetition,
                    seed=seed,
                    run_index=removed_prefix_count + 1,
                    removed_prefix_count=removed_prefix_count,
                    included_rods=included_rods,
                    excluded_rods=excluded_rods,
                    default_order=default_order,
                    searcher=searcher,
                    removal_sequence=removal_sequence,
                    elapsed_ns=elapsed_ns,
                )
                write_row(writer, csv_file, row)

                print(
                    f"Run {removed_prefix_count + 1}/{len(default_order)}: "
                    f"{len(included_rods)} rods, "
                    f"{elapsed_ns / 1_000_000_000:.3f}s"
                )
                visualize_run(
                    args=args,
                    run_index=removed_prefix_count + 1,
                    searcher=searcher,
                    removal_sequence=removal_sequence,
                )

    print(f"\nSaved results to: {args.output_csv}")


if __name__ == "__main__":
    main()
