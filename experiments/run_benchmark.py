from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from time import perf_counter

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backward_search import AssemblyPlanner
from DataClasses import AssemblyPlan
from experiment_metrics import (
    CounterMetrics,
    structural_summary,
    summarize_records,
    support_summary,
)
from truss import Truss


DEFAULT_STRATEGIES = [
    "default",
    "baseline",
    "highest_first",
    "rankbased",
    "full_reduce_support",
    "fast_reduce_support",
    "reduced_new_supports",
    "depth_first_cumulative_additions",
]


def create_rai_builder(truss, main_robot_arm_count, metrics, random_seed=0):
    from rai.builder import RaiTrussBuilder

    builder = RaiTrussBuilder(
        truss=truss,
        radius=0.005,
        scale=0.001,
        main_robot_arm_count=main_robot_arm_count,
        metrics=metrics,
        random_seed=random_seed,
    )
    builder.import_robots()
    return builder


def validate_with_rai(
    builder,
    q_initial,
    structural_steps,
    rai_cache,
    use_rrt,
    do_shortcut,
    view_last_komo_attempt,
    use_ssik_initialization,
    support_fractions,
):
    from main import validate_structural_plan_with_rai

    return validate_structural_plan_with_rai(
        builder=builder,
        q_initial=q_initial,
        structural_steps=structural_steps,
        rai_cache=rai_cache,
        use_rrt=use_rrt,
        do_shortcut=do_shortcut,
        view_last_komo_attempt=view_last_komo_attempt,
        use_ssik_initialization=use_ssik_initialization,
        support_fractions=support_fractions,
    )


def show_strategy_in_viser(args, truss, strategy_name, accepted_sequence, accepted_records):
    removal_plan = AssemblyPlan(
        removal_sequence=list(accepted_sequence),
        records=list(accepted_records),
    )
    assembly_plan = AssemblyPlan.reverse_removal_plan_to_assembly(removal_plan)

    replay_builder = create_rai_builder(
        truss=truss,
        main_robot_arm_count=args.main_robot_arm_count,
        metrics=CounterMetrics(),
        random_seed=0,
    )

    print(
        f"\nOpening Viser replay for strategy '{strategy_name}' "
        f"on port {args.viser_port}."
    )

    replay_builder.display_recorded_plan_viser(
        assembly_plan,
        port=args.viser_port,
        pause_time=args.viser_pause_time,
        rod_pos=(-3.0, -1.0, 1.0),
        rod_ori=(0.5, 0.0, 0.5, 0.70710678),
        replay_mode="assembly",
        replay_reduction=args.viser_reduction,
    )


def flatten_for_csv(row):
    flattened = {}

    for key, value in row.items():
        if isinstance(value, (list, tuple)):
            flattened[key] = json.dumps(value)
        elif isinstance(value, dict):
            flattened[key] = json.dumps(value, sort_keys=True)
        else:
            flattened[key] = value

    return flattened


def run_structural_round(
    truss,
    strategy_name,
    support_grippers,
    forbidden_transitions,
    seed,
    max_runtime,
    shuffle_ties,
    capture_key,
    optimal_objective,
    support_target_order,
):
    searcher = AssemblyPlanner(
        truss=truss,
        builder=None,
        max_supports=len(support_grippers),
        support_grippers=support_grippers,
        forbidden_transitions=forbidden_transitions,
        strategy_name=strategy_name,
        random_seed=seed,
        shuffle_ties=shuffle_ties,
        optimal_objective=optimal_objective,
        support_target_order=support_target_order,
    )

    start = perf_counter()
    sequence = searcher.backward_search(
        capture_key=capture_key,
        max_runtime=max_runtime,
    )
    elapsed = perf_counter() - start

    return searcher, sequence, elapsed


def run_strategy(args, strategy_name, repeat_index):
    truss = Truss.from_json(args.truss)
    support_grippers = tuple(args.support_grippers.split(","))
    support_fractions = tuple(
        float(value)
        for value in args.support_fractions.split(",")
        if value.strip()
    )
    if not support_fractions:
        raise ValueError("--support-fractions must not be empty")

    seed = args.seed + repeat_index
    metrics = CounterMetrics()

    total_start = perf_counter()
    forbidden_transitions = set()
    rai_cache = {}
    accepted_sequence = None
    accepted_records = []
    accepted_structural_steps = []
    final_searcher = None

    cumulative = {
        "structural_time_s": 0.0,
        "search_expansions": 0,
        "search_attempted_transitions": 0,
        "search_enqueued_candidates": 0,
        "rigidity_check_calls": 0,
        "rigidity_cache_hits": 0,
        "rigidity_cache_misses": 0,
    }

    rai_builder = None
    rai_initial_q = None

    if args.rai:
        rai_builder = create_rai_builder(
            truss=truss,
            main_robot_arm_count=args.main_robot_arm_count,
            metrics=metrics,
            random_seed=seed,
        )
        rai_initial_q = rai_builder.C.getJointState().copy()

    for replan_index in range(args.max_replans):
        searcher, sequence, structural_time = run_structural_round(
            truss=truss,
            strategy_name=strategy_name,
            support_grippers=support_grippers,
            forbidden_transitions=forbidden_transitions,
            seed=seed,
            max_runtime=args.max_runtime,
            shuffle_ties=args.shuffle_ties,
            capture_key=args.capture_key,
            optimal_objective=args.optimal_objective,
            support_target_order=args.support_target_order,
        )
        final_searcher = searcher
        cumulative["structural_time_s"] += structural_time

        structural = structural_summary(searcher)
        for key in (
            "search_expansions",
            "search_attempted_transitions",
            "search_enqueued_candidates",
            "rigidity_check_calls",
            "rigidity_cache_hits",
            "rigidity_cache_misses",
        ):
            cumulative[key] += structural[key]

        if sequence is None:
            break

        if not args.rai:
            accepted_sequence = list(sequence)
            accepted_structural_steps = list(
                searcher.final_node.structural_steps
            )
            break

        validation = validate_with_rai(
            builder=rai_builder,
            q_initial=rai_initial_q,
            structural_steps=searcher.final_node.structural_steps,
            rai_cache=rai_cache,
            use_rrt=args.rrt,
            do_shortcut=args.shortcut,
            view_last_komo_attempt=args.view_last_komo_attempt,
            use_ssik_initialization=not args.no_ssik_initialization,
            support_fractions=support_fractions,
        )

        if validation["success"]:
            accepted_sequence = list(sequence)
            accepted_records = list(validation["records"])
            accepted_structural_steps = list(
                searcher.final_node.structural_steps
            )
            break

        failed_transition = (
            AssemblyPlanner.structural_transition_key_from_step(
                validation["failed_step"]
            )
        )
        forbidden_transitions.add(failed_transition)

    total_time = perf_counter() - total_start
    success = accepted_sequence is not None

    row = {
        "strategy": strategy_name,
        "repeat": repeat_index,
        "seed": seed,
        "shuffle_ties": args.shuffle_ties,
        "success": success,
        "use_rai": args.rai,
        "use_rrt": args.rrt if args.rai else False,
        "do_shortcut": args.shortcut if args.rai else False,
        "total_time_s": total_time,
        "structural_replans": len(forbidden_transitions) + 1,
        "forbidden_rai_transitions": len(forbidden_transitions),
        "sequence_length": len(accepted_sequence or []),
        "removal_sequence": accepted_sequence or [],
        "assembly_sequence": (
            list(reversed(accepted_sequence))
            if accepted_sequence is not None
            else []
        ),
        "search_stop_reason": (
            final_searcher.search_stop_reason
            if final_searcher is not None
            else None
        ),
        "support_target_order": args.support_target_order,
        "optimal_objective": (
            args.optimal_objective
            if strategy_name == "optimal_supports"
            else None
        ),
        "optimality_proven": (
            final_searcher.optimality_proven
            if final_searcher is not None
            else False
        ),
        "best_support_moves": (
            final_searcher.best_support_moves
            if final_searcher is not None
            else None
        ),
        "best_support_peak": (
            final_searcher.best_support_peak
            if final_searcher is not None
            else None
        ),
        "best_support_steps": (
            final_searcher.best_support_steps
            if final_searcher is not None
            else None
        ),
        "optimal_support_moves": (
            final_searcher.optimal_support_moves
            if final_searcher is not None
            else None
        ),
        "optimal_support_peak": (
            final_searcher.optimal_support_peak
            if final_searcher is not None
            else None
        ),
        "optimal_support_steps": (
            final_searcher.optimal_support_steps
            if final_searcher is not None
            else None
        ),
    }

    row.update(cumulative)
    row.update(support_summary(accepted_structural_steps))
    row.update(metrics.snapshot())

    if args.rai and rai_builder is not None:
        row.update(
            summarize_records(
                accepted_records,
                joint_names=rai_builder.C.getJointNames(),
            )
        )

    if args.viser and args.rai and success:
        show_strategy_in_viser(
            args=args,
            truss=truss,
            strategy_name=strategy_name,
            accepted_sequence=accepted_sequence,
            accepted_records=accepted_records,
        )

    return row


def write_results(rows, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "benchmark.csv"
    json_path = output_dir / "benchmark_details.json"

    fieldnames = sorted(
        {
            key
            for row in rows
            for key in flatten_for_csv(row)
        }
    )

    with csv_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow(flatten_for_csv(row))

    with json_path.open("w") as json_file:
        json.dump(rows, json_file, indent=2, sort_keys=True)

    return csv_path, json_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare structural and RAI planning strategies."
    )
    parser.add_argument(
        "--truss",
        default=str(
            PROJECT_ROOT
            / "JSON"
            / "own_examples"
            / "260804_RobArchDemo_ini.json"
        ),
    )
    parser.add_argument(
        "--strategies",
        default=",".join(DEFAULT_STRATEGIES),
        help="Comma-separated strategy names.",
    )
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--shuffle-ties",
        action="store_true",
        help="Use seeded random tie-breaking instead of rod-id tie-breaking.",
    )
    parser.add_argument("--max-runtime", type=float, default=1800.0)
    parser.add_argument(
        "--support-target-order",
        choices=(
            "highest_first",
            "lowest_first",
            "random",
            "closest_to_removed",
            "furthest_from_supported",
        ),
        default="lowest_first",
        help=(
            "Ordering used when choosing structural support rods; distance "
            "modes use Euclidean distance between rod centers."
        ),
    )
    parser.add_argument(
        "--optimal-objective",
        choices=("support_moves", "support_steps", "peak"),
        default="support_moves",
        help="Objective used by the optimal_supports strategy.",
    )
    parser.add_argument("--max-replans", type=int, default=1000)
    parser.add_argument(
        "--capture-key",
        default="v",
        help=(
            "Terminal key used to start/stop structural debug capture. "
            "Use an empty value with --capture-key='' to disable."
        ),
    )
    parser.add_argument(
        "--no-capture",
        action="store_true",
        help="Disable terminal hotkey capture during structural search.",
    )
    parser.add_argument("--rai", action="store_true")
    parser.add_argument("--rrt", action="store_true")
    parser.add_argument("--shortcut", action="store_true")
    parser.add_argument(
        "--view-last-komo-attempt",
        action="store_true",
        help=(
            "When RAI validation exhausts a removal transition, open the "
            "viewer on the last analytical KOMO combination actually tried."
        ),
    )
    parser.add_argument(
        "--no-ssik-initialization",
        action="store_true",
        help=(
            "Sample mobile-base placements only and let KOMO solve arm joints "
            "without SSIK-seeded arm configurations."
        ),
    )
    parser.add_argument(
        "--viser",
        action="store_true",
        help="Open a Viser replay after each successful RAI-validated strategy.",
    )
    parser.add_argument("--viser-port", type=int, default=8080)
    parser.add_argument("--viser-reduction", type=int, default=10)
    parser.add_argument("--viser-pause-time", type=float, default=0.03)
    parser.add_argument("--main-robot-arm-count", type=int, default=1)
    parser.add_argument(
        "--support-grippers",
        default=(
            "h1_a1_ur_gripper_center,"
            "h2_a1_ur_gripper_center"
        ),
    )
    parser.add_argument(
        "--support-fractions",
        default="0.5, 0.6",
        help=(
            "Comma-separated rod fractions to try for newly added supports."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "experiments" / "results"),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    args.capture_key = None if args.no_capture else args.capture_key.strip()
    if not args.capture_key:
        args.capture_key = None

    strategies = [
        strategy.strip()
        for strategy in args.strategies.split(",")
        if strategy.strip()
    ]

    rows = []

    for repeat_index in range(args.repeat):
        for strategy_name in strategies:
            print(
                "\n"
                + "=" * 70
                + f"\nBenchmark strategy: {strategy_name}"
                + f"\nRepeat: {repeat_index}"
                + "\n"
                + "=" * 70
            )
            rows.append(
                run_strategy(
                    args=args,
                    strategy_name=strategy_name,
                    repeat_index=repeat_index,
                )
            )

    csv_path, json_path = write_results(rows, args.output_dir)

    print("\nBenchmark complete.")
    print(f"CSV:  {csv_path}")
    print(f"JSON: {json_path}")


if __name__ == "__main__":
    main()
