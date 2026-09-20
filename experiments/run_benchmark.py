from __future__ import annotations

import argparse
import csv
import json
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
    "highest_first",
    "fewest_mandatory_supports",
    "fast_reduce_support",
    "baseline",
]

RESULTS_CSV_NAME = "benchmark.csv"
RESULTS_JSON_NAME = "benchmark_details.json"
CONFIG_JSON_NAME = "benchmark_config.json"
POSE_REPLAY_JSONL_NAME = "benchmark_pose_replays.jsonl"
POSE_REPLAY_FORMAT = "tamp-scaffolding-pose-replay"
POSE_REPLAY_VERSION = 1


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
    deadline,
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
        deadline=deadline,
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


def serialize_motion_record(record):
    return {
        "rod_id": int(record.rod_id),
        "segments": [
            np.asarray(segment, dtype=float).tolist()
            for segment in record.segments
        ],
        "events": [
            {
                "rod_id": int(event.rod_id),
                "segment_id": int(event.segment_id),
                "parent": event.parent,
                "child": event.child,
                "action": event.action,
            }
            for event in record.events
        ],
    }


def make_pose_replay_record(
    *,
    args,
    strategy_name,
    repeat_index,
    seed,
    accepted_sequence,
    accepted_records,
    rai_builder,
):
    if not args.rai or not accepted_records:
        return None

    return {
        "record_type": "run",
        "strategy": strategy_name,
        "repeat": repeat_index,
        "seed": seed,
        "use_rrt": bool(args.rrt),
        "removal_sequence": [
            int(rod_id) for rod_id in accepted_sequence
        ],
        "assembly_sequence": [
            int(rod_id) for rod_id in reversed(accepted_sequence)
        ],
        "joint_names": list(rai_builder.C.getJointNames()),
        "records": [
            serialize_motion_record(record)
            for record in accepted_records
        ],
    }


def remaining_runtime(deadline):
    if deadline is None:
        return None
    return max(0.0, deadline - perf_counter())


def run_structural_round(
    truss,
    strategy_name,
    support_grippers,
    forbidden_transitions,
    seed,
    max_runtime,
    shuffle_ties,
    capture_key,
    support_target_order,
    require_connected_supports,
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
        support_target_order=support_target_order,
        require_connected_supports=require_connected_supports,
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
    deadline = (
        total_start + args.max_total_runtime
        if args.max_total_runtime is not None
        else None
    )
    forbidden_transitions = set()
    rai_cache = {}
    accepted_sequence = None
    accepted_records = []
    accepted_structural_steps = []
    final_searcher = None
    replan_trace = []
    plans_generated = 0
    pose_failure_count = 0
    pose_validation_time_s = 0.0
    benchmark_stop_reason = "max_replans"

    cumulative = {
        "structural_time_s": 0.0,
        "search_expansions": 0,
        "search_attempted_transitions": 0,
        "search_enqueued_candidates": 0,
        "rigidity_check_calls": 0,
        "rigidity_cache_hits": 0,
        "rigidity_cache_misses": 0,
        "rigidity_incremental_support_updates": 0,
        "rigidity_incremental_support_cache_hits": 0,
        "rigidity_incremental_support_fallbacks": 0,
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

    for _replan_index in range(args.max_replans):
        remaining = remaining_runtime(deadline)
        if remaining is not None and remaining <= 0:
            benchmark_stop_reason = "total_runtime_limit"
            break

        structural_runtime = args.max_runtime
        if remaining is not None:
            structural_runtime = min(structural_runtime, remaining)

        plans_generated += 1
        searcher, sequence, structural_time = run_structural_round(
            truss=truss,
            strategy_name=strategy_name,
            support_grippers=support_grippers,
            forbidden_transitions=forbidden_transitions,
            seed=seed,
            max_runtime=structural_runtime,
            shuffle_ties=args.shuffle_ties,
            capture_key=args.capture_key,
            support_target_order=args.support_target_order,
            require_connected_supports=args.require_connected_supports,
        )
        final_searcher = searcher
        cumulative["structural_time_s"] += structural_time
        plan_trace = {
            "plan_number": plans_generated,
            "structural_time_s": structural_time,
            "structural_success": sequence is not None,
            "structural_stop_reason": searcher.search_stop_reason,
            "sequence_length": len(sequence or []),
        }

        structural = structural_summary(searcher)
        for key in (
            "search_expansions",
            "search_attempted_transitions",
            "search_enqueued_candidates",
            "rigidity_check_calls",
            "rigidity_cache_hits",
            "rigidity_cache_misses",
            "rigidity_incremental_support_updates",
            "rigidity_incremental_support_cache_hits",
            "rigidity_incremental_support_fallbacks",
        ):
            cumulative[key] += structural[key]

        if sequence is None:
            plan_trace["outcome"] = "structural_search_failed"
            replan_trace.append(plan_trace)
            benchmark_stop_reason = (
                f"structural_{searcher.search_stop_reason}"
            )
            break

        if not args.rai:
            accepted_sequence = list(sequence)
            accepted_structural_steps = list(
                searcher.final_node.structural_steps
            )
            plan_trace["outcome"] = "structural_plan_accepted"
            replan_trace.append(plan_trace)
            benchmark_stop_reason = "complete"
            break

        pose_start = perf_counter()
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
            deadline=deadline,
        )
        validation_time = perf_counter() - pose_start
        pose_validation_time_s += validation_time
        plan_trace.update(
            {
                "pose_validation_time_s": validation_time,
                "pose_success": validation["success"],
                "validated_transition_count": len(
                    validation["records"]
                ),
                "pose_stop_reason": validation["stop_reason"],
            }
        )

        if validation["success"]:
            accepted_sequence = list(sequence)
            accepted_records = list(validation["records"])
            accepted_structural_steps = list(
                searcher.final_node.structural_steps
            )
            plan_trace["outcome"] = "pose_feasible"
            replan_trace.append(plan_trace)
            benchmark_stop_reason = "complete"
            break

        if validation["stop_reason"] == "total_runtime_limit":
            plan_trace["outcome"] = "total_runtime_limit"
            plan_trace["failed_step_index"] = validation["failed_index"]
            replan_trace.append(plan_trace)
            benchmark_stop_reason = "total_runtime_limit"
            break

        failed_step = validation["failed_step"]
        if failed_step is None:
            raise RuntimeError(
                "RAI validation failed without identifying a transition."
            )

        plan_trace.update(
            {
                "outcome": "pose_infeasible",
                "failed_step_index": validation["failed_index"],
                "failed_rod": int(failed_step.rod_id),
                "failed_supports_before": dict(
                    failed_step.supports_before
                ),
                "failed_supports_after": dict(
                    failed_step.supports_after
                ),
            }
        )
        replan_trace.append(plan_trace)
        pose_failure_count += 1

        failed_transition = (
            AssemblyPlanner.structural_transition_key_from_step(
                failed_step
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
        "pose_validation_time_s": pose_validation_time_s,
        "structural_plans_generated": plans_generated,
        "structural_replans": plans_generated,
        "rai_pose_failures": pose_failure_count,
        "rai_replans": max(0, plans_generated - 1),
        "forbidden_rai_transitions": len(forbidden_transitions),
        "first_plan_pose_feasible": (
            bool(args.rai and success and plans_generated == 1)
        ),
        "benchmark_stop_reason": benchmark_stop_reason,
        "max_total_runtime_s": args.max_total_runtime,
        "replan_trace": replan_trace,
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
        "require_connected_supports": args.require_connected_supports,
    }

    row.update(cumulative)
    row.update(support_summary(accepted_structural_steps))
    row.update(metrics.snapshot())
    for metric_name in (
        "komo_calls",
        "komo_errors",
        "komo_feasible",
        "komo_infeasible",
        "komo_ssik_success",
        "rai_cache_hits",
        "rai_cache_misses",
        "rai_robot_model_imports",
        "rai_robot_template_cache_hits",
        "rai_scene_template_restores",
        "rai_transition_attempts",
        "rai_transition_failures",
        "rai_transition_successes",
        "rrt_accepted_candidates",
        "rrt_calls",
        "rrt_candidate_rejections",
        "rrt_failures",
        "rrt_successes",
    ):
        row.setdefault(metric_name, 0)

    for metric_name in (
        "komo_time_s",
        "rai_robot_import_time_s",
        "rai_robot_template_restore_time_s",
        "rai_scene_reset_time_s",
        "rrt_time_s",
    ):
        row.setdefault(metric_name, 0.0)

    for metric_name in (
        "komo_ssik_attempts_until_success",
        "komo_ssik_available_combinations",
    ):
        row.setdefault(metric_name, [])

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

    pose_replay = make_pose_replay_record(
        args=args,
        strategy_name=strategy_name,
        repeat_index=repeat_index,
        seed=seed,
        accepted_sequence=accepted_sequence or [],
        accepted_records=accepted_records,
        rai_builder=rai_builder,
    )

    return row, pose_replay


def write_results(rows, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / RESULTS_CSV_NAME
    json_path = output_dir / RESULTS_JSON_NAME
    csv_tmp_path = csv_path.with_suffix(".csv.tmp")
    json_tmp_path = json_path.with_suffix(".json.tmp")

    fieldnames = sorted(
        {
            key
            for row in rows
            for key in flatten_for_csv(row)
        }
    )

    with csv_tmp_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow(flatten_for_csv(row))

    with json_tmp_path.open("w") as json_file:
        json.dump(rows, json_file, indent=2, sort_keys=True)

    csv_tmp_path.replace(csv_path)
    json_tmp_path.replace(json_path)
    return csv_path, json_path


def benchmark_config(args):
    return {
        "format_version": 1,
        "truss": str(Path(args.truss).resolve()),
        "seed": args.seed,
        "shuffle_ties": args.shuffle_ties,
        "max_runtime": args.max_runtime,
        "max_total_runtime": args.max_total_runtime,
        "max_replans": args.max_replans,
        "support_target_order": args.support_target_order,
        "require_connected_supports": args.require_connected_supports,
        "use_rai": args.rai,
        "use_rrt": args.rrt,
        "do_shortcut": args.shortcut,
        "use_ssik_initialization": not args.no_ssik_initialization,
        "main_robot_arm_count": args.main_robot_arm_count,
        "support_grippers": args.support_grippers,
        "support_fractions": args.support_fractions,
    }


def write_json_atomic(path, value):
    path = Path(path)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w") as output_file:
        json.dump(value, output_file, indent=2, sort_keys=True)
    tmp_path.replace(path)


def prepare_output(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / CONFIG_JSON_NAME
    details_path = output_dir / RESULTS_JSON_NAME
    config = benchmark_config(args)

    if args.resume and config_path.exists():
        with config_path.open() as config_file:
            existing_config = json.load(config_file)

        if existing_config != config:
            differing_keys = sorted(
                key
                for key in set(existing_config) | set(config)
                if existing_config.get(key) != config.get(key)
            )
            raise ValueError(
                "Cannot resume because benchmark settings changed: "
                + ", ".join(differing_keys)
            )

        if details_path.exists():
            with details_path.open() as details_file:
                rows = json.load(details_file)
        else:
            rows = []
    else:
        rows = []
        write_json_atomic(config_path, config)

    if args.resume and not config_path.exists():
        write_json_atomic(config_path, config)

    return output_dir, rows


def open_pose_replay_file(args, output_dir):
    replay_path = output_dir / POSE_REPLAY_JSONL_NAME
    append = args.resume and replay_path.exists()
    replay_file = replay_path.open(
        "a" if append else "w",
        encoding="utf-8",
    )

    if not append:
        replay_file.write(
            json.dumps(
                {
                    "record_type": "metadata",
                    "format": POSE_REPLAY_FORMAT,
                    "version": POSE_REPLAY_VERSION,
                    "truss": str(Path(args.truss).resolve()),
                    "main_robot_arm_count": args.main_robot_arm_count,
                    "rod_position": [-3.0, -1.0, 1.0],
                    "rod_orientation": [
                        0.5,
                        0.0,
                        0.5,
                        0.70710678,
                    ],
                },
                sort_keys=True,
            )
            + "\n"
        )
        replay_file.flush()

    return replay_path, replay_file


def write_pose_replay(replay_file, replay_record):
    if replay_record is None:
        return
    replay_file.write(
        json.dumps(replay_record, separators=(",", ":")) + "\n"
    )
    replay_file.flush()


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
        "--max-total-runtime",
        type=float,
        default=3600.0,
        help=(
            "Maximum wall-clock seconds for one strategy/seed, including "
            "all structural replans and pose validation."
        ),
    )
    parser.add_argument(
        "--support-target-order",
        choices=AssemblyPlanner.SUPPORT_TARGET_ORDERS,
        default="random",
        help=(
            "Ordering used when choosing structural support rods; distance "
            "modes use Euclidean distance between rod centers."
        ),
    )
    parser.add_argument(
        "--require-connected-supports",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Require every supported rod to remain coupled to at least one "
            "other active rod (default: enabled)."
        ),
    )
    parser.add_argument(
        "--max-replans",
        type=int,
        default=1000,
        help=(
            "Maximum structural plans generated for one strategy/seed, "
            "including the initial plan."
        ),
    )
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
        default="0.4, 0.5, 0.6",
        help=(
            "Comma-separated rod fractions to try for newly added supports."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "experiments" / "results"),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Continue a compatible interrupted benchmark and skip completed "
            "strategy/seed pairs."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.repeat <= 0:
        raise ValueError("--repeat must be positive.")
    if args.max_runtime <= 0:
        raise ValueError("--max-runtime must be positive.")
    if args.max_total_runtime is not None and args.max_total_runtime <= 0:
        raise ValueError("--max-total-runtime must be positive.")
    if args.max_replans <= 0:
        raise ValueError("--max-replans must be positive.")

    args.capture_key = None if args.no_capture else args.capture_key.strip()
    if not args.capture_key:
        args.capture_key = None

    strategies = [
        strategy.strip()
        for strategy in args.strategies.split(",")
        if strategy.strip()
    ]
    unknown_strategies = set(strategies) - set(
        AssemblyPlanner.STRATEGY_NAMES
    )
    if unknown_strategies:
        raise ValueError(
            "Unknown strategies: "
            f"{', '.join(sorted(unknown_strategies))}."
        )

    output_dir, rows = prepare_output(args)
    completed = {
        (
            row["strategy"],
            int(row["repeat"]),
            int(row["seed"]),
        )
        for row in rows
    }
    replay_path, replay_file = open_pose_replay_file(
        args,
        output_dir,
    )
    csv_path = output_dir / RESULTS_CSV_NAME
    json_path = output_dir / RESULTS_JSON_NAME

    try:
        for repeat_index in range(args.repeat):
            seed = args.seed + repeat_index

            for strategy_name in strategies:
                run_key = (strategy_name, repeat_index, seed)
                if args.resume and run_key in completed:
                    print(
                        "Skipping completed run: "
                        f"{strategy_name}, repeat {repeat_index}, seed {seed}"
                    )
                    continue

                print(
                    "\n"
                    + "=" * 70
                    + f"\nBenchmark strategy: {strategy_name}"
                    + f"\nRepeat: {repeat_index}"
                    + f"\nSeed: {seed}"
                    + "\n"
                    + "=" * 70
                )
                row, pose_replay = run_strategy(
                    args=args,
                    strategy_name=strategy_name,
                    repeat_index=repeat_index,
                )

                write_pose_replay(replay_file, pose_replay)
                rows.append(row)
                completed.add(run_key)
                csv_path, json_path = write_results(
                    rows,
                    output_dir,
                )
                print(
                    "Saved incremental result: "
                    f"{strategy_name}, repeat {repeat_index}, seed {seed}"
                )
    finally:
        replay_file.close()

    print("\nBenchmark complete.")
    print(f"CSV:  {csv_path}")
    print(f"JSON: {json_path}")
    print(f"Pose replays: {replay_path}")


if __name__ == "__main__":
    main()
