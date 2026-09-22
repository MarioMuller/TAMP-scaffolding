from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backward_search import AssemblyPlanner
from experiments.run_benchmark import (
    CONFIG_JSON_NAME,
    POSE_REPLAY_FORMAT,
    POSE_REPLAY_JSONL_NAME,
    POSE_REPLAY_VERSION,
    RESULTS_JSON_NAME,
    run_strategy,
    write_results,
)
from experiments.structural_experiment_utils import load_filtered_truss


RUN_SEED_STRIDE = 10_000


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run structural search plus RAI validation on a full scaffold "
            "and on prefixes formed by removing two rods at a time."
        )
    )
    parser.add_argument(
        "--truss",
        type=Path,
        default=(
            PROJECT_ROOT
            / "JSON"
            / "own_examples"
            / "260804_RobArchDemo_two_boxes.json"
        ),
    )
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--seed", type=int, default=200452)
    parser.add_argument(
        "--strategy-name",
        choices=AssemblyPlanner.STRATEGY_NAMES,
        default="fast_reduce_support",
    )
    parser.add_argument(
        "--support-target-order",
        choices=AssemblyPlanner.SUPPORT_TARGET_ORDERS,
        default="lowest_first",
    )
    parser.add_argument("--shuffle-ties", action="store_true")
    parser.add_argument("--max-runtime", type=float, default=400.0)
    parser.add_argument("--max-total-runtime", type=float, default=7200.0)
    parser.add_argument("--max-replans", type=int, default=50)
    parser.add_argument("--main-robot-arm-count", type=int, default=1)
    parser.add_argument(
        "--support-grippers",
        default=(
            "h1_a1_ur_gripper_center,"
            "h2_a1_ur_gripper_center"
        ),
    )
    parser.add_argument("--support-fractions", default="0.4,0.5,0.6")
    parser.add_argument("--rrt", action="store_true")
    parser.add_argument("--shortcut", action="store_true")
    parser.add_argument("--view-last-komo-attempt", action="store_true")
    parser.add_argument("--no-ssik-initialization", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            PROJECT_ROOT
            / "experiments"
            / "results"
            / "evaluation"
            / "current"
            / "two_boxes_one_arm_rai_scaling_5reps"
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue an interrupted compatible run.",
    )
    return parser.parse_args()


def validate_args(args):
    if args.repetitions <= 0:
        raise ValueError("--repetitions must be positive.")
    if args.max_runtime <= 0:
        raise ValueError("--max-runtime must be positive.")
    if args.max_total_runtime <= 0:
        raise ValueError("--max-total-runtime must be positive.")
    if args.max_replans <= 0:
        raise ValueError("--max-replans must be positive.")
    if args.main_robot_arm_count not in (1, 2):
        raise ValueError("--main-robot-arm-count must be 1 or 2.")


def config_for(args):
    return {
        "format": "tamp-scaffolding-rai-scaling",
        "version": 1,
        "truss": str(args.truss.resolve()),
        "repetitions": args.repetitions,
        "seed": args.seed,
        "run_seed_stride": RUN_SEED_STRIDE,
        "strategy_name": args.strategy_name,
        "support_target_order": args.support_target_order,
        "shuffle_ties": args.shuffle_ties,
        "max_runtime": args.max_runtime,
        "max_total_runtime": args.max_total_runtime,
        "max_replans": args.max_replans,
        "main_robot_arm_count": args.main_robot_arm_count,
        "support_grippers": args.support_grippers,
        "support_fractions": args.support_fractions,
        "use_rrt": args.rrt,
        "do_shortcut": args.shortcut,
        "use_ssik_initialization": not args.no_ssik_initialization,
        "prefix_step": 2,
    }


def prepare_output(args):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config_path = args.output_dir / CONFIG_JSON_NAME
    details_path = args.output_dir / RESULTS_JSON_NAME
    config = config_for(args)

    if args.resume:
        if not config_path.exists():
            raise ValueError("Cannot resume: benchmark_config.json is missing.")
        existing_config = json.loads(config_path.read_text())
        if existing_config != config:
            raise ValueError(
                "Cannot resume because the saved configuration differs "
                "from the requested configuration."
            )
        rows = json.loads(details_path.read_text()) if details_path.exists() else []
    else:
        rows = []
        config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")

    return rows


def open_replay_file(args):
    replay_path = args.output_dir / POSE_REPLAY_JSONL_NAME
    append = args.resume and replay_path.exists()
    replay_file = replay_path.open("a" if append else "w", encoding="utf-8")

    if not append:
        metadata = {
            "record_type": "metadata",
            "format": POSE_REPLAY_FORMAT,
            "version": POSE_REPLAY_VERSION,
            "truss": str(args.truss.resolve()),
            "main_robot_arm_count": args.main_robot_arm_count,
            "rod_position": [-3.0, -1.0, 1.0],
            "rod_orientation": [0.5, 0.0, 0.5, 0.70710678],
            "scaling_prefix_step": 2,
        }
        replay_file.write(json.dumps(metadata, sort_keys=True) + "\n")
        replay_file.flush()

    return replay_path, replay_file


def add_scaling_fields(
    row,
    *,
    repetition,
    run_index,
    removed_prefix_count,
    included_rods,
    excluded_rods,
    initial_supported,
    default_order,
):
    removal_sequence = list(row.get("removal_sequence", []))
    row.update(
        {
            "scaling_repetition": repetition,
            "scaling_run_index": run_index,
            "removed_prefix_count": removed_prefix_count,
            "included_rod_count": len(included_rods),
            "included_rods": sorted(int(rod) for rod in included_rods),
            "excluded_rods": sorted(int(rod) for rod in excluded_rods),
            "initial_supports": dict(initial_supported),
            "initial_supported_rods": sorted(
                int(rod) for rod in initial_supported.values()
            ),
            "initial_support_count": len(initial_supported),
            "default_order": [int(rod) for rod in default_order],
            "matches_default_suffix": (
                bool(row.get("success"))
                and removal_sequence == default_order[removed_prefix_count:]
            ),
        }
    )


def write_replay(
    replay_file,
    replay,
    *,
    repetition,
    run_index,
    removed_prefix_count,
    included_rods,
    initial_supported,
):
    if replay is None:
        return
    replay.update(
        {
            "scaling_repetition": repetition,
            "scaling_run_index": run_index,
            "removed_prefix_count": removed_prefix_count,
            "included_rods": sorted(int(rod) for rod in included_rods),
            "initial_supports": dict(initial_supported),
        }
    )
    replay_file.write(json.dumps(replay, separators=(",", ":")) + "\n")
    replay_file.flush()


def run_once(
    args,
    *,
    truss,
    repetition_index,
    seed,
    initial_supported=None,
    initial_support_q=None,
    initial_q=None,
):
    # run_strategy expects the benchmark command's two presentation controls.
    args.rai = True
    args.capture_key = None
    args.viser = False

    return run_strategy(
        args=args,
        strategy_name=args.strategy_name,
        repeat_index=repetition_index,
        truss=truss,
        seed=seed,
        initial_supported=initial_supported,
        initial_support_q=initial_support_q,
        initial_q=initial_q,
        return_artifacts=True,
    )


def main():
    args = parse_args()
    validate_args(args)
    rows = prepare_output(args)
    replay_path, replay_file = open_replay_file(args)
    all_rods = sorted(load_filtered_truss(args.truss).elements)
    prefix_counts = list(range(0, len(all_rods), 2))
    completed = {
        (int(row["scaling_repetition"]), int(row["removed_prefix_count"]))
        for row in rows
    }

    try:
        for repetition_index in range(args.repetitions):
            repetition = repetition_index + 1
            repetition_seed = args.seed + repetition_index * RUN_SEED_STRIDE
            repetition_keys = {
                (repetition, prefix_count)
                for prefix_count in prefix_counts
            }
            if args.resume and repetition_keys <= completed:
                print(f"Skipping complete repetition {repetition}.")
                continue

            print(
                "\n"
                + "=" * 70
                + f"\nRAI scaling repetition {repetition}/{args.repetitions}"
                + f"\nFull-run seed: {repetition_seed}"
                + "\n"
                + "=" * 70
            )

            full_truss = load_filtered_truss(args.truss)
            full_row, full_replay, full_artifacts = run_once(
                args,
                truss=full_truss,
                repetition_index=repetition_index,
                seed=repetition_seed,
            )
            default_order = list(full_artifacts["accepted_sequence"])
            full_key = (repetition, 0)

            if full_key not in completed:
                add_scaling_fields(
                    full_row,
                    repetition=repetition,
                    run_index=1,
                    removed_prefix_count=0,
                    included_rods=all_rods,
                    excluded_rods=[],
                    initial_supported={},
                    default_order=default_order,
                )
                rows.append(full_row)
                completed.add(full_key)
                write_results(rows, args.output_dir)
                write_replay(
                    replay_file,
                    full_replay,
                    repetition=repetition,
                    run_index=1,
                    removed_prefix_count=0,
                    included_rods=all_rods,
                    initial_supported={},
                )

            if not full_row["success"]:
                print(
                    "The full scaffold did not pass structural and RAI "
                    "validation; skipping this repetition's prefixes."
                )
                continue

            state_trace = full_artifacts["state_trace"]
            if len(state_trace) != len(default_order) + 1:
                raise RuntimeError(
                    "The validated full plan did not provide one physical "
                    "state snapshot per removal step."
                )

            total_runs = len(prefix_counts)
            print(
                f"Run 1/{total_runs}: {len(all_rods)} rods, "
                f"success={full_row['success']}, "
                f"{full_row['total_time_s']:.3f}s"
            )

            for run_index, removed_prefix_count in enumerate(
                prefix_counts[1:],
                start=2,
            ):
                key = (repetition, removed_prefix_count)
                if args.resume and key in completed:
                    print(
                        f"Skipping recorded prefix with "
                        f"{len(all_rods) - removed_prefix_count} rods."
                    )
                    continue

                excluded_rods = default_order[:removed_prefix_count]
                excluded_set = set(excluded_rods)
                included_rods = [
                    rod for rod in all_rods if rod not in excluded_set
                ]
                initial_state = state_trace[removed_prefix_count]
                initial_supported = dict(initial_state["supported"])
                run_seed = repetition_seed + run_index - 1
                filtered_truss = load_filtered_truss(
                    args.truss,
                    included_rods,
                )

                row, replay, _artifacts = run_once(
                    args,
                    truss=filtered_truss,
                    repetition_index=repetition_index,
                    seed=run_seed,
                    initial_supported=initial_supported,
                    initial_support_q=initial_state["support_q"],
                    initial_q=initial_state["q"],
                )
                add_scaling_fields(
                    row,
                    repetition=repetition,
                    run_index=run_index,
                    removed_prefix_count=removed_prefix_count,
                    included_rods=included_rods,
                    excluded_rods=excluded_rods,
                    initial_supported=initial_supported,
                    default_order=default_order,
                )
                rows.append(row)
                completed.add(key)
                write_results(rows, args.output_dir)
                write_replay(
                    replay_file,
                    replay,
                    repetition=repetition,
                    run_index=run_index,
                    removed_prefix_count=removed_prefix_count,
                    included_rods=included_rods,
                    initial_supported=initial_supported,
                )
                print(
                    f"Run {run_index}/{total_runs}: seed {run_seed}, "
                    f"{len(included_rods)} rods, "
                    f"{len(initial_supported)} inherited supports, "
                    f"success={row['success']}, "
                    f"{row['total_time_s']:.3f}s"
                )
    finally:
        replay_file.close()

    csv_path, json_path = write_results(rows, args.output_dir)
    print("\nRAI scaling complete.")
    print(f"CSV:  {csv_path}")
    print(f"JSON: {json_path}")
    print(f"Pose replays: {replay_path}")


if __name__ == "__main__":
    main()
