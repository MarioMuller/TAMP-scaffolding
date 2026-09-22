from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from DataClasses import AssemblyPlan, AttachmentEvent, RodPathRecord
from experiment_metrics import CounterMetrics
from experiments.run_benchmark import (
    POSE_REPLAY_FORMAT,
    POSE_REPLAY_VERSION,
    create_rai_builder,
)
from experiments.structural_experiment_utils import load_filtered_truss


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Replay a saved benchmark pose plan in Viser without rerunning "
            "structural or RAI planning."
        )
    )
    parser.add_argument("replay_jsonl", type=Path)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--strategy")
    parser.add_argument("--repeat", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--replay-mode",
        choices=("assembly", "removal"),
        default="assembly",
    )
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--pause-time", type=float, default=0.1)
    parser.add_argument("--replay-reduction", type=int, default=1)
    return parser.parse_args()


def load_replays(path):
    metadata = None
    replay_by_key = {}

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
                key = (
                    record["strategy"],
                    int(record["repeat"]),
                    int(record["seed"]),
                )
                replay_by_key[key] = record

    if metadata is None:
        raise ValueError(f"Replay metadata is missing from {path}.")
    if metadata.get("format") != POSE_REPLAY_FORMAT:
        raise ValueError(
            f"Unsupported replay format: {metadata.get('format')!r}."
        )
    if metadata.get("version") != POSE_REPLAY_VERSION:
        raise ValueError(
            f"Unsupported replay version: {metadata.get('version')!r}."
        )

    return metadata, list(replay_by_key.values())


def list_replays(replays):
    if not replays:
        print("No complete RAI-validated pose plans were saved.")
        return

    print("strategy                      repeat    seed  rods  records  rrt")
    for replay in sorted(
        replays,
        key=lambda item: (
            item["repeat"],
            item["strategy"],
            item["seed"],
        ),
    ):
        print(
            f"{replay['strategy']:<29} "
            f"{replay['repeat']:>6} "
            f"{replay['seed']:>7} "
            f"{len(replay['removal_sequence']):>5} "
            f"{len(replay['records']):>8} "
            f"{str(replay['use_rrt']):>4}"
        )


def select_replay(replays, args):
    matches = [
        replay
        for replay in replays
        if (
            args.strategy is None
            or replay["strategy"] == args.strategy
        )
        and (
            args.repeat is None
            or replay["repeat"] == args.repeat
        )
        and (
            args.seed is None
            or replay["seed"] == args.seed
        )
    ]

    if not matches:
        raise ValueError(
            "No pose replay matches the requested selection. "
            "Use --list to see successful runs."
        )
    if len(matches) > 1:
        raise ValueError(
            f"The selection matches {len(matches)} runs. Specify strategy, "
            "repeat, and/or seed more precisely."
        )

    return matches[0]


def deserialize_motion_record(data):
    return RodPathRecord(
        rod_id=int(data["rod_id"]),
        segments=[
            np.asarray(segment, dtype=float)
            for segment in data["segments"]
        ],
        events=[
            AttachmentEvent(
                rod_id=int(event["rod_id"]),
                segment_id=int(event["segment_id"]),
                parent=event["parent"],
                child=event["child"],
                action=event["action"],
            )
            for event in data["events"]
        ],
    )


def display_replay(metadata, replay, args):
    truss = load_filtered_truss(
        metadata["truss"],
        replay.get("included_rods"),
    )
    builder = create_rai_builder(
        truss=truss,
        main_robot_arm_count=metadata["main_robot_arm_count"],
        metrics=CounterMetrics(),
        random_seed=replay["seed"],
    )

    current_joint_names = list(builder.C.getJointNames())
    if current_joint_names != replay["joint_names"]:
        raise RuntimeError(
            "The current RAI model joint ordering differs from the saved "
            "replay. Replaying these configurations would be unsafe."
        )

    removal_plan = AssemblyPlan(
        removal_sequence=list(replay["removal_sequence"]),
        records=[
            deserialize_motion_record(record)
            for record in replay["records"]
        ],
    )
    plan = (
        AssemblyPlan.reverse_removal_plan_to_assembly(removal_plan)
        if args.replay_mode == "assembly"
        else removal_plan
    )

    print(
        f"Opening {args.replay_mode} replay for {replay['strategy']}, "
        f"repeat {replay['repeat']}, seed {replay['seed']}."
    )
    builder.display_recorded_plan_viser(
        plan,
        port=args.port,
        pause_time=args.pause_time,
        rod_pos=metadata["rod_position"],
        rod_ori=metadata["rod_orientation"],
        replay_mode=args.replay_mode,
        replay_reduction=args.replay_reduction,
    )


def main():
    args = parse_args()
    if args.pause_time <= 0:
        raise ValueError("--pause-time must be positive.")
    if args.replay_reduction <= 0:
        raise ValueError("--replay-reduction must be positive.")

    metadata, replays = load_replays(args.replay_jsonl)
    if args.list:
        list_replays(replays)
        return

    replay = select_replay(replays, args)
    display_replay(metadata, replay, args)


if __name__ == "__main__":
    main()
