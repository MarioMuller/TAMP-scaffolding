from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Any

import numpy as np


@dataclass
class CounterMetrics:
    counters: dict[str, int] = field(default_factory=dict)
    totals: dict[str, float] = field(default_factory=dict)
    samples: dict[str, list] = field(default_factory=dict)

    def inc(self, name: str, amount: int = 1) -> None:
        self.counters[name] = self.counters.get(name, 0) + amount

    def add(self, name: str, value: float) -> None:
        self.totals[name] = self.totals.get(name, 0.0) + float(value)
        
    def sample(self, name: str, value) -> None:
            self.samples.setdefault(name, []).append(value)
    

    def snapshot(self) -> dict[str, Any]:
        values: dict[str, Any] = {}
        values.update(self.counters)
        values.update(self.totals)
        values.update(self.samples)
        return values
    
    

class Timer:
    def __init__(self, metrics: CounterMetrics, name: str):
        self.metrics = metrics
        self.name = name
        self.start = None

    def __enter__(self):
        self.start = perf_counter()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.metrics.add(self.name, perf_counter() - self.start)


def path_cost(path, weights=None) -> float:
    path = np.asarray(path, dtype=float)

    if path.ndim != 2 or len(path) < 2:
        return 0.0

    diffs = np.diff(path, axis=0)

    if weights is not None:
        diffs = diffs * np.asarray(weights, dtype=float)

    return float(np.sum(np.linalg.norm(diffs, axis=1)))


def summarize_records(records, joint_names=None) -> dict[str, Any]:
    summary = {
        "record_count": len(records),
        "motion_segments": 0,
        "path_points": 0,
        "path_cost": 0.0,
        "support_add_events": 0,
        "support_release_events": 0,
        "main_attach_events": 0,
        "main_detach_events": 0,
        "locked_support_drift_total": 0.0,
        "locked_support_drift_max": 0.0,
    }

    base_indices = {}
    support_indices_by_gripper = {}

    if joint_names is not None:
        joint_names = list(joint_names)

        for base_name in (
            "husky_base_XYPhi_joint",
            "h1_base_XYPhi_joint",
            "h2_base_XYPhi_joint",
        ):
            indices = [
                index
                for index, joint_name in enumerate(joint_names)
                if (
                    joint_name == base_name
                    or joint_name.startswith(f"{base_name}:")
                )
            ]
            if indices:
                base_indices[base_name] = indices
                summary[f"{base_name}_travel"] = 0.0

        support_grippers = {
            event.parent
            for record in records
            for event in record.events
            if event.parent.startswith("h")
        }

        for support_gripper in support_grippers:
            arm_name = support_gripper.removesuffix(
                "_ur_gripper_center"
            )
            robot_prefix = support_gripper.split("_", maxsplit=1)[0]
            base_joint = f"{robot_prefix}_base_XYPhi_joint"
            arm_prefix = f"{arm_name}_"

            indices = [
                index
                for index, joint_name in enumerate(joint_names)
                if (
                    joint_name == base_joint
                    or joint_name.startswith(f"{base_joint}:")
                    or joint_name.startswith(arm_prefix)
                )
            ]

            if indices:
                support_indices_by_gripper[support_gripper] = indices

    active_supports = set()

    for record in records:
        summary["motion_segments"] += len(record.segments)

        pre_events = [
            event
            for event in record.events
            if event.segment_id == -1
        ]

        for event in pre_events:
            if event.parent.startswith("h"):
                if event.action == "attach":
                    active_supports.add(event.parent)
                elif event.action == "detach":
                    active_supports.discard(event.parent)

        for event in record.events:
            is_support = event.parent.startswith("h")
            is_main = event.parent.startswith("a")

            if is_support and event.action == "attach":
                summary["support_add_events"] += 1
            elif is_support and event.action == "detach":
                summary["support_release_events"] += 1
            elif is_main and event.action == "attach":
                summary["main_attach_events"] += 1
            elif is_main and event.action == "detach":
                summary["main_detach_events"] += 1

        for segment_id, segment in enumerate(record.segments):
            path = np.asarray(segment, dtype=float)
            summary["path_points"] += len(path)
            summary["path_cost"] += path_cost(path)

            for support_gripper in active_supports:
                indices = support_indices_by_gripper.get(
                    support_gripper,
                    [],
                )

                if path.ndim == 2 and path.shape[0] > 1 and indices:
                    drift = float(
                        np.max(
                            np.linalg.norm(
                                path[:, indices] - path[0, indices],
                                axis=1,
                            )
                        )
                    )
                    summary["locked_support_drift_total"] += drift
                    summary["locked_support_drift_max"] = max(
                        summary["locked_support_drift_max"],
                        drift,
                    )

            for base_name, indices in base_indices.items():
                if path.ndim == 2 and path.shape[0] > 1:
                    base_path = path[:, indices]
                    summary[f"{base_name}_travel"] += path_cost(base_path)

            segment_events = [
                event
                for event in record.events
                if event.segment_id == segment_id
            ]

            for event in segment_events:
                if not event.parent.startswith("h"):
                    continue

                if event.action == "attach":
                    active_supports.add(event.parent)
                elif event.action == "detach":
                    active_supports.discard(event.parent)

    return summary


def structural_summary(searcher) -> dict[str, Any]:
    cache_info = searcher.rigidity.cache_info()
    structural_steps = getattr(searcher.final_node, "structural_steps", [])

    summary = {
        "search_stop_reason": searcher.search_stop_reason,
        "search_expansions": searcher.search_expansions,
        "search_attempted_transitions": getattr(
            searcher,
            "search_attempted_transitions",
            0,
        ),
        "search_enqueued_candidates": getattr(
            searcher,
            "search_enqueued_candidates",
            0,
        ),
        "rigidity_check_calls": cache_info["check_calls"],
        "rigidity_cache_hits": cache_info["cache_hits"],
        "rigidity_cache_misses": cache_info["cache_misses"],
        "rigidity_cached_entries": cache_info["cached_entries"],
    }
    summary.update(support_summary(structural_steps))
    return summary


def support_summary(structural_steps) -> dict[str, Any]:
    peak_supports = max(
        (len(step.supports_after) for step in structural_steps),
        default=0,
    )

    support_steps = sum(
        len(step.supports_after)
        for step in structural_steps
    )

    supported_rod_steps = sum(
        len(set(step.supports_after.values()))
        for step in structural_steps
    )

    support_assignment_episodes = 0
    supported_rod_episodes = 0
    previous_assignments = set()
    previous_supported_rods = set()

    for step in structural_steps:
        assignments = set(step.supports_after.items())
        supported_rods = set(step.supports_after.values())

        support_assignment_episodes += len(
            assignments - previous_assignments
        )
        supported_rod_episodes += len(
            supported_rods - previous_supported_rods
        )

        previous_assignments = assignments
        previous_supported_rods = supported_rods

    support_additions = sum(
        len(step.added_supports)
        for step in structural_steps
    )

    support_releases = sum(
        len(step.released_supports)
        for step in structural_steps
    )

    return {
        "peak_supports": peak_supports,
        "support_steps": support_steps,
        "supported_rod_steps": supported_rod_steps,
        "support_assignment_episodes": support_assignment_episodes,
        "supported_rod_episodes": supported_rod_episodes,
        "support_additions": support_additions,
        "support_releases": support_releases,
    }
