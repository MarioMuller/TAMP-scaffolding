#!/usr/bin/env python3
"""Analyze one-arm RAI pose benchmarks.

Reliability counts include every recorded run. Runtime, search effort, pose
validation, and support-use statistics are calculated from successful runs
only, so failed runs cannot appear as artificial zero-support solutions.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable


# Replan traces can make an individual benchmark CSV field much larger than
# the csv module's conservative 128 KiB default.
csv.field_size_limit(sys.maxsize)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "results"
    / "evaluation"
    / "legacy"
    / "single_arm_pose_support_fractions_0.5_0.6"
)
OUTPUT_DIR = Path(__file__).resolve().parent
DEFAULT_SUMMARY_CSV = OUTPUT_DIR / "one_arm_pose_summary.csv"
DEFAULT_RUNS_CSV = OUTPUT_DIR / "one_arm_pose_runs.csv"
DEFAULT_MARKDOWN = OUTPUT_DIR / "one_arm_pose_table.md"
DEFAULT_PNG = OUTPUT_DIR / "one_arm_pose_table.png"

IGNORED_STRATEGIES = {
    "fast_reduce_support_moves",
}

REQUIRED_COLUMNS = {
    "strategy",
    "repeat",
    "seed",
    "success",
    "benchmark_stop_reason",
    "total_time_s",
    "pose_validation_time_s",
    "structural_time_s",
    "structural_plans_generated",
    "structural_replans",
    "rai_transition_attempts",
    "rai_transition_failures",
    "rai_transition_successes",
    "komo_calls",
    "komo_feasible",
    "komo_infeasible",
    "first_plan_support_moves",
    "first_plan_support_steps",
    "support_moves",
    "support_steps",
    "first_to_final_exact_match",
    "first_to_final_common_prefix_length",
    "first_to_final_changed_positions",
}

SUCCESS_METRICS = (
    "total_time_s",
    "pose_validation_time_s",
    "structural_time_s",
    "rai_robot_import_time_s",
    "rai_robot_template_restore_time_s",
    "structural_plans_generated",
    "structural_replans",
    "rai_transition_attempts",
    "rai_transition_failures",
    "rai_transition_successes",
    "komo_calls",
    "komo_feasible",
    "komo_infeasible",
    "first_plan_support_moves",
    "first_plan_support_steps",
    "support_moves",
    "support_steps",
    "first_to_final_common_prefix_length",
    "first_to_final_changed_positions",
)

RUN_COLUMNS = (
    "strategy",
    "repeat",
    "seed",
    "success",
    "benchmark_stop_reason",
    "total_time_s",
    "pose_validation_time_s",
    "structural_time_s",
    "rai_robot_import_time_s",
    "rai_robot_template_restore_time_s",
    "rai_builder_setup_time_s",
    "rai_runtime_time_s",
    "rai_runtime_percent",
    "backward_search_runtime_percent",
    "other_runtime_percent",
    "structural_plans_generated",
    "structural_replans",
    "rai_transition_attempts",
    "rai_transition_failures",
    "komo_calls",
    "komo_feasible",
    "komo_infeasible",
    "first_plan_support_moves",
    "first_plan_support_steps",
    "support_moves",
    "support_steps",
    "first_to_final_exact_match",
    "first_to_final_common_prefix_length",
    "first_to_final_changed_positions",
    "source_file",
)

STRATEGY_ORDER = {
    strategy: index
    for index, strategy in enumerate(
        (
            "fast_reduce_support",
            "baseline",
            "fewest_mandatory_supports",
            "highest_first",
            "default",
        )
    )
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze one-arm pose benchmarks. Numerical summaries use "
            "successful runs only; failure counts use all runs."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Directory searched recursively for benchmark.csv files.",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=DEFAULT_SUMMARY_CSV,
    )
    parser.add_argument(
        "--runs-csv",
        type=Path,
        default=DEFAULT_RUNS_CSV,
    )
    parser.add_argument(
        "--markdown",
        type=Path,
        default=DEFAULT_MARKDOWN,
    )
    parser.add_argument(
        "--png",
        type=Path,
        default=DEFAULT_PNG,
    )
    parser.add_argument(
        "--exclude-strategy",
        action="append",
        default=[],
        help="Strategy to omit; may be supplied more than once.",
    )
    return parser.parse_args()


def parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise ValueError(f"Cannot parse Boolean value: {value!r}")


def number(row: dict[str, str], column: str) -> float:
    value = row.get(column, "").strip()
    return float(value) if value else math.nan


def finite(values: Iterable[float]) -> list[float]:
    return [value for value in values if math.isfinite(value)]


def distribution(values: Iterable[float]) -> dict[str, float]:
    usable = finite(values)
    return {
        "mean": statistics.mean(usable) if usable else math.nan,
        "median": statistics.median(usable) if usable else math.nan,
        "std": statistics.stdev(usable) if len(usable) >= 2 else math.nan,
        "min": min(usable) if usable else math.nan,
        "max": max(usable) if usable else math.nan,
    }


def ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else math.nan


def runtime_percent(row: dict[str, str], column: str) -> float:
    return 100.0 * ratio(number(row, column), number(row, "total_time_s"))


def optional_number(row: dict[str, str], column: str) -> float:
    value = number(row, column)
    return value if math.isfinite(value) else 0.0


def rai_builder_setup_time(row: dict[str, str]) -> float:
    return (
        optional_number(row, "rai_robot_import_time_s")
        + optional_number(row, "rai_robot_template_restore_time_s")
    )


def rai_runtime_time(row: dict[str, str]) -> float:
    return number(row, "pose_validation_time_s") + rai_builder_setup_time(row)


def value_runtime_percent(row: dict[str, str], value: float) -> float:
    return 100.0 * ratio(value, number(row, "total_time_s"))


def other_runtime_percent(row: dict[str, str]) -> float:
    total = number(row, "total_time_s")
    measured = (
        rai_runtime_time(row)
        + number(row, "structural_time_s")
    )
    return 100.0 * ratio(max(0.0, total - measured), total)


def strategy_sort_key(strategy: str) -> tuple[int, str]:
    return STRATEGY_ORDER.get(strategy, len(STRATEGY_ORDER)), strategy


def load_rows(input_dir: Path) -> list[dict[str, str]]:
    csv_paths = sorted(input_dir.rglob("benchmark.csv"))
    if not csv_paths:
        raise FileNotFoundError(
            f"No benchmark.csv files found below {input_dir}."
        )

    rows_by_key: dict[tuple[str, int, int], dict[str, str]] = {}

    for csv_path in csv_paths:
        with csv_path.open(newline="", encoding="utf-8") as csv_file:
            reader = csv.DictReader(csv_file)
            columns = set(reader.fieldnames or ())
            missing = REQUIRED_COLUMNS - columns
            if missing:
                raise ValueError(
                    f"{csv_path} is missing columns: "
                    f"{', '.join(sorted(missing))}"
                )

            for row in reader:
                key = (
                    row["strategy"],
                    int(row["repeat"]),
                    int(row["seed"]),
                )
                if key in rows_by_key:
                    raise ValueError(
                        "Duplicate strategy/repeat/seed record: "
                        f"{key} in {csv_path}."
                    )
                row["source_file"] = str(csv_path)
                rows_by_key[key] = row

    return list(rows_by_key.values())


def validate_configs(
    input_dir: Path,
    expected_main_arm_count: int,
) -> list[dict[str, object]]:
    configs = []
    for config_path in sorted(input_dir.rglob("benchmark_config.json")):
        with config_path.open(encoding="utf-8") as config_file:
            config = json.load(config_file)

        if config.get("main_robot_arm_count") != expected_main_arm_count:
            raise ValueError(
                f"{config_path} does not use "
                f"{expected_main_arm_count} main robot arm(s)."
            )
        if not config.get("use_rai"):
            raise ValueError(f"{config_path} does not enable RAI.")

        configs.append(config)

    if not configs:
        raise FileNotFoundError(
            f"No benchmark_config.json files found below {input_dir}."
        )

    comparable_keys = (
        "main_robot_arm_count",
        "max_replans",
        "max_runtime",
        "max_total_runtime",
        "require_connected_supports",
        "shuffle_ties",
        "support_fractions",
        "support_grippers",
        "support_target_order",
        "truss",
        "use_rai",
        "use_rrt",
        "use_ssik_initialization",
    )
    reference = {key: configs[0].get(key) for key in comparable_keys}
    for config in configs[1:]:
        candidate = {key: config.get(key) for key in comparable_keys}
        if candidate != reference:
            raise ValueError(
                "Benchmark configurations differ in fields other than seed."
            )

    return configs


def summarize_strategy(
    strategy: str,
    rows: list[dict[str, str]],
) -> dict[str, object]:
    successful = [row for row in rows if parse_bool(row["success"])]
    failed = [row for row in rows if not parse_bool(row["success"])]
    stop_reasons = Counter(row["benchmark_stop_reason"] for row in failed)

    summary: dict[str, object] = {
        "strategy": strategy,
        "runs_total": len(rows),
        "successful_runs": len(successful),
        "failed_runs": len(failed),
        "success_rate": ratio(len(successful), len(rows)),
        "max_replans_failures": stop_reasons.get("max_replans", 0),
        "max_total_runtime_failures": stop_reasons.get(
            "max_total_runtime",
            0,
        ),
        "other_failures": (
            len(failed)
            - stop_reasons.get("max_replans", 0)
            - stop_reasons.get("max_total_runtime", 0)
        ),
        "successful_zero_replan_runs": sum(
            number(row, "structural_replans") == 0
            for row in successful
        ),
        "successful_first_plan_exact_matches": sum(
            bool(row["first_to_final_exact_match"].strip())
            and parse_bool(row["first_to_final_exact_match"])
            for row in successful
        ),
    }
    summary["successful_zero_replan_rate"] = ratio(
        int(summary["successful_zero_replan_runs"]),
        len(successful),
    )
    summary["successful_first_plan_exact_match_rate"] = ratio(
        int(summary["successful_first_plan_exact_matches"]),
        len(successful),
    )

    successful_total_time = sum(
        number(row, "total_time_s") for row in successful
    )
    successful_rai_time = sum(rai_runtime_time(row) for row in successful)
    successful_backward_search_time = sum(
        number(row, "structural_time_s") for row in successful
    )
    summary["rai_runtime_share_successful_percent"] = 100.0 * ratio(
        successful_rai_time,
        successful_total_time,
    )
    summary["backward_search_runtime_share_successful_percent"] = (
        100.0
        * ratio(
            successful_backward_search_time,
            successful_total_time,
        )
    )
    summary["other_runtime_share_successful_percent"] = 100.0 * ratio(
        max(
            0.0,
            successful_total_time
            - successful_rai_time
            - successful_backward_search_time,
        ),
        successful_total_time,
    )

    for metric in SUCCESS_METRICS:
        stats = distribution(number(row, metric) for row in successful)
        for statistic_name, value in stats.items():
            summary[f"{metric}_successful_{statistic_name}"] = value

    derived_runtime_metrics = {
        "rai_runtime_percent": (
            value_runtime_percent(row, rai_runtime_time(row))
            for row in successful
        ),
        "backward_search_runtime_percent": (
            runtime_percent(row, "structural_time_s")
            for row in successful
        ),
        "other_runtime_percent": (
            other_runtime_percent(row) for row in successful
        ),
    }
    for metric, values in derived_runtime_metrics.items():
        stats = distribution(values)
        for statistic_name, value in stats.items():
            summary[f"{metric}_successful_{statistic_name}"] = value

    return summary


def write_csv(
    path: Path,
    rows: list[dict[str, object]],
    fieldnames: Iterable[str] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(fieldnames or rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def format_number(value: object, digits: int = 2) -> str:
    numeric = float(value)
    if not math.isfinite(numeric):
        return "n.a."
    return f"{numeric:.{digits}f}"


def strategy_label(strategy: str) -> str:
    return strategy.replace("_", " ")


def compact_table_rows(
    summaries: list[dict[str, object]],
) -> tuple[list[str], list[list[str]]]:
    headers = [
        "Strategy",
        "Success",
        "Runtime mean [s]",
        "Runtime median [s]",
        "RAI total [%]",
        "Backward search [%]",
        "Unclassified [%]",
        "Replans mean",
        "Replans median",
        "Support moves median",
        "Support steps median",
    ]
    table_rows = []
    for summary in summaries:
        table_rows.append([
            strategy_label(str(summary["strategy"])),
            (
                f"{summary['successful_runs']}/"
                f"{summary['runs_total']} "
                f"({100.0 * float(summary['success_rate']):.0f}%)"
            ),
            format_number(summary["total_time_s_successful_mean"]),
            format_number(summary["total_time_s_successful_median"]),
            format_number(
                summary["rai_runtime_percent_successful_mean"],
                2,
            ),
            format_number(
                summary["backward_search_runtime_percent_successful_mean"],
                2,
            ),
            format_number(
                summary["other_runtime_percent_successful_mean"],
                2,
            ),
            format_number(
                summary["structural_replans_successful_mean"],
                1,
            ),
            format_number(
                summary["structural_replans_successful_median"],
                1,
            ),
            format_number(
                summary["support_moves_successful_median"],
                1,
            ),
            format_number(
                summary["support_steps_successful_median"],
                1,
            ),
        ])
    return headers, table_rows


def write_markdown(
    path: Path,
    summaries: list[dict[str, object]],
    support_fractions: str,
    benchmark_title: str,
) -> None:
    headers, table_rows = compact_table_rows(summaries)
    lines = [
        f"# {benchmark_title}",
        "",
        f"Support fractions: `{support_fractions}`",
        "",
        (
            "All numerical statistics use successful runs only. Failure "
            "counts and success rates use all recorded runs."
        ),
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend(
        "| " + " | ".join(row) + " |"
        for row in table_rows
    )
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_png(
    path: Path,
    summaries: list[dict[str, object]],
    support_fractions: str,
    benchmark_title: str,
) -> None:
    import matplotlib.pyplot as plt

    headers, table_rows = compact_table_rows(summaries)
    png_headers = [
        header.replace(" ", "\n", 1)
        if header not in {"Strategy", "Success"}
        else header
        for header in headers
    ]
    figure_height = max(3.2, 1.5 + 0.55 * len(table_rows))
    figure, axis = plt.subplots(figsize=(22.0, figure_height))
    axis.axis("off")
    axis.set_title(
        f"{benchmark_title} (support fractions {support_fractions})",
        fontsize=18,
        fontweight="bold",
        pad=16,
    )
    table = axis.table(
        cellText=table_rows,
        colLabels=png_headers,
        cellLoc="center",
        colLoc="center",
        loc="center",
        colWidths=[
            0.18, 0.08, 0.10, 0.10, 0.09,
            0.10, 0.08, 0.08, 0.08, 0.10, 0.10,
        ],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.0, 1.65)

    for (row_index, _), cell in table.get_celld().items():
        cell.set_edgecolor("#B8BEC7")
        if row_index == 0:
            cell.set_facecolor("#2E4057")
            cell.set_text_props(color="white", fontweight="bold")
        elif row_index % 2 == 0:
            cell.set_facecolor("#EEF2F5")
        else:
            cell.set_facecolor("white")

    figure.text(
        0.5,
        0.03,
        (
            "Means and runtime shares give each successful run equal weight. "
            "RAI total includes builder setup and pose validation."
        ),
        ha="center",
        fontsize=11,
        color="#4F5B66",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def run_analysis(
    args: argparse.Namespace,
    *,
    expected_main_arm_count: int,
    benchmark_title: str,
) -> None:
    configs = validate_configs(
        args.input_dir,
        expected_main_arm_count,
    )
    rows = load_rows(args.input_dir)
    excluded = IGNORED_STRATEGIES | set(args.exclude_strategy)
    rows = [row for row in rows if row["strategy"] not in excluded]
    if not rows:
        raise ValueError("No benchmark rows remain after filtering.")

    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row["strategy"], []).append(row)

    summaries = [
        summarize_strategy(strategy, grouped[strategy])
        for strategy in sorted(grouped, key=strategy_sort_key)
    ]
    write_csv(args.summary_csv, summaries)

    run_rows = []
    for row in sorted(
        rows,
        key=lambda row: (
            strategy_sort_key(row["strategy"]),
            int(row["seed"]),
            int(row["repeat"]),
        ),
    ):
        run_row = {column: row.get(column, "") for column in RUN_COLUMNS}
        run_row.update(
            {
                "rai_builder_setup_time_s": rai_builder_setup_time(row),
                "rai_runtime_time_s": rai_runtime_time(row),
                "rai_runtime_percent": value_runtime_percent(
                    row,
                    rai_runtime_time(row),
                ),
                "backward_search_runtime_percent": runtime_percent(
                    row,
                    "structural_time_s",
                ),
                "other_runtime_percent": other_runtime_percent(row),
            }
        )
        run_rows.append(run_row)
    write_csv(args.runs_csv, run_rows, RUN_COLUMNS)

    support_fractions = str(configs[0].get("support_fractions", "unknown"))
    write_markdown(
        args.markdown,
        summaries,
        support_fractions,
        benchmark_title,
    )
    write_png(
        args.png,
        summaries,
        support_fractions,
        benchmark_title,
    )

    print(f"Loaded {len(rows)} runs from {args.input_dir}")
    print(f"Summary CSV: {args.summary_csv}")
    print(f"Per-run CSV: {args.runs_csv}")
    print(f"Markdown: {args.markdown}")
    print(f"PNG: {args.png}")


def main() -> None:
    run_analysis(
        parse_args(),
        expected_main_arm_count=1,
        benchmark_title="One-arm pose benchmark",
    )


if __name__ == "__main__":
    main()
