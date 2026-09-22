#!/usr/bin/env python3
"""Analyze and plot successful rigidity scaling runs across strategies."""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "results"
    / "evaluation"
    / "current"
    / "scaling_strategy_comparison_lowest_first_5reps"
)
DEFAULT_OUTPUT_PLOT = Path(__file__).with_name(
    "scaling_runtime_by_scaffold_size.png"
)
DEFAULT_MEAN_OUTPUT_PLOT = Path(__file__).with_name(
    "scaling_runtime_by_scaffold_size_mean.png"
)
DEFAULT_SEPARATE_OUTPUT_PLOT = Path(__file__).with_name(
    "scaling_runtime_by_scaffold_size_separate_repetitions.png"
)
DEFAULT_PREFIX_CSV = Path(__file__).with_name(
    "scaling_prefix_runtime_summary.csv"
)
DEFAULT_STRATEGY_CSV = Path(__file__).with_name(
    "scaling_strategy_summary.csv"
)

REQUIRED_COLUMNS = {
    "repetition",
    "seed",
    "run_index",
    "included_rod_count",
    "success",
    "elapsed_s",
    "search_stop_reason",
    "support_target_order",
}

STRATEGY_ORDER = {
    name: index
    for index, name in enumerate(
        (
            "default",
            "highest_first",
            "fewest_mandatory_supports",
            "fast_reduce_support",
            "baseline",
        )
    )
}

STRATEGY_COLORS = {
    "default": "#2364AA",
    "highest_first": "#F28E2B",
    "fewest_mandatory_supports": "#59A14F",
    "fast_reduce_support": "#D1495B",
    "baseline": "#6C757D",
}


@dataclass(frozen=True)
class StrategyResults:
    strategy: str
    source_file: Path
    rows: list[dict[str, str]]
    repetitions: frozenset[int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Average successful runtime at every scaffold size and compare "
            "all removal strategies in one plot."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Directory containing the all-step scaling CSV files.",
    )
    parser.add_argument(
        "--output-plot",
        type=Path,
        help="Plot path. A mode-specific filename is used by default.",
    )
    parser.add_argument(
        "--output-prefix-csv",
        type=Path,
        default=DEFAULT_PREFIX_CSV,
        help="Runtime and run counts for each strategy/scaffold size.",
    )
    parser.add_argument(
        "--output-strategy-csv",
        type=Path,
        default=DEFAULT_STRATEGY_CSV,
        help="Overall runtime and failure summary for each strategy.",
    )
    parser.add_argument(
        "--linear",
        action="store_true",
        help="Use a linear runtime axis instead of the default log axis.",
    )
    parser.add_argument(
        "--mean",
        action="store_true",
        help="Plot mean successful runtime instead of the default median.",
    )
    parser.add_argument(
        "--seperate",
        "--separate",
        dest="separate_repetitions",
        action="store_true",
        help=(
            "Plot every repetition separately and overlay the mean of "
            "successful runs."
        ),
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the plot after saving it.",
    )
    for strategy in STRATEGY_ORDER:
        parser.add_argument(
            f"--{strategy.replace('_', '-')}",
            dest="excluded_strategies",
            action="append_const",
            const=strategy,
            help=f"Exclude {strategy_label(strategy)} from the analysis.",
        )
    return parser.parse_args()


def parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise ValueError(f"Cannot parse Boolean value: {value!r}")


def finite(values: Iterable[float]) -> list[float]:
    return [value for value in values if math.isfinite(value)]


def mean(values: Iterable[float]) -> float:
    usable = finite(values)
    return statistics.mean(usable) if usable else math.nan


def median(values: Iterable[float]) -> float:
    usable = finite(values)
    return statistics.median(usable) if usable else math.nan


def sample_std(values: Iterable[float]) -> float:
    usable = finite(values)
    return statistics.stdev(usable) if len(usable) >= 2 else math.nan


def minimum(values: Iterable[float]) -> float:
    usable = finite(values)
    return min(usable) if usable else math.nan


def maximum(values: Iterable[float]) -> float:
    usable = finite(values)
    return max(usable) if usable else math.nan


def strategy_label(strategy: str) -> str:
    return strategy.replace("_", " ").title()


def strategy_sort_key(strategy: str) -> tuple[int, str]:
    return STRATEGY_ORDER.get(strategy, len(STRATEGY_ORDER)), strategy


def strategy_from_filename(csv_path: Path, support_target_order: str) -> str:
    experiment_name = re.sub(
        r"_all_steps_\d+reps$",
        "",
        csv_path.stem,
    )
    target_suffix = f"_{support_target_order}"
    if not experiment_name.endswith(target_suffix):
        raise ValueError(
            f"Cannot identify strategy from {csv_path.name}; expected the "
            f"name before '_all_steps' to end in {target_suffix!r}."
        )
    return experiment_name[: -len(target_suffix)]


def load_strategy_results(csv_path: Path) -> StrategyResults:
    with csv_path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        columns = set(reader.fieldnames or ())
        missing = REQUIRED_COLUMNS - columns
        if missing:
            raise ValueError(
                f"{csv_path} is missing columns: {', '.join(sorted(missing))}"
            )
        rows = list(reader)

    if not rows:
        raise ValueError(f"{csv_path} contains no result rows.")

    target_orders = {row["support_target_order"] for row in rows}
    if len(target_orders) != 1:
        raise ValueError(
            f"{csv_path} contains multiple support-target policies: "
            f"{sorted(target_orders)}"
        )

    target_order = target_orders.pop()
    strategy = strategy_from_filename(csv_path, target_order)
    repetitions = frozenset(int(row["repetition"]) for row in rows)
    return StrategyResults(strategy, csv_path, rows, repetitions)


def runtime_distribution(rows: Iterable[dict[str, str]]) -> dict[str, float]:
    runtimes = [float(row["elapsed_s"]) for row in rows]
    return {
        "runtime_successful_mean_s": mean(runtimes),
        "runtime_successful_median_s": median(runtimes),
        "runtime_successful_std_s": sample_std(runtimes),
        "runtime_successful_min_s": minimum(runtimes),
        "runtime_successful_max_s": maximum(runtimes),
    }


def build_prefix_summaries(
    results: list[StrategyResults],
    scaffold_sizes: list[int],
) -> list[dict[str, object]]:
    summaries = []
    for result in sorted(results, key=lambda item: strategy_sort_key(item.strategy)):
        by_size: dict[int, list[dict[str, str]]] = defaultdict(list)
        for row in result.rows:
            by_size[int(row["included_rod_count"])].append(row)

        expected_per_size = len(result.repetitions)
        for scaffold_size in scaffold_sizes:
            rows = by_size.get(scaffold_size, [])
            successful = [row for row in rows if parse_bool(row["success"])]
            failed = [row for row in rows if not parse_bool(row["success"])]
            stop_reasons = Counter(row["search_stop_reason"] for row in failed)
            summary: dict[str, object] = {
                "removal_strategy": result.strategy,
                "included_rod_count": scaffold_size,
                "expected_runs": expected_per_size,
                "recorded_runs": len(rows),
                "successful_runs": len(successful),
                "failed_runs": len(failed),
                "missing_not_attempted_runs": expected_per_size - len(rows),
                "runtime_limit_failures": stop_reasons.get(
                    "runtime_limit", 0
                ),
                "open_list_exhausted_failures": stop_reasons.get(
                    "open_list_exhausted", 0
                ),
            }
            summary.update(runtime_distribution(successful))
            summaries.append(summary)

    return summaries


def build_strategy_summaries(
    results: list[StrategyResults],
    scaffold_size_count: int,
) -> list[dict[str, object]]:
    summaries = []
    for result in sorted(results, key=lambda item: strategy_sort_key(item.strategy)):
        successful = [
            row for row in result.rows if parse_bool(row["success"])
        ]
        failed = [
            row for row in result.rows if not parse_bool(row["success"])
        ]
        stop_reasons = Counter(row["search_stop_reason"] for row in failed)
        expected_runs = len(result.repetitions) * scaffold_size_count
        summary: dict[str, object] = {
            "removal_strategy": result.strategy,
            "source_file": result.source_file.name,
            "repetitions": len(result.repetitions),
            "scaffold_sizes": scaffold_size_count,
            "expected_runs": expected_runs,
            "recorded_runs": len(result.rows),
            "successful_runs": len(successful),
            "failed_runs": len(failed),
            "missing_not_attempted_runs": expected_runs - len(result.rows),
            "runtime_limit_failures": stop_reasons.get("runtime_limit", 0),
            "open_list_exhausted_failures": stop_reasons.get(
                "open_list_exhausted", 0
            ),
            "other_failures": len(failed)
            - stop_reasons.get("runtime_limit", 0)
            - stop_reasons.get("open_list_exhausted", 0),
        }
        summary.update(runtime_distribution(successful))
        summaries.append(summary)

    return summaries


def write_csv(rows: list[dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_scaling(
    prefix_summaries: list[dict[str, object]],
    results: list[StrategyResults],
    scaffold_sizes: list[int],
    output_path: Path,
    *,
    log_scale: bool,
    separate_repetitions: bool,
    use_mean: bool,
    show: bool,
) -> None:
    mpl_config_dir = Path("/tmp/tamp_scaffolding_matplotlib")
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 6.8), constrained_layout=True)
    if separate_repetitions:
        summaries_by_strategy: dict[str, list[dict[str, object]]] = (
            defaultdict(list)
        )
        for summary in prefix_summaries:
            summaries_by_strategy[
                str(summary["removal_strategy"])
            ].append(summary)

        for result in sorted(
            results, key=lambda item: strategy_sort_key(item.strategy)
        ):
            rows_by_repetition: dict[int, dict[int, dict[str, str]]] = (
                defaultdict(dict)
            )
            for row in result.rows:
                repetition = int(row["repetition"])
                scaffold_size = int(row["included_rod_count"])
                rows_by_repetition[repetition][scaffold_size] = row

            for repetition in sorted(result.repetitions):
                repetition_rows = rows_by_repetition[repetition]
                runtimes = []
                for scaffold_size in scaffold_sizes:
                    row = repetition_rows.get(scaffold_size)
                    if row is None or not parse_bool(row["success"]):
                        runtimes.append(math.nan)
                    else:
                        runtimes.append(float(row["elapsed_s"]))

                if not any(math.isfinite(runtime) for runtime in runtimes):
                    continue

                ax.plot(
                    scaffold_sizes,
                    runtimes,
                    color=STRATEGY_COLORS.get(result.strategy),
                    linewidth=1.0,
                    alpha=0.25,
                )

            strategy_summaries = sorted(
                summaries_by_strategy[result.strategy],
                key=lambda row: int(row["included_rod_count"]),
            )
            mean_scaffold_sizes = [
                int(row["included_rod_count"])
                for row in strategy_summaries
            ]
            mean_runtimes = [
                float(row["runtime_successful_mean_s"])
                for row in strategy_summaries
            ]
            if any(math.isfinite(runtime) for runtime in mean_runtimes):
                ax.plot(
                    mean_scaffold_sizes,
                    mean_runtimes,
                    color=STRATEGY_COLORS.get(result.strategy),
                    linewidth=2.6,
                    label=f"{strategy_label(result.strategy)} mean",
                    zorder=3,
                )
    else:
        runtime_metric = (
            "runtime_successful_mean_s"
            if use_mean
            else "runtime_successful_median_s"
        )
        grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
        for summary in prefix_summaries:
            grouped[str(summary["removal_strategy"])].append(summary)

        for strategy in sorted(grouped, key=strategy_sort_key):
            strategy_rows = sorted(
                grouped[strategy],
                key=lambda row: int(row["included_rod_count"]),
            )
            strategy_scaffold_sizes = [
                int(row["included_rod_count"]) for row in strategy_rows
            ]
            runtimes = [float(row[runtime_metric]) for row in strategy_rows]
            if not any(math.isfinite(runtime) for runtime in runtimes):
                continue

            ax.plot(
                strategy_scaffold_sizes,
                runtimes,
                color=STRATEGY_COLORS.get(strategy),
                linewidth=2.0,
                label=strategy_label(strategy),
            )

    if log_scale:
        ax.set_yscale("log")

    ax.set_xlabel("Rods in scaffold")
    if separate_repetitions:
        runtime_label = "Successful runtime"
        plot_note = (
            "Thin lines are individual repetitions; bold lines are means "
            "of successful runs. Failed and skipped points are gaps."
        )
    elif use_mean:
        runtime_label = "Mean successful runtime"
        plot_note = (
            "Each point is the mean of successful runs; failed and skipped "
            "runs are excluded."
        )
    else:
        runtime_label = "Median successful runtime"
        plot_note = (
            "Each point is the median of successful runs; failed and skipped "
            "runs are excluded."
        )
    scale_label = ", log scale" if log_scale else ""
    ax.set_ylabel(f"{runtime_label} (seconds{scale_label})")
    ax.set_title(
        "Individual rigidity scaling runs and means by removal strategy"
        if separate_repetitions
        else "Rigidity search scaling by removal strategy"
    )
    ax.grid(True, which="both", linewidth=0.6, alpha=0.28)
    ax.legend(frameon=False, ncol=2)
    ax.text(
        0.0,
        -0.14,
        plot_note,
        transform=ax.transAxes,
        fontsize=9,
        color="#5f6368",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, facecolor="white")
    if show:
        plt.show()
    plt.close(fig)


def print_strategy_summary(summaries: list[dict[str, object]]) -> None:
    print("\nStrategy summary (runtime uses successful runs only):")
    for summary in summaries:
        average = float(summary["runtime_successful_mean_s"])
        average_text = "n.a." if not math.isfinite(average) else f"{average:.3f}s"
        print(
            f"  {strategy_label(str(summary['removal_strategy']))}: "
            f"average={average_text}, "
            f"successful={summary['successful_runs']}, "
            f"failed={summary['failed_runs']}, "
            f"skipped/not recorded={summary['missing_not_attempted_runs']}"
        )


def main() -> None:
    args = parse_args()
    csv_paths = sorted(args.input_dir.glob("*.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No CSV files found in {args.input_dir}")

    excluded_strategies = set(args.excluded_strategies or ())
    results = [
        result
        for result in (
            load_strategy_results(csv_path) for csv_path in csv_paths
        )
        if result.strategy not in excluded_strategies
    ]
    if not results:
        raise ValueError("All available strategies were excluded.")

    if excluded_strategies:
        print(
            "Excluded strategies: "
            + ", ".join(
                strategy_label(strategy)
                for strategy in sorted(
                    excluded_strategies,
                    key=strategy_sort_key,
                )
            )
        )
    scaffold_sizes = sorted(
        {
            int(row["included_rod_count"])
            for result in results
            for row in result.rows
        }
    )
    prefix_summaries = build_prefix_summaries(results, scaffold_sizes)
    strategy_summaries = build_strategy_summaries(
        results, len(scaffold_sizes)
    )

    write_csv(prefix_summaries, args.output_prefix_csv)
    write_csv(strategy_summaries, args.output_strategy_csv)
    if args.output_plot is not None:
        output_plot = args.output_plot
    elif args.separate_repetitions:
        output_plot = DEFAULT_SEPARATE_OUTPUT_PLOT
    elif args.mean:
        output_plot = DEFAULT_MEAN_OUTPUT_PLOT
    else:
        output_plot = DEFAULT_OUTPUT_PLOT
    plot_scaling(
        prefix_summaries,
        results,
        scaffold_sizes,
        output_plot,
        log_scale=not args.linear,
        separate_repetitions=args.separate_repetitions,
        use_mean=args.mean,
        show=args.show,
    )
    print_strategy_summary(strategy_summaries)
    print(f"\nPrefix summary:   {args.output_prefix_csv}")
    print(f"Strategy summary: {args.output_strategy_csv}")
    print(f"Scaling plot:     {output_plot}")


if __name__ == "__main__":
    main()
