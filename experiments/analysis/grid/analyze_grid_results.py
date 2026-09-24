#!/usr/bin/env python3
"""Summarize the rigidity strategy/support-target grid experiments.

The detailed CSV contains one row per strategy/target-policy combination.
All calculated runtime, search-effort, rigidity, and support-use metrics use
successful runs only. Failed runs remain visible through reliability counts.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
import statistics
from collections import Counter
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[3]
REPORT_RESULT_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "results"
    / "report"
    / "rigidity_target_grid_10reps"
)
DEFAULT_INPUT_DIR = REPORT_RESULT_DIR
DEFAULT_OUTPUT_CSV = REPORT_RESULT_DIR / "analysis" / "grid_comparison_metrics.csv"
DEFAULT_OUTPUT_MARKDOWN = REPORT_RESULT_DIR / "analysis" / "grid_comparison_table.md"
DEFAULT_OUTPUT_PNG = REPORT_RESULT_DIR / "analysis" / "grid_comparison_table.png"
DEFAULT_SCATTER_PNG = (
    REPORT_RESULT_DIR / "analysis" / "support_tradeoff.png"
)
DEFAULT_SCATTER_PDF = (
    REPORT_RESULT_DIR / "analysis" / "support_tradeoff.pdf"
)

REQUIRED_COLUMNS = {
    "seed",
    "included_rod_count",
    "success",
    "elapsed_s",
    "removal_sequence",
    "search_stop_reason",
    "search_expansions",
    "search_attempted_transitions",
    "search_enqueued_candidates",
    "search_backtracks",
    "rigidity_check_calls",
    "rigidity_cache_hits",
    "rigidity_cache_misses",
    "rigidity_cached_entries",
    "rigidity_incremental_support_updates",
    "rigidity_incremental_support_cache_hits",
    "rigidity_incremental_support_fallbacks",
    "support_target_order",
    "require_connected_supports",
    "peak_supports",
    "support_steps",
    "support_moves",
    "supported_rod_steps",
    "support_assignment_episodes",
    "supported_rod_episodes",
    "support_additions",
    "support_releases",
}

SEARCH_METRICS = (
    "search_expansions",
    "search_attempted_transitions",
    "search_enqueued_candidates",
    "search_backtracks",
    "rigidity_check_calls",
    "rigidity_cache_hits",
    "rigidity_cache_misses",
    "rigidity_cached_entries",
    "rigidity_incremental_support_updates",
    "rigidity_incremental_support_cache_hits",
    "rigidity_incremental_support_fallbacks",
)

SUPPORT_METRICS = (
    "peak_supports",
    "support_steps",
    "support_moves",
    "supported_rod_steps",
    "support_assignment_episodes",
    "supported_rod_episodes",
    "support_additions",
    "support_releases",
)

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
TARGET_ORDER = {
    name: index
    for index, name in enumerate(
        (
            "lowest_first",
            "least_connected",
            "random",
            "furthest_from_removed",
        )
    )
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Calculate comparison metrics for the rigidity strategy/support-"
            "target grid."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Directory containing grid_*.csv files.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_OUTPUT_CSV,
        help="Detailed machine-readable comparison table.",
    )
    parser.add_argument(
        "--output-markdown",
        type=Path,
        default=DEFAULT_OUTPUT_MARKDOWN,
        help="Compact table for reports and quick inspection.",
    )
    parser.add_argument(
        "--output-png",
        type=Path,
        default=DEFAULT_OUTPUT_PNG,
        help="PNG rendering of the compact comparison table.",
    )
    parser.add_argument(
        "--scatter-png",
        type=Path,
        default=DEFAULT_SCATTER_PNG,
        help="PNG rendering of the support-move/support-step trade-off.",
    )
    parser.add_argument(
        "--scatter-pdf",
        type=Path,
        default=DEFAULT_SCATTER_PDF,
        help="Vector PDF of the support-move/support-step trade-off.",
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
    value = row[column].strip()
    if not value:
        return math.nan
    return float(value)


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


def safe_ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else math.nan


def add_distribution(
    summary: dict[str, object],
    prefix: str,
    values: Iterable[float],
    *,
    include_total: bool = False,
) -> None:
    usable = finite(values)
    summary[f"{prefix}_mean"] = mean(usable)
    summary[f"{prefix}_median"] = median(usable)
    summary[f"{prefix}_std"] = sample_std(usable)
    summary[f"{prefix}_min"] = minimum(usable)
    summary[f"{prefix}_max"] = maximum(usable)
    if include_total:
        summary[f"{prefix}_total"] = sum(usable)


def strategy_from_filename(csv_path: Path, target_order: str) -> str:
    experiment_name = csv_path.stem.removeprefix("grid_")
    experiment_name = re.sub(r"_\d+reps$", "", experiment_name)
    target_suffix = f"_{target_order}"
    if not experiment_name.endswith(target_suffix):
        raise ValueError(
            f"Cannot identify strategy in {csv_path.name}; expected filename "
            f"to end with {target_suffix!r} before the repetition suffix."
        )
    return experiment_name[: -len(target_suffix)]


def load_rows(csv_path: Path) -> list[dict[str, str]]:
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
    return rows


def summarize_file(csv_path: Path) -> dict[str, object]:
    rows = load_rows(csv_path)
    target_orders = {row["support_target_order"] for row in rows}
    if len(target_orders) != 1:
        raise ValueError(
            f"{csv_path} contains multiple support-target policies: "
            f"{sorted(target_orders)}"
        )

    target_order = target_orders.pop()
    strategy = strategy_from_filename(csv_path, target_order)
    successful = [row for row in rows if parse_bool(row["success"])]
    failed = [row for row in rows if not parse_bool(row["success"])]
    stop_reasons = Counter(row["search_stop_reason"] for row in failed)
    seeds = [int(row["seed"]) for row in rows]
    rod_counts = [int(row["included_rod_count"]) for row in rows]
    connected_values = {
        parse_bool(row["require_connected_supports"]) for row in rows
    }

    summary: dict[str, object] = {
        "removal_strategy": strategy,
        "support_target_order": target_order,
        "source_file": csv_path.name,
        "runs_total": len(rows),
        "successful_runs": len(successful),
        "failed_runs": len(failed),
        "success_rate": safe_ratio(len(successful), len(rows)),
        "runtime_limit_failures": stop_reasons.get("runtime_limit", 0),
        "open_list_exhausted_failures": stop_reasons.get(
            "open_list_exhausted", 0
        ),
        "other_failures": len(failed)
        - stop_reasons.get("runtime_limit", 0)
        - stop_reasons.get("open_list_exhausted", 0),
        "seed_min": min(seeds),
        "seed_max": max(seeds),
        "included_rod_count_min": min(rod_counts),
        "included_rod_count_max": max(rod_counts),
        "require_connected_supports": (
            connected_values.pop() if len(connected_values) == 1 else "mixed"
        ),
    }

    add_distribution(
        summary,
        "runtime_s_successful_runs",
        (number(row, "elapsed_s") for row in successful),
        include_total=True,
    )

    for metric in SEARCH_METRICS:
        add_distribution(
            summary,
            f"{metric}_successful_runs",
            (number(row, metric) for row in successful),
        )

    attempted = sum(
        number(row, "search_attempted_transitions") for row in successful
    )
    backtracks = sum(number(row, "search_backtracks") for row in successful)
    checks = sum(number(row, "rigidity_check_calls") for row in successful)
    cache_hits = sum(number(row, "rigidity_cache_hits") for row in successful)
    incremental_updates = sum(
        number(row, "rigidity_incremental_support_updates")
        for row in successful
    )
    incremental_hits = sum(
        number(row, "rigidity_incremental_support_cache_hits")
        for row in successful
    )
    incremental_fallbacks = sum(
        number(row, "rigidity_incremental_support_fallbacks")
        for row in successful
    )
    incremental_attempts = (
        incremental_updates + incremental_hits + incremental_fallbacks
    )

    summary["backtrack_rate_successful_runs"] = safe_ratio(
        backtracks, attempted
    )
    summary["rigidity_cache_hit_rate_successful_runs"] = safe_ratio(
        cache_hits, checks
    )
    summary["incremental_support_cache_hit_rate_successful_runs"] = safe_ratio(
        incremental_hits, incremental_attempts
    )
    summary["incremental_support_fallback_rate_successful_runs"] = safe_ratio(
        incremental_fallbacks, incremental_attempts
    )

    for metric in SUPPORT_METRICS:
        add_distribution(
            summary,
            f"{metric}_successful_runs",
            (number(row, metric) for row in successful),
        )

    sequences = Counter(
        row["removal_sequence"] for row in successful if row["removal_sequence"]
    )
    dominant_sequence_count = max(sequences.values(), default=0)
    summary["unique_successful_removal_sequences"] = len(sequences)
    summary["dominant_sequence_count"] = dominant_sequence_count
    summary["dominant_sequence_share"] = safe_ratio(
        dominant_sequence_count, len(successful)
    )

    return summary


def sort_key(summary: dict[str, object]) -> tuple[int, str, int, str]:
    strategy = str(summary["removal_strategy"])
    target = str(summary["support_target_order"])
    return (
        STRATEGY_ORDER.get(strategy, len(STRATEGY_ORDER)),
        strategy,
        TARGET_ORDER.get(target, len(TARGET_ORDER)),
        target,
    )


def write_csv(summaries: list[dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(summaries[0])
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summaries)


def display_name(value: object) -> str:
    return str(value).replace("_", " ")


def format_number(value: object, decimals: int = 2) -> str:
    numeric = float(value)
    return "n.a." if not math.isfinite(numeric) else f"{numeric:.{decimals}f}"


def write_markdown(
    summaries: list[dict[str, object]], output_path: Path
) -> None:
    lines = [
        "# Rigidity Grid Comparison",
        "",
        (
            "All calculated metrics use successful runs only. Success reports "
            "completed runs out of all attempted runs."
        ),
        "",
        (
            "| Removal strategy | Support target | Success | Mean runtime "
            "successful runs (s) | Median successful runtime (s) | Mean support "
            "moves | Mean support steps | Mean expansions | Mean rigidity "
            "checks |"
        ),
        (
            "|---|---|---:|---:|---:|---:|---:|---:|---:|"
        ),
    ]

    for summary in summaries:
        success = f"{summary['successful_runs']}/{summary['runs_total']}"
        lines.append(
            "| "
            + " | ".join(
                (
                    display_name(summary["removal_strategy"]),
                    display_name(summary["support_target_order"]),
                    success,
                    format_number(summary["runtime_s_successful_runs_mean"]),
                    format_number(
                        summary["runtime_s_successful_runs_median"]
                    ),
                    format_number(
                        summary["support_moves_successful_runs_mean"]
                    ),
                    format_number(
                        summary["support_steps_successful_runs_mean"]
                    ),
                    format_number(
                        summary["search_expansions_successful_runs_mean"], 1
                    ),
                    format_number(
                        summary["rigidity_check_calls_successful_runs_mean"], 1
                    ),
                )
            )
            + " |"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_png(summaries: list[dict[str, object]], output_path: Path) -> None:
    mpl_config_dir = Path("/tmp/tamp_scaffolding_matplotlib")
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))

    import matplotlib.pyplot as plt

    headers = (
        "Removal\nstrategy",
        "Support\ntarget",
        "Success",
        "Mean runtime,\nsuccesses (s)",
        "Median runtime,\nsuccesses (s)",
        "Mean support\nmoves",
        "Mean support\nsteps",
        "Mean\nexpansions",
        "Mean rigidity\nchecks",
    )
    table_rows = []
    for summary in summaries:
        table_rows.append(
            (
                display_name(summary["removal_strategy"]),
                display_name(summary["support_target_order"]),
                f"{summary['successful_runs']}/{summary['runs_total']}",
                format_number(summary["runtime_s_successful_runs_mean"]),
                format_number(summary["runtime_s_successful_runs_median"]),
                format_number(
                    summary["support_moves_successful_runs_mean"]
                ),
                format_number(
                    summary["support_steps_successful_runs_mean"]
                ),
                format_number(
                    summary["search_expansions_successful_runs_mean"], 1
                ),
                format_number(
                    summary["rigidity_check_calls_successful_runs_mean"], 1
                ),
            )
        )

    fig, ax = plt.subplots(figsize=(19, 11.5), constrained_layout=False)
    fig.patch.set_facecolor("white")
    ax.set_axis_off()
    fig.text(
        0.03,
        0.965,
        "Rigidity search grid comparison",
        fontsize=21,
        fontweight="bold",
        color="#202124",
        ha="left",
        va="top",
    )
    fig.text(
        0.03,
        0.93,
        (
            "All calculated metrics use successful runs only; success reports "
            "completed runs out of all attempts."
        ),
        fontsize=12,
        color="#5f6368",
        ha="left",
        va="top",
    )

    table = ax.table(
        cellText=table_rows,
        colLabels=headers,
        cellLoc="center",
        colLoc="center",
        colWidths=(
            0.17,
            0.15,
            0.065,
            0.105,
            0.12,
            0.10,
            0.10,
            0.09,
            0.10,
        ),
        bbox=(0.015, 0.015, 0.97, 0.88),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10.5)

    for (row_index, column_index), cell in table.get_celld().items():
        cell.set_edgecolor("#d7dadd")
        cell.set_linewidth(0.6)
        cell.get_text().set_color("#202124")
        if row_index == 0:
            cell.set_facecolor("#343a40")
            cell.get_text().set_color("white")
            cell.get_text().set_fontweight("bold")
            cell.set_height(0.065)
            continue

        cell.set_facecolor("#ffffff" if row_index % 2 else "#f3f5f6")
        if column_index in {0, 1}:
            cell.get_text().set_ha("left")

        if column_index == 2:
            summary = summaries[row_index - 1]
            complete = summary["successful_runs"] == summary["runs_total"]
            cell.set_facecolor("#dcefdc" if complete else "#f5dddd")
            cell.get_text().set_fontweight("bold")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_support_tradeoff(
    summaries: list[dict[str, object]],
    png_path: Path,
    pdf_path: Path,
) -> None:
    mpl_config_dir = Path("/tmp/tamp_scaffolding_matplotlib")
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))

    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    strategy_styles = {
        "default": ("Default", "#0072B2"),
        "highest_first": ("Highest first", "#E69F00"),
        "fewest_mandatory_supports": ("Fewest mandatory", "#009E73"),
        "fast_reduce_support": ("Fast reduce support", "#D55E00"),
        "baseline": ("Baseline", "#CC79A7"),
    }
    target_styles = {
        "random": ("Random", "o"),
        "lowest_first": ("Lowest first", "s"),
        "least_connected": ("Least connected", "^"),
        "furthest_from_removed": ("Furthest", "D"),
    }

    fig, ax = plt.subplots(figsize=(7.1, 4.1), constrained_layout=True)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    plotted_by_strategy: dict[str, list[tuple[float, float]]] = {
        strategy: [] for strategy in strategy_styles
    }
    for summary in summaries:
        strategy = str(summary["removal_strategy"])
        target = str(summary["support_target_order"])
        moves = float(summary["support_moves_successful_runs_mean"])
        steps = float(summary["support_steps_successful_runs_mean"])
        if not (math.isfinite(moves) and math.isfinite(steps)):
            continue

        _, color = strategy_styles[strategy]
        _, marker = target_styles[target]
        ax.scatter(
            moves,
            steps,
            s=68,
            marker=marker,
            facecolor="white",
            edgecolor=color,
            linewidth=1.6,
            zorder=3,
        )
        plotted_by_strategy[strategy].append((moves, steps))

    label_offsets = {
        "default": (8, -2),
        "highest_first": (8, 0),
        "fewest_mandatory_supports": (8, 0),
        "fast_reduce_support": (8, -2),
        "baseline": (8, 0),
    }
    for strategy, points in plotted_by_strategy.items():
        if not points:
            continue
        label, color = strategy_styles[strategy]
        rightmost = max(points, key=lambda point: point[0])
        ax.annotate(
            label,
            xy=rightmost,
            xytext=label_offsets[strategy],
            textcoords="offset points",
            color=color,
            fontsize=12,
            fontweight="semibold",
            va="center",
        )

    target_handles = [
        Line2D(
            [0],
            [0],
            marker=marker,
            linestyle="none",
            markerfacecolor="white",
            markeredgecolor="#4d4d4d",
            markeredgewidth=1.4,
            markersize=8.5,
            label=label,
        )
        for label, marker in target_styles.values()
    ]
    target_legend = ax.legend(
        handles=target_handles,
        loc="upper right",
        ncol=1,
        frameon=False,
        fontsize=12,
        handletextpad=0.4,
        columnspacing=1.2,
    )
    ax.add_artist(target_legend)

    ax.set_xlabel("Mean support moves", fontsize=13)
    ax.set_ylabel("Mean support steps", fontsize=13)
    ax.tick_params(labelsize=11)
    ax.grid(True, color="#d9dde1", linewidth=0.6, alpha=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(0, 80)
    ax.set_ylim(0, 300)

    png_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=300, facecolor="white")
    fig.savefig(pdf_path, facecolor="white")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    csv_paths = sorted(args.input_dir.glob("grid_*.csv"))
    if not csv_paths:
        raise FileNotFoundError(
            f"No grid_*.csv files found in {args.input_dir}"
        )

    summaries = sorted(
        (summarize_file(csv_path) for csv_path in csv_paths),
        key=sort_key,
    )
    write_csv(summaries, args.output_csv)
    write_markdown(summaries, args.output_markdown)
    write_png(summaries, args.output_png)
    write_support_tradeoff(
        summaries,
        args.scatter_png,
        args.scatter_pdf,
    )

    print(f"Analyzed {len(csv_paths)} grid result files.")
    print(f"Detailed metrics: {args.output_csv}")
    print(f"Compact table:   {args.output_markdown}")
    print(f"PNG table:       {args.output_png}")
    print(f"Trade-off PNG:   {args.scatter_png}")
    print(f"Trade-off PDF:   {args.scatter_pdf}")


if __name__ == "__main__":
    main()
