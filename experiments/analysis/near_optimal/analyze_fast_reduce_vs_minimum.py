#!/usr/bin/env python3
"""Compare fast support reduction with the expensive 20-rod references."""

from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[3]
REPORT_RESULT_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "results"
    / "report"
    / "near_optimal_20_rods"
)
DEFAULT_INPUT_DIR = REPORT_RESULT_DIR
DEFAULT_FAST_FILE = "fast_reduce_support_20rods_10reps.csv"
DEFAULT_FAST_FURTHEST_FILE = (
    "fast_reduce_support_furthest_from_removed_20rods_10reps.csv"
)
DEFAULT_DEFAULT_LOWEST_FILE = "default_lowest_first_20rods_10reps.csv"
DEFAULT_DEFAULT_FURTHEST_FILE = (
    "default_furthest_from_removed_20rods_10reps.csv"
)
DEFAULT_RANDOM_FILE = "baseline_random_20rods_10reps.csv"
DEFAULT_BASELINE_LOWEST_FILE = "baseline_lowest_first_20rods_10reps.csv"
DEFAULT_STEP_FILE = "reduced_overall_support_steps_20rods_1rep.csv"
DEFAULT_MOVE_FILE = "reduced_overall_supports_20rods_1rep.csv"
DEFAULT_RUN_OUTPUT = (
    REPORT_RESULT_DIR / "analysis" / "fast_reduce_vs_reference_minima_runs.csv"
)
DEFAULT_SUMMARY_OUTPUT = (
    REPORT_RESULT_DIR / "analysis" / "fast_reduce_vs_reference_minima_summary.csv"
)
DEFAULT_MARKDOWN_OUTPUT = (
    REPORT_RESULT_DIR / "analysis" / "fast_reduce_vs_reference_minima.md"
)
DEFAULT_PNG_OUTPUT = (
    REPORT_RESULT_DIR / "analysis" / "fast_reduce_vs_reference_minima.png"
)

REQUIRED_COLUMNS = {
    "repetition",
    "seed",
    "success",
    "elapsed_s",
    "search_stop_reason",
    "peak_supports",
    "removal_sequence",
    "support_steps",
    "support_moves",
    "support_additions",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare ten fast_reduce_support runs with the support-step and "
            "support-move reference minima from the expensive searches."
        )
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument(
        "--fast-file",
        action="append",
        dest="fast_files",
        help=(
            "Fast-run CSV filename relative to --input-dir. May be supplied "
            "more than once; defaults to the original 10-run file."
        ),
    )
    parser.add_argument("--step-reference-file", default=DEFAULT_STEP_FILE)
    parser.add_argument("--move-reference-file", default=DEFAULT_MOVE_FILE)
    parser.add_argument(
        "--fast-furthest-file",
        default=DEFAULT_FAST_FURTHEST_FILE,
    )
    parser.add_argument(
        "--default-lowest-file",
        default=DEFAULT_DEFAULT_LOWEST_FILE,
    )
    parser.add_argument(
        "--default-furthest-file",
        default=DEFAULT_DEFAULT_FURTHEST_FILE,
    )
    parser.add_argument(
        "--random-comparison-file",
        default=DEFAULT_RANDOM_FILE,
        help=(
            "Optional baseline-removal/random-support CSV filename relative "
            "to --input-dir."
        ),
    )
    parser.add_argument(
        "--baseline-lowest-file",
        default=DEFAULT_BASELINE_LOWEST_FILE,
    )
    parser.add_argument("--output-runs", type=Path, default=DEFAULT_RUN_OUTPUT)
    parser.add_argument(
        "--output-summary",
        type=Path,
        default=DEFAULT_SUMMARY_OUTPUT,
    )
    parser.add_argument(
        "--output-markdown",
        type=Path,
        default=DEFAULT_MARKDOWN_OUTPUT,
    )
    parser.add_argument(
        "--output-png",
        type=Path,
        default=DEFAULT_PNG_OUTPUT,
    )
    return parser.parse_args()


def parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise ValueError(f"Cannot parse Boolean value: {value!r}")


def load_successful_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        columns = set(reader.fieldnames or ())
        missing = REQUIRED_COLUMNS - columns
        if missing:
            raise ValueError(
                f"{path} is missing columns: {', '.join(sorted(missing))}"
            )
        rows = list(reader)

    successful = [row for row in rows if parse_bool(row["success"])]
    if not successful:
        raise ValueError(f"{path} contains no successful runs.")
    return successful


def floats(rows: Iterable[dict[str, str]], column: str) -> list[float]:
    return [float(row[column]) for row in rows]


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize_method(
    strategy: str,
    label: str,
    rows: list[dict[str, str]],
    source_file: str,
) -> dict[str, object]:
    summary: dict[str, object] = {
        "strategy": strategy,
        "method": label,
        "successful_runs": len(rows),
        "distinct_seeds": len({int(row["seed"]) for row in rows}),
        "unique_removal_paths": len(
            {row["removal_sequence"] for row in rows}
        ),
        "seeds": ",".join(
            str(seed) for seed in sorted({int(row["seed"]) for row in rows})
        ),
        "source_file": source_file,
    }
    for output_name, column in (
        ("support_steps", "support_steps"),
        ("support_moves", "support_moves"),
        ("peak_supports", "peak_supports"),
        ("runtime_s", "elapsed_s"),
    ):
        values = floats(rows, column)
        summary[f"{output_name}_min"] = min(values)
        summary[f"{output_name}_max"] = max(values)
        summary[f"{output_name}_average"] = statistics.mean(values)
        summary[f"{output_name}_median"] = statistics.median(values)
    return summary


def format_range(
    summary: dict[str, object],
    metric: str,
    digits: int,
) -> str:
    minimum = float(summary[f"{metric}_min"])
    maximum = float(summary[f"{metric}_max"])
    return f"{minimum:.{digits}f}-{maximum:.{digits}f}"


def format_distribution(
    summary: dict[str, object],
    metric: str,
    digits: int,
) -> str:
    if int(summary["successful_runs"]) == 1:
        return f"{float(summary[f'{metric}_average']):.{digits}f}"
    return " / ".join(
        (
            format_range(summary, metric, digits),
            f"{float(summary[f'{metric}_average']):.{digits}f}",
            f"{float(summary[f'{metric}_median']):.{digits}f}",
        )
    )


def abbreviated_seeds(summary: dict[str, object]) -> str:
    seeds = [int(value) for value in str(summary["seeds"]).split(",")]
    if len(seeds) <= 10:
        return ",".join(str(seed) for seed in seeds)
    return (
        f"{seeds[0]},{seeds[1]},...,{seeds[-1]} "
        f"({len(seeds)} distinct seeds)"
    )


def write_png(
    path: Path,
    *,
    summaries: list[dict[str, object]],
) -> None:
    import matplotlib.pyplot as plt

    headers = [
        "Method",
        "Successful\nruns",
        "Support steps\nrange / avg. / median",
        "Support moves\nrange / avg. / median",
        "Runtime [s]\nrange / avg. / median",
    ]
    table_rows = [
        [
            str(summary["method"]).replace(" + ", "\n+ ", 1),
            str(summary["successful_runs"]),
            format_distribution(summary, "support_steps", 1),
            format_distribution(summary, "support_moves", 1),
            format_distribution(summary, "runtime_s", 3),
        ]
        for summary in summaries
    ]

    figure, axis = plt.subplots(figsize=(15.5, 6.2))
    axis.axis("off")
    axis.set_title(
        "20-Rod Support Search Comparison",
        fontsize=18,
        fontweight="bold",
        pad=18,
    )
    table = axis.table(
        cellText=table_rows,
        colLabels=headers,
        cellLoc="center",
        colLoc="center",
        loc="center",
        colWidths=[0.31, 0.09, 0.20, 0.20, 0.20],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.0, 2.35)

    for (row_index, column_index), cell in table.get_celld().items():
        cell.set_edgecolor("#B8BEC7")
        if row_index == 0:
            cell.set_facecolor("#2E4057")
            cell.set_text_props(color="white", fontweight="bold")
        elif row_index % 2 == 0:
            cell.set_facecolor("#EEF2F5")
        else:
            cell.set_facecolor("white")
        if row_index > 0 and column_index == 0:
            cell.set_text_props(ha="left")

    figure.text(
        0.5,
        0.06,
        (
            "Cells report range / average / median over successful runs; "
            "the expensive reference methods contain one run each."
        ),
        ha="center",
        fontsize=11,
        color="#4F5B66",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    fast_paths = [
        args.input_dir / filename
        for filename in (args.fast_files or [DEFAULT_FAST_FILE])
    ]
    step_path = args.input_dir / args.step_reference_file
    move_path = args.input_dir / args.move_reference_file
    fast_furthest_path = args.input_dir / args.fast_furthest_file
    default_lowest_path = args.input_dir / args.default_lowest_file
    default_furthest_path = args.input_dir / args.default_furthest_file
    baseline_lowest_path = args.input_dir / args.baseline_lowest_file
    random_path = (
        args.input_dir / args.random_comparison_file
        if args.random_comparison_file
        else None
    )

    fast_rows = [
        row
        for fast_path in fast_paths
        for row in load_successful_rows(fast_path)
    ]
    fast_seeds = [int(row["seed"]) for row in fast_rows]
    if len(fast_seeds) != len(set(fast_seeds)):
        raise ValueError("The selected fast-run files contain duplicate seeds.")
    fast_furthest_rows = load_successful_rows(fast_furthest_path)
    default_lowest_rows = load_successful_rows(default_lowest_path)
    default_furthest_rows = load_successful_rows(default_furthest_path)
    baseline_lowest_rows = load_successful_rows(baseline_lowest_path)
    step_reference_rows = load_successful_rows(step_path)
    move_reference_rows = load_successful_rows(move_path)
    random_rows = (
        load_successful_rows(random_path)
        if random_path is not None
        else None
    )

    comparison_rows = []
    for method, rows in (
        ("baseline_lowest", baseline_lowest_rows),
        ("default_lowest", default_lowest_rows),
        ("default_furthest", default_furthest_rows),
        ("fast_reduce_support_lowest", fast_rows),
        ("fast_reduce_support_furthest", fast_furthest_rows),
    ):
        comparison_rows.extend(
            {
                "method": method,
                "repetition": int(row["repetition"]),
                "seed": int(row["seed"]),
                "runtime_s": float(row["elapsed_s"]),
                "support_steps": int(row["support_steps"]),
                "support_moves": int(row["support_moves"]),
                "peak_supports": int(row["peak_supports"]),
            }
            for row in rows
        )
    if random_rows is not None:
        comparison_rows.extend(
            {
                "method": "baseline_random",
                "repetition": int(row["repetition"]),
                "seed": int(row["seed"]),
                "runtime_s": float(row["elapsed_s"]),
                "support_steps": int(row["support_steps"]),
                "support_moves": int(row["support_moves"]),
                "peak_supports": int(row["peak_supports"]),
            }
            for row in random_rows
        )
    summary_rows = [
        summarize_method(
            "baseline_lowest",
            "Uninformed baseline + lowest",
            baseline_lowest_rows,
            baseline_lowest_path.name,
        ),
        summarize_method(
            "default_lowest",
            "Default + lowest",
            default_lowest_rows,
            default_lowest_path.name,
        ),
        summarize_method(
            "default_furthest",
            "Default + furthest",
            default_furthest_rows,
            default_furthest_path.name,
        ),
        summarize_method(
            "fast_reduce_support",
            "Fast reduce support + lowest",
            fast_rows,
            ",".join(path.name for path in fast_paths),
        ),
        summarize_method(
            "fast_reduce_support_furthest",
            "Fast reduce support + furthest",
            fast_furthest_rows,
            fast_furthest_path.name,
        ),
        summarize_method(
            "reduced_overall_support_steps",
            "Reduced overall support steps",
            step_reference_rows,
            step_path.name,
        ),
        summarize_method(
            "reduced_overall_supports",
            "Reduced overall supports",
            move_reference_rows,
            move_path.name,
        ),
    ]
    if random_rows is not None:
        summary_rows.insert(
            0,
            summarize_method(
                "baseline_random",
                "Uninformed baseline + random supports",
                random_rows,
                random_path.name,
            ),
        )

    write_csv(args.output_runs, comparison_rows)
    write_csv(args.output_summary, summary_rows)

    markdown_rows = "\n".join(
        "| "
        + " | ".join(
            (
                str(summary["method"]),
                str(summary["successful_runs"]),
                format_distribution(summary, "support_steps", 1),
                format_distribution(summary, "support_moves", 1),
                format_distribution(summary, "runtime_s", 3),
            )
        )
        + " |"
        for summary in summary_rows
    )
    markdown = f"""# 20-Rod Support Search Comparison

Only successful runs are included.

| Method | Successful runs | Support steps (range / avg. / median) | Support moves (range / avg. / median) | Runtime [s] (range / avg. / median) |
|---|---:|---:|---:|---:|
{markdown_rows}

Cells report range / average / median over successful runs.

Heuristic-run seeds: `{abbreviated_seeds(summary_rows[0])}`.

The reference values are the best results in the supplied expensive-search files. They are exact for the transition choices explored by those searches, but they do not prove a global physical optimum over every possible support-target assignment.
"""
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.write_text(markdown, encoding="utf-8")
    write_png(
        args.output_png,
        summaries=summary_rows,
    )

    print(markdown)
    print(f"Detailed runs: {args.output_runs}")
    print(f"Summary CSV:   {args.output_summary}")
    print(f"Markdown:      {args.output_markdown}")
    print(f"PNG:           {args.output_png}")


if __name__ == "__main__":
    main()
