#!/usr/bin/env python3
"""Compare fast support reduction with the expensive 20-rod references."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "results"
    / "evaluation"
    / "current"
    / "near_optimal_20_rods"
)
DEFAULT_FAST_FILE = "fast_reduce_support_20rods_10reps.csv"
DEFAULT_STEP_FILE = "reduced_overall_support_steps_20rods_1rep.csv"
DEFAULT_MOVE_FILE = "reduced_overall_supports_20rods_1rep.csv"
DEFAULT_RUN_OUTPUT = Path(__file__).with_name(
    "fast_reduce_vs_reference_minima_runs.csv"
)
DEFAULT_SUMMARY_OUTPUT = Path(__file__).with_name(
    "fast_reduce_vs_reference_minima_summary.csv"
)
DEFAULT_MARKDOWN_OUTPUT = Path(__file__).with_name(
    "fast_reduce_vs_reference_minima.md"
)
DEFAULT_PNG_OUTPUT = Path(__file__).with_name(
    "fast_reduce_vs_reference_minima.png"
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
        "--random-comparison-file",
        help=(
            "Optional baseline-removal/random-support CSV filename relative "
            "to --input-dir."
        ),
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
    return summary


def format_range(
    summary: dict[str, object],
    metric: str,
    digits: int,
) -> str:
    minimum = float(summary[f"{metric}_min"])
    maximum = float(summary[f"{metric}_max"])
    return f"{minimum:.{digits}f}-{maximum:.{digits}f}"


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
        "Distinct\nseeds",
        "Unique removal\npaths",
        "Support steps\nmin-max",
        "Support moves\nmin-max",
        "Peak supports\nmin-max",
        "Runtime [s]\nmin-max",
    ]
    table_rows = [
        [
            str(summary["method"]),
            str(summary["successful_runs"]),
            str(summary["distinct_seeds"]),
            str(summary["unique_removal_paths"]),
            format_range(summary, "support_steps", 0),
            format_range(summary, "support_moves", 0),
            format_range(summary, "peak_supports", 0),
            format_range(summary, "runtime_s", 3),
        ]
        for summary in summaries
    ]

    figure, axis = plt.subplots(figsize=(15.5, 4.4))
    axis.axis("off")
    axis.set_title(
        "20-Rod Support Search Comparison",
        fontsize=16,
        fontweight="bold",
        pad=18,
    )
    table = axis.table(
        cellText=table_rows,
        colLabels=headers,
        cellLoc="center",
        colLoc="center",
        loc="center",
        colWidths=[0.24, 0.09, 0.09, 0.11, 0.12, 0.12, 0.11, 0.14],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.55)

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

    fast_summary = summaries[0]
    figure.text(
        0.5,
        0.06,
        (
            "Cells report the minimum and maximum over successful runs; "
            f"the {fast_summary['successful_runs']} fast runs use "
            f"{fast_summary['distinct_seeds']} distinct seeds."
        ),
        ha="center",
        fontsize=9,
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
    step_reference_rows = load_successful_rows(step_path)
    move_reference_rows = load_successful_rows(move_path)
    random_rows = (
        load_successful_rows(random_path)
        if random_path is not None
        else None
    )

    comparison_rows = [
        {
            "repetition": int(row["repetition"]),
            "seed": int(row["seed"]),
            "runtime_s": float(row["elapsed_s"]),
            "support_steps": int(row["support_steps"]),
            "support_moves": int(row["support_moves"]),
            "peak_supports": int(row["peak_supports"]),
        }
        for row in fast_rows
    ]
    summary_rows = [
        summarize_method(
            "fast_reduce_support",
            "Fast reduce support",
            fast_rows,
            ",".join(path.name for path in fast_paths),
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
            1,
            summarize_method(
                "baseline_random",
                "Random removal + random supports",
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
                str(summary["distinct_seeds"]),
                str(summary["unique_removal_paths"]),
                format_range(summary, "support_steps", 0),
                format_range(summary, "support_moves", 0),
                format_range(summary, "peak_supports", 0),
                format_range(summary, "runtime_s", 3),
            )
        )
        + " |"
        for summary in summary_rows
    )
    markdown = f"""# 20-Rod Support Search Comparison

Only successful runs are included.

| Method | Successful runs | Distinct seeds | Unique removal paths | Support steps min-max | Support moves min-max | Peak supports min-max | Runtime [s] min-max |
|---|---:|---:|---:|---:|---:|---:|---:|
{markdown_rows}

Fast-reduction seeds: `{abbreviated_seeds(summary_rows[0])}`.

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
