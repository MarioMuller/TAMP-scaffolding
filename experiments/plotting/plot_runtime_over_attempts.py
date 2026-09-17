#!/usr/bin/env python3
"""Plot search runtime over scaffold size for the first two repetitions."""

import argparse
import csv
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = (
    PROJECT_ROOT
    / "experiments"
    / "results"
    / "archive"
    / "rigidity_check_scaling_baseline.csv"
)
DEFAULT_OUTPUT = Path(__file__).with_name(
    "runtime_over_attempts_first_two_runs.png"
)

MPLCONFIGDIR = Path("/tmp/tamp_scaffolding_matplotlib")
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))

import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Create a log-scale scatter plot of runtime over scaffold size for "
            "repetitions 1 and 2."
        )
    )
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--title")
    parser.add_argument(
        "--linear",
        action="store_true",
        help="Use a linear runtime axis instead of a logarithmic axis.",
    )
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def load_first_two_repetitions(csv_path):
    points = {1: [], 2: []}

    with csv_path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        required_columns = {
            "repetition",
            "included_rod_count",
            "elapsed_s",
        }
        missing_columns = required_columns - set(reader.fieldnames or [])

        if missing_columns:
            raise ValueError(
                "CSV is missing required columns: "
                + ", ".join(sorted(missing_columns))
            )

        for row in reader:
            repetition = int(row["repetition"])
            if repetition not in points:
                continue

            elapsed_s = float(row["elapsed_s"])
            if elapsed_s <= 0:
                continue

            points[repetition].append(
                (int(row["included_rod_count"]), elapsed_s)
            )

    for repetition in points:
        points[repetition].sort(key=lambda point: point[0])

    return points


def strategy_label(csv_path):
    name = csv_path.stem.removeprefix("rigidity_check_scaling_")
    return name.replace("_", " ").title()


def plot_runtime(points, output_path, title, log_scale=True, show=False):
    fig, ax = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
    colors = {1: "#2364aa", 2: "#d1495b"}

    for repetition, repetition_points in points.items():
        if not repetition_points:
            continue

        rod_counts, runtimes = zip(*repetition_points)
        ax.scatter(
            rod_counts,
            runtimes,
            s=30,
            alpha=0.8,
            color=colors[repetition],
            label=f"Repetition {repetition}",
        )

    if log_scale:
        ax.set_yscale("log")

    ax.set_xlabel("Rods in scaffold")
    ax.set_ylabel(
        "Runtime (seconds, log scale)" if log_scale else "Runtime (seconds)"
    )
    ax.set_title(title)
    ax.grid(True, which="both", linewidth=0.6, alpha=0.3)
    ax.legend(frameon=False)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)

    if show:
        plt.show()

    plt.close(fig)


def main():
    args = parse_args()
    points = load_first_two_repetitions(args.input_csv)

    if not any(points.values()):
        raise ValueError("No rows found for repetitions 1 or 2.")

    title = args.title or (
        f"{strategy_label(args.input_csv)} runtime by scaffold size"
    )
    plot_runtime(
        points,
        args.output,
        title=title,
        log_scale=not args.linear,
        show=args.show,
    )
    print(f"Saved plot to {args.output}")


if __name__ == "__main__":
    main()
