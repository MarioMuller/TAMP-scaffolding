#!/usr/bin/env python3
"""Analyze the legacy two-arm RAI pose benchmarks."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
ONE_ARM_ANALYSIS_DIR = Path(__file__).resolve().parents[1] / "one_arm"

if str(ONE_ARM_ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ONE_ARM_ANALYSIS_DIR))

from analyze_one_arm_pose import run_analysis


DEFAULT_INPUT_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "results"
    / "evaluation"
    / "legacy"
    / "dual_arm_pose_support_fractions_0.5_0.6"
)
OUTPUT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze two-arm pose benchmarks. Numerical summaries use "
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
        default=OUTPUT_DIR / "two_arm_pose_summary.csv",
    )
    parser.add_argument(
        "--runs-csv",
        type=Path,
        default=OUTPUT_DIR / "two_arm_pose_runs.csv",
    )
    parser.add_argument(
        "--markdown",
        type=Path,
        default=OUTPUT_DIR / "two_arm_pose_table.md",
    )
    parser.add_argument(
        "--png",
        type=Path,
        default=OUTPUT_DIR / "two_arm_pose_table.png",
    )
    parser.add_argument(
        "--exclude-strategy",
        action="append",
        default=[],
        help="Strategy to omit; may be supplied more than once.",
    )
    return parser.parse_args()


def main() -> None:
    run_analysis(
        parse_args(),
        expected_main_arm_count=2,
        benchmark_title="Two-arm pose benchmark",
    )


if __name__ == "__main__":
    main()
