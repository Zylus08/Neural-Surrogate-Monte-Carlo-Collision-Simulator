#!/usr/bin/env python3
"""
main.py — CLI entry point for the Particle Collision Monte Carlo Simulator.

Usage
─────
    # Default run (100 000 events, full pipeline)
    python src/main.py

    # Custom event count with plots and dataset export
    python src/main.py --events 50000 --plot --save-dataset

    # Scan a specific energy window
    python src/main.py --events 200000 --energy-min 100 --energy-max 500

Run ``python src/main.py --help`` for the complete option list.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time


def _build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser."""
    parser = argparse.ArgumentParser(
        prog="particle-collision-simulator",
        description=(
            "Monte Carlo simulator for simplified high-energy particle "
            "collisions.  Generates collision event datasets, produces "
            "publication-quality plots, and runs statistical analyses."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python src/main.py --events 50000\n"
            "  python src/main.py --events 200000 --energy-min 10 --energy-max 500\n"
            "  python src/main.py --events 100000 --save-dataset --plot\n"
        ),
    )

    parser.add_argument(
        "--events", "-n",
        type=int,
        default=100_000,
        help="Number of collision events to generate (default: 100 000).",
    )
    parser.add_argument(
        "--energy-min",
        type=float,
        default=1.0,
        help="Minimum beam energy in GeV (default: 1.0).",
    )
    parser.add_argument(
        "--energy-max",
        type=float,
        default=1000.0,
        help="Maximum beam energy in GeV (default: 1000.0).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).  Use -1 for non-deterministic.",
    )
    parser.add_argument(
        "--save-dataset",
        action="store_true",
        default=True,
        help="Save the generated dataset to data/raw/ (default: True).",
    )
    parser.add_argument(
        "--no-save-dataset",
        action="store_true",
        help="Do NOT save the dataset.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        default=True,
        help="Generate Matplotlib plots in results/plots/ (default: True).",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip plot generation.",
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default=None,
        help="Project root directory (default: auto-detected).",
    )

    return parser


def _resolve_output_dir(cli_value: str | None) -> str:
    """Determine the project root directory.

    If the user didn't pass ``--output-dir`` we walk up from the current
    working directory (or the script location) until we find a directory
    that contains ``src/`` and ``data/``.
    """
    if cli_value:
        return os.path.abspath(cli_value)

    # Try the parent of the directory containing this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidate = os.path.dirname(script_dir)
    if os.path.isdir(os.path.join(candidate, "src")) and os.path.isdir(
        os.path.join(candidate, "data")
    ):
        return candidate

    # Fallback to cwd
    return os.getcwd()


def main(argv: list[str] | None = None) -> None:
    """Parse CLI arguments and run the full simulation pipeline."""

    parser = _build_parser()
    args = parser.parse_args(argv)

    # ── Resolve flags ───────────────────────────────────────────────────
    save_dataset = args.save_dataset and not args.no_save_dataset
    generate_plots = args.plot and not args.no_plot
    seed = args.seed if args.seed >= 0 else None
    output_dir = _resolve_output_dir(args.output_dir)

    # ── Lazy imports (so --help is instant) ─────────────────────────────
    from config import SimulationConfig
    from simulation import run_simulation
    from dataset import save_dataset as _save_ds
    from visualization import generate_all_plots
    from analysis import run_analysis

    # ── Build configuration ─────────────────────────────────────────────
    config = SimulationConfig(
        number_of_events=args.events,
        energy_range=(args.energy_min, args.energy_max),
        angle_range=(0.0, math.pi),
        random_seed=seed,
        save_dataset=save_dataset,
        generate_plots=generate_plots,
        output_dir=output_dir,
    )

    # ── Banner ──────────────────────────────────────────────────────────
    print()
    print("╔══════════════════════════════════════════════════════╗")
    print("║   Particle Collision Monte Carlo Simulator          ║")
    print("║   High-Energy Physics · Computational Research      ║")
    print("╚══════════════════════════════════════════════════════╝")
    print()

    wall_start = time.perf_counter()

    # ── 1. Simulation ───────────────────────────────────────────────────
    df = run_simulation(config)

    # ── 2. Save dataset ─────────────────────────────────────────────────
    if save_dataset:
        print("  ──── Saving Dataset ────")
        _save_ds(df, output_dir)
        print()

    # ── 3. Plots ────────────────────────────────────────────────────────
    if generate_plots:
        generate_all_plots(df, output_dir)

    # ── 4. Statistical analysis ─────────────────────────────────────────
    print("  ──── Statistical Analysis ────")
    report = run_analysis(df, output_dir)
    print(report)

    # ── Done ────────────────────────────────────────────────────────────
    wall_total = time.perf_counter() - wall_start
    print(f"\n  ⏱  Total wall time: {wall_total:.2f} s")
    print("  ✔  Simulation complete.\n")


if __name__ == "__main__":
    main()
