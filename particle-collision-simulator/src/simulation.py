"""
simulation.py — High-level simulation orchestrator.

This module glues together the Monte Carlo engine (``montecarlo.py``) and
the physics calculations (``physics.py``) into a single, easy-to-call
pipeline.  It is the main entry point that ``main.py`` delegates to.

Workflow
────────
1.  Instantiate a ``MonteCarloEngine`` with the user's configuration.
2.  Sample stochastic event parameters (energies, angles).
3.  Compute the full kinematics for every event in a vectorised pass.
4.  Pack the results into a ``pandas.DataFrame`` for downstream use.
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from config import SimulationConfig
from montecarlo import MonteCarloEngine
from physics import compute_kinematics_vectorised


def run_simulation(config: SimulationConfig) -> pd.DataFrame:
    """Execute the full Monte Carlo simulation and return a DataFrame.

    Parameters
    ----------
    config : SimulationConfig
        All tuneable parameters for the run.

    Returns
    -------
    pd.DataFrame
        One row per collision event with columns:
        ``event_id, particle_A_energy, particle_B_energy,
        collision_angle, momentum_x, momentum_y, momentum_z,
        final_energy_1, final_energy_2``.
    """
    print(config.summary())
    print()

    # ── 1. Initialise the Monte Carlo engine ────────────────────────────
    engine = MonteCarloEngine(config)
    print(f"  ⚙  {engine.seed_info()}")

    # ── 2. Sample stochastic parameters ─────────────────────────────────
    print("  → Sampling event parameters …")
    t0 = time.perf_counter()
    params = engine.generate_event_parameters()
    dt_sample = time.perf_counter() - t0
    print(f"    ✔ Sampling complete  ({dt_sample:.2f} s)")

    energies_a: np.ndarray = params["energies_a"]
    energies_b: np.ndarray = params["energies_b"]
    thetas: np.ndarray = params["thetas"]

    # ── 3. Compute kinematics (vectorised) ──────────────────────────────
    print("  → Computing kinematics …")
    t0 = time.perf_counter()

    # Seed numpy's legacy RNG so that the azimuthal-angle sampling inside
    # `compute_kinematics_vectorised` is also reproducible.
    if config.random_seed is not None:
        np.random.seed(config.random_seed + 1)

    kinematics = compute_kinematics_vectorised(
        energies_a,
        energies_b,
        thetas,
        mass_a=config.particle_a_mass,
        mass_b=config.particle_b_mass,
    )
    dt_kin = time.perf_counter() - t0
    print(f"    ✔ Kinematics complete  ({dt_kin:.2f} s)")

    # ── 4. Assemble the dataset ─────────────────────────────────────────
    print("  → Assembling dataset …")
    n = config.number_of_events
    df = pd.DataFrame(
        {
            "event_id": np.arange(1, n + 1, dtype=np.int64),
            "particle_A_energy": energies_a,
            "particle_B_energy": energies_b,
            "collision_angle": thetas,
            "momentum_x": kinematics["momentum_x"],
            "momentum_y": kinematics["momentum_y"],
            "momentum_z": kinematics["momentum_z"],
            "final_energy_1": kinematics["final_energy_1"],
            "final_energy_2": kinematics["final_energy_2"],
            "sqrt_s": kinematics["sqrt_s"],
        }
    )

    total = dt_sample + dt_kin
    print(f"    ✔ {n:,} events generated in {total:.2f} s "
          f"({n / total:,.0f} events/s)\n")

    return df
