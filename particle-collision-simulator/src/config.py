"""
config.py — Global simulation configuration and physical constants.

This module centralises every tuneable parameter so that the rest of the
codebase never hard-codes magic numbers.  All energies are in GeV, momenta
in GeV/c, and angles in radians unless stated otherwise.

Physical constants are taken from the Particle Data Group (PDG) 2024 review.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Tuple

# =============================================================================
# Physical constants (natural units: ℏ = c = 1)
# =============================================================================
SPEED_OF_LIGHT: float = 1.0          # c  (natural units)
PROTON_MASS_GEV: float = 0.938272    # GeV/c²
PION_MASS_GEV: float = 0.13957      # GeV/c²  (charged pion)
ELECTRON_MASS_GEV: float = 0.000511  # GeV/c²
MUON_MASS_GEV: float = 0.10566      # GeV/c²

# TODO 1: Add support for multiple particle types.
# Extend this mapping and let the user choose which particles collide.
PARTICLE_MASSES: dict[str, float] = {
    "proton": PROTON_MASS_GEV,
    "pion": PION_MASS_GEV,
    "electron": ELECTRON_MASS_GEV,
    "muon": MUON_MASS_GEV,
}


# =============================================================================
# Simulation parameters (dataclass for easy CLI overriding)
# =============================================================================
@dataclass
class SimulationConfig:
    """Holds every parameter that steers a simulation run.

    Attributes
    ----------
    number_of_events : int
        Total collision events to generate.
    energy_range : Tuple[float, float]
        (E_min, E_max) in GeV for the flat prior on beam energies.
    angle_range : Tuple[float, float]
        (θ_min, θ_max) in radians for the collision angle sampling.
    random_seed : int | None
        Seed for reproducibility.  ``None`` → non-deterministic.
    particle_a_mass : float
        Rest mass of beam particle A in GeV/c².
    particle_b_mass : float
        Rest mass of beam particle B in GeV/c².
    save_dataset : bool
        Whether to persist the generated dataset to disk.
    generate_plots : bool
        Whether to produce Matplotlib figures.
    output_dir : str
        Root directory for all outputs.
    """

    number_of_events: int = 100_000
    energy_range: Tuple[float, float] = (1.0, 1000.0)
    angle_range: Tuple[float, float] = (0.0, math.pi)
    random_seed: int | None = 42
    particle_a_mass: float = PROTON_MASS_GEV
    particle_b_mass: float = PROTON_MASS_GEV
    save_dataset: bool = True
    generate_plots: bool = True
    output_dir: str = "."

    # TODO 7: Add experiment configuration files (YAML / JSON) so users can
    # load pre-defined experiment setups instead of passing CLI flags.

    def __post_init__(self) -> None:
        """Validate ranges after initialisation."""
        e_min, e_max = self.energy_range
        if e_min <= 0 or e_max <= e_min:
            raise ValueError(
                f"energy_range must satisfy 0 < E_min < E_max, got {self.energy_range}"
            )
        a_min, a_max = self.angle_range
        if a_min < 0 or a_max > math.pi or a_min >= a_max:
            raise ValueError(
                f"angle_range must satisfy 0 ≤ θ_min < θ_max ≤ π, got {self.angle_range}"
            )

    def summary(self) -> str:
        """Return a human-readable summary string."""
        lines = [
            "╔══════════════════════════════════════════╗",
            "║        Simulation Configuration          ║",
            "╚══════════════════════════════════════════╝",
            f"  Events         : {self.number_of_events:>12,}",
            f"  Energy range   : [{self.energy_range[0]:.1f}, {self.energy_range[1]:.1f}] GeV",
            f"  Angle range    : [{self.angle_range[0]:.3f}, {self.angle_range[1]:.3f}] rad",
            f"  Particle A mass: {self.particle_a_mass:.6f} GeV/c²",
            f"  Particle B mass: {self.particle_b_mass:.6f} GeV/c²",
            f"  Random seed    : {self.random_seed}",
            f"  Save dataset   : {self.save_dataset}",
            f"  Generate plots : {self.generate_plots}",
        ]
        return "\n".join(lines)
