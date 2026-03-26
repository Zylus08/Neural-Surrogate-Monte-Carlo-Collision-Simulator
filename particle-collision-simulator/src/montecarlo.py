"""
montecarlo.py — Monte Carlo sampling engine.

This module is responsible for the *stochastic* part of the simulation:
drawing random beam energies, collision angles, and any other probabilistic
quantities from their prior distributions.

The engine is intentionally decoupled from the physics calculations so that
different sampling strategies (flat priors, importance sampling, MCMC, …)
can be swapped in without touching the kinematics code.

Monte Carlo Method — Brief Overview
────────────────────────────────────
Monte Carlo methods estimate deterministic quantities by repeated random
sampling.  In high-energy physics the technique is used to:

  1. **Generate pseudo-data** that mimics detector output.
  2. **Integrate** high-dimensional phase-space integrals that are
     analytically intractable.
  3. **Propagate uncertainties** by sampling from distributions of
     nuisance parameters.

Here we use the simplest variant — *direct sampling* from flat (uniform)
priors on energy and angle — which is appropriate for an inclusive,
model-independent event generator.

TODO 5: Add parallel simulation for large event sets using
        ``multiprocessing.Pool`` or ``concurrent.futures``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import stats

from config import SimulationConfig


class MonteCarloEngine:
    """Stochastic event generator.

    Parameters
    ----------
    config : SimulationConfig
        Simulation parameters (energy range, angle range, seed, …).
    """

    def __init__(self, config: SimulationConfig) -> None:
        self.config = config
        self._rng = np.random.default_rng(config.random_seed)

    # ── public API ──────────────────────────────────────────────────────
    def sample_energies(self, n: int | None = None) -> NDArray[np.float64]:
        """Draw beam energies from a flat prior on [E_min, E_max].

        A uniform distribution is the least informative (maximum-entropy)
        prior when only the support is known; it assigns equal probability
        to every energy in the range.

        Parameters
        ----------
        n : int, optional
            Number of samples.  Defaults to ``config.number_of_events``.

        Returns
        -------
        (n,) array of energies in GeV.
        """
        n = n or self.config.number_of_events
        lo, hi = self.config.energy_range
        return self._rng.uniform(lo, hi, size=n)

    def sample_angles(self, n: int | None = None) -> NDArray[np.float64]:
        """Draw collision angles from a *sin θ*-weighted distribution.

        In 3-D the solid-angle element is dΩ = sin θ dθ dφ, so a
        physically uniform angular distribution requires weighting by
        sin θ.  We achieve this via inverse-transform sampling:

            cos θ ~ Uniform(cos θ_max, cos θ_min)
            θ     = arccos(cos θ)

        Parameters
        ----------
        n : int, optional
            Number of samples.

        Returns
        -------
        (n,) array of polar angles in radians.
        """
        n = n or self.config.number_of_events
        lo, hi = self.config.angle_range
        cos_lo = np.cos(hi)   # note: cos is decreasing
        cos_hi = np.cos(lo)
        cos_theta = self._rng.uniform(cos_lo, cos_hi, size=n)
        return np.arccos(cos_theta)

    def sample_azimuthal(self, n: int | None = None) -> NDArray[np.float64]:
        """Draw azimuthal angles φ uniformly in [0, 2π)."""
        n = n or self.config.number_of_events
        return self._rng.uniform(0.0, 2.0 * np.pi, size=n)

    def generate_event_parameters(
        self,
    ) -> dict[str, NDArray[np.float64]]:
        """One-shot generation of all stochastic event parameters.

        Returns
        -------
        dict with keys
            ``energies_a``, ``energies_b``, ``thetas``
        each of shape ``(number_of_events,)``.
        """
        n = self.config.number_of_events
        return {
            "energies_a": self.sample_energies(n),
            "energies_b": self.sample_energies(n),
            "thetas": self.sample_angles(n),
        }

    # ── diagnostics ─────────────────────────────────────────────────────
    def seed_info(self) -> str:
        """Return a string describing the RNG state (for logging)."""
        return (
            f"MonteCarloEngine  seed={self.config.random_seed}  "
            f"events={self.config.number_of_events:,}"
        )
