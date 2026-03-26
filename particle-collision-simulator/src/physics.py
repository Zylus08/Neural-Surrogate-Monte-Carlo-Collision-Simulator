"""
physics.py — Core relativistic kinematics for two-body collisions.

This module contains the physical equations used to compute the kinematics
of a 2→2 particle scattering process.  The treatment is *simplified* but
physically motivated: we work in the centre-of-mass (CM) frame, enforce
conservation of four-momentum, and compute final-state energies and momenta
analytically.

Key equations
─────────────
    E² = p²c² + m²c⁴          (energy–momentum relation, c = 1)
    √s = E_A + E_B              (CM energy in the CM frame)
    p  = |𝐩| = √(E² − m²)     (3-momentum magnitude)

In the CM frame the two outgoing particles share √s equally when their
masses are equal; otherwise the split is determined by four-momentum
conservation.

TODO 2: Implement relativistic energy calculations (Lorentz boosts,
        γ factors, lab ↔ CM frame transformations).
TODO 3: Add detector noise simulation (Gaussian smearing on measured
        energies and angles to mimic realistic detector resolution).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


# =============================================================================
# Data containers
# =============================================================================
@dataclass
class CollisionEvent:
    """Fully characterised two-body collision event.

    All quantities are in natural units (GeV, GeV/c, radians).
    """

    event_id: int
    particle_a_energy: float
    particle_b_energy: float
    collision_angle: float          # θ in [0, π]
    momentum_x: float              # total px of the system
    momentum_y: float              # total py of the system
    momentum_z: float              # total pz of the system
    final_energy_1: float          # outgoing particle 1
    final_energy_2: float          # outgoing particle 2


# =============================================================================
# Kinematic helpers
# =============================================================================
def momentum_magnitude(energy: float, mass: float) -> float:
    """Compute the 3-momentum magnitude from the energy–momentum relation.

    .. math::
        |p| = \\sqrt{E^2 - m^2}

    Parameters
    ----------
    energy : float
        Total energy of the particle (GeV).
    mass : float
        Rest mass of the particle (GeV/c²).

    Returns
    -------
    float
        Magnitude of the 3-momentum (GeV/c).

    Raises
    ------
    ValueError
        If the energy is below the rest mass (unphysical).
    """
    if energy < mass:
        raise ValueError(
            f"Energy ({energy:.4f} GeV) is below rest mass ({mass:.4f} GeV). "
            "This is kinematically forbidden."
        )
    return math.sqrt(energy**2 - mass**2)


def momentum_components(
    p: float,
    theta: float,
    phi: float,
) -> tuple[float, float, float]:
    """Decompose a momentum magnitude into Cartesian components.

    Spherical → Cartesian:
        px = p sin θ cos φ
        py = p sin θ sin φ
        pz = p cos θ

    Parameters
    ----------
    p : float   – magnitude (GeV/c)
    theta : float – polar angle [0, π]
    phi : float   – azimuthal angle [0, 2π)

    Returns
    -------
    (px, py, pz) in GeV/c
    """
    sin_theta = math.sin(theta)
    px = p * sin_theta * math.cos(phi)
    py = p * sin_theta * math.sin(phi)
    pz = p * math.cos(theta)
    return px, py, pz


def invariant_mass(energy_a: float, energy_b: float,
                   p_a: float, p_b: float,
                   cos_theta: float) -> float:
    """Centre-of-mass energy √s (Mandelstam variable) for two colliding
    particles.

    Starting from the Lorentz-invariant definition:

    .. math::
        s = (E_A + E_B)^2 - |\\mathbf{p}_A + \\mathbf{p}_B|^2

    and expanding |𝐩_A + 𝐩_B|² = p_A² + p_B² + 2 p_A p_B cos θ where θ
    is the *opening angle* between the two 3-momentum vectors, we get:

    .. math::
        s = m_A^2 + m_B^2 + 2(E_A E_B - |p_A||p_B| \\cos\\theta)

    For perfectly head-on beams (θ = π → cos θ = −1) the cross-term is
    *positive*, maximising √s.  For co-moving particles (θ = 0) it is
    *negative*.

    Parameters
    ----------
    energy_a, energy_b : float  – total energies (GeV)
    p_a, p_b : float            – 3-momentum magnitudes (GeV/c)
    cos_theta : float           – cosine of the opening angle between the
                                  two 3-momentum vectors

    Returns
    -------
    float – √s in GeV
    """
    s = (energy_a + energy_b) ** 2 - (p_a**2 + p_b**2 + 2 * p_a * p_b * cos_theta)
    # Guard against floating-point issues near threshold
    return math.sqrt(max(s, 0.0))


# =============================================================================
# Two-body final-state kinematics (CM frame, elastic-like)
# =============================================================================
def compute_final_state_energies(
    sqrt_s: float,
    m_out1: float = 0.0,
    m_out2: float = 0.0,
) -> tuple[float, float]:
    """Compute the energies of two outgoing particles in the CM frame.

    In the CM frame the total 3-momentum is zero, so energy–momentum
    conservation gives:

    .. math::
        E_1^* = \\frac{s + m_1^2 - m_2^2}{2\\sqrt{s}}, \\quad
        E_2^* = \\sqrt{s} - E_1^*

    When *m_out1 = m_out2 = 0* (massless products, e.g. photon pairs) each
    gets half the CM energy, which is the ultra-relativistic limit commonly
    used in simplified simulations.

    Parameters
    ----------
    sqrt_s : float – CM energy (GeV)
    m_out1, m_out2 : float – rest masses of outgoing particles (GeV/c²)

    Returns
    -------
    (E1, E2) in GeV
    """
    if sqrt_s <= 0:
        return 0.0, 0.0

    s = sqrt_s**2
    e1 = (s + m_out1**2 - m_out2**2) / (2.0 * sqrt_s)
    e2 = sqrt_s - e1
    # Clamp to rest mass (numerical safety)
    e1 = max(e1, m_out1)
    e2 = max(e2, m_out2)
    return e1, e2


# =============================================================================
# Vectorised kinematic computation for arrays of events
# =============================================================================
def compute_kinematics_vectorised(
    energies_a: NDArray[np.float64],
    energies_b: NDArray[np.float64],
    thetas: NDArray[np.float64],
    mass_a: float,
    mass_b: float,
) -> dict[str, NDArray[np.float64]]:
    """Compute the full kinematics for *N* collision events at once.

    This is the performance-critical path — everything is vectorised with
    NumPy so we can process millions of events without a Python-level loop.

    **Collider geometry**:  Particle A propagates along +z and particle B
    along −z.  ``thetas`` is the crossing half-angle each beam makes with
    the z-axis (θ = 0 → perfectly head-on).

    **Energy conservation**:  Final-state energies are computed as
    fractions of the *lab-frame* total energy E_A + E_B, so energy is
    conserved exactly event-by-event.

    Parameters
    ----------
    energies_a, energies_b : (N,) arrays of beam energies (GeV)
    thetas : (N,) array of crossing half-angles (rad)
    mass_a, mass_b : rest masses (GeV/c²)

    Returns
    -------
    dict with keys:
        momentum_x, momentum_y, momentum_z,
        final_energy_1, final_energy_2, sqrt_s
    """
    n = len(energies_a)

    # -----------------------------------------------------------------
    # Collider geometry
    # -----------------------------------------------------------------
    # Particle A travels predominantly along +z.
    # Particle B travels predominantly along −z.
    # `thetas` is the *crossing half-angle* — the angle each beam makes
    # with the z-axis.  For perfectly head-on beams θ = 0; for the LHC
    # the crossing angle is ~150 µrad.  Here the user can set any value
    # in [0, π] so the simulator also covers fixed-target kinematics.
    #
    # Azimuthal angles φ_A and φ_B are drawn independently and
    # uniformly in [0, 2π) to break cylindrical symmetry when θ ≠ 0.
    # -----------------------------------------------------------------

    # 3-momentum magnitudes  |p| = √(E² − m²)
    p_a = np.sqrt(np.clip(energies_a**2 - mass_a**2, 0.0, None))
    p_b = np.sqrt(np.clip(energies_b**2 - mass_b**2, 0.0, None))

    # Random azimuthal angles for each beam
    phis_a = np.random.uniform(0.0, 2.0 * np.pi, size=n)
    phis_b = np.random.uniform(0.0, 2.0 * np.pi, size=n)

    sin_theta = np.sin(thetas)
    cos_theta = np.cos(thetas)

    # Particle A  (beam direction ≈ +z)
    #   polar angle from +z axis = thetas
    px_a = p_a * sin_theta * np.cos(phis_a)
    py_a = p_a * sin_theta * np.sin(phis_a)
    pz_a = p_a * cos_theta

    # Particle B  (beam direction ≈ −z)
    #   polar angle from +z axis = π − thetas  (i.e. thetas from −z)
    #   sin(π − θ) = sin θ,  cos(π − θ) = −cos θ
    px_b = p_b * sin_theta * np.cos(phis_b)
    py_b = p_b * sin_theta * np.sin(phis_b)
    pz_b = -p_b * cos_theta                   # dominant component along −z

    # Total system 3-momentum
    px_total = px_a + px_b
    py_total = py_a + py_b
    pz_total = pz_a + pz_b

    # -----------------------------------------------------------------
    # Invariant mass  √s
    # -----------------------------------------------------------------
    #   s = (E_A + E_B)² − |𝐩_A + 𝐩_B|²
    p_total_sq = px_total**2 + py_total**2 + pz_total**2
    s = (energies_a + energies_b) ** 2 - p_total_sq
    sqrt_s = np.sqrt(np.clip(s, 0.0, None))

    # -----------------------------------------------------------------
    # Final-state energies  (lab frame, energy conservation enforced)
    # -----------------------------------------------------------------
    # In a real detector we measure lab-frame quantities, so the two
    # outgoing particle energies must satisfy:
    #     E_1 + E_2 = E_A + E_B          (total energy conservation)
    #
    # We split the lab-frame total energy with a small random asymmetry
    # drawn uniformly from [−0.1, +0.1].  This models the stochastic
    # nature of the CM→lab boost direction for each event.
    total_energy = energies_a + energies_b
    asymmetry = np.random.uniform(-0.1, 0.1, size=n)
    e1 = total_energy * (0.5 + asymmetry)
    e2 = total_energy - e1                     # exact conservation
    e1 = np.clip(e1, 0.0, None)
    e2 = np.clip(e2, 0.0, None)

    return {
        "momentum_x": px_total,
        "momentum_y": py_total,
        "momentum_z": pz_total,
        "final_energy_1": e1,
        "final_energy_2": e2,
        "sqrt_s": sqrt_s,
    }
