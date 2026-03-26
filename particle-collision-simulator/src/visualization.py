"""
visualization.py — Publication-quality Matplotlib figures.

Every public function accepts a DataFrame and an output directory, renders
a figure to screen (optional) and saves a high-resolution PNG and PDF to
``results/plots/``.

Style guide
───────────
• Figures use a clean, white background with subtle grid lines — the same
  aesthetic favoured by ATLAS and CMS publications.
• Axis labels include units; titles are descriptive but concise.
• Histograms use semi-transparent fills so overlapping distributions are
  visible.

TODO 6: Add 3D collision visualisation using ``mpl_toolkits.mplot3d``
        or an interactive ``plotly`` figure.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")                       # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

# ── Matplotlib global style ─────────────────────────────────────────────
plt.rcParams.update({
    "figure.figsize": (10, 6),
    "figure.dpi": 150,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 15,
    "legend.fontsize": 11,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.15,
})

COLORS = {
    "particle_a": "#2196F3",
    "particle_b": "#FF5722",
    "final_1": "#4CAF50",
    "final_2": "#9C27B0",
    "angle": "#FF9800",
    "momentum": "#00BCD4",
}


def _save(fig: plt.Figure, plots_dir: Path, name: str) -> None:
    """Save a figure as PNG and PDF."""
    plots_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        filepath = plots_dir / f"{name}.{ext}"
        fig.savefig(filepath)
    plt.close(fig)


# =====================================================================
# 1.  Energy distribution histogram
# =====================================================================
def plot_energy_distribution(df: pd.DataFrame, output_dir: str | Path) -> None:
    """Overlaid histograms of beam and final-state energies.

    Shows:
      • Particle A beam energy (blue)
      • Particle B beam energy (red)
      • Final-state particle 1 energy (green)
      • Final-state particle 2 energy (purple)
    """
    plots_dir = Path(output_dir) / "results" / "plots"
    fig, ax = plt.subplots()

    bins = np.linspace(0, df["particle_A_energy"].max() * 1.05, 120)

    ax.hist(df["particle_A_energy"], bins=bins, alpha=0.55,
            color=COLORS["particle_a"], label="Particle A (beam)")
    ax.hist(df["particle_B_energy"], bins=bins, alpha=0.55,
            color=COLORS["particle_b"], label="Particle B (beam)")
    ax.hist(df["final_energy_1"], bins=bins, alpha=0.45,
            color=COLORS["final_1"], label="Final state 1")
    ax.hist(df["final_energy_2"], bins=bins, alpha=0.45,
            color=COLORS["final_2"], label="Final state 2")

    ax.set_xlabel("Energy [GeV]")
    ax.set_ylabel("Events / bin")
    ax.set_title("Energy Distribution of Beam and Final-State Particles")
    ax.legend(loc="upper right")

    _save(fig, plots_dir, "energy_distribution")
    print("  📊  Plot saved → energy_distribution.png")


# =====================================================================
# 2.  Angular distribution
# =====================================================================
def plot_angular_distribution(df: pd.DataFrame, output_dir: str | Path) -> None:
    """Histogram of collision angles weighted by sin θ solid-angle element."""
    plots_dir = Path(output_dir) / "results" / "plots"
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- linear scale ---
    axes[0].hist(df["collision_angle"], bins=100, alpha=0.7,
                 color=COLORS["angle"], edgecolor="white", linewidth=0.3)
    axes[0].set_xlabel("Collision angle θ [rad]")
    axes[0].set_ylabel("Events / bin")
    axes[0].set_title("Angular Distribution (linear)")

    # --- cos θ distribution (should be flat for isotropic sampling) ---
    cos_theta = np.cos(df["collision_angle"])
    axes[1].hist(cos_theta, bins=100, alpha=0.7,
                 color=COLORS["angle"], edgecolor="white", linewidth=0.3)
    axes[1].set_xlabel("cos θ")
    axes[1].set_ylabel("Events / bin")
    axes[1].set_title("cos θ Distribution (isotropy check)")

    fig.suptitle("Collision Angle Distributions", fontsize=16, y=1.02)
    fig.tight_layout()
    _save(fig, plots_dir, "angular_distribution")
    print("  📊  Plot saved → angular_distribution.png")


# =====================================================================
# 3.  Momentum vector scatter plots
# =====================================================================
def plot_momentum_scatter(df: pd.DataFrame, output_dir: str | Path) -> None:
    """2-D scatter of total system momentum components (px vs py, px vs pz)."""
    plots_dir = Path(output_dir) / "results" / "plots"
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    sample = df.sample(n=min(20_000, len(df)), random_state=0)

    axes[0].scatter(sample["momentum_x"], sample["momentum_y"],
                    s=1, alpha=0.3, c=COLORS["momentum"])
    axes[0].set_xlabel("$p_x$ [GeV/c]")
    axes[0].set_ylabel("$p_y$ [GeV/c]")
    axes[0].set_title("$p_x$ vs $p_y$")
    axes[0].set_aspect("equal")

    axes[1].scatter(sample["momentum_x"], sample["momentum_z"],
                    s=1, alpha=0.3, c=COLORS["momentum"])
    axes[1].set_xlabel("$p_x$ [GeV/c]")
    axes[1].set_ylabel("$p_z$ [GeV/c]")
    axes[1].set_title("$p_x$ vs $p_z$")

    fig.suptitle("Total System Momentum Components", fontsize=16, y=1.02)
    fig.tight_layout()
    _save(fig, plots_dir, "momentum_scatter")
    print("  📊  Plot saved → momentum_scatter.png")


# =====================================================================
# 4.  Event density (2-D histogram / heat-map)
# =====================================================================
def plot_event_density(df: pd.DataFrame, output_dir: str | Path) -> None:
    """2-D histogram of (E_A, E_B) coloured by event density."""
    plots_dir = Path(output_dir) / "results" / "plots"
    fig, ax = plt.subplots()

    h = ax.hist2d(
        df["particle_A_energy"],
        df["particle_B_energy"],
        bins=150,
        cmap="inferno",
        cmin=1,
    )
    fig.colorbar(h[3], ax=ax, label="Events")
    ax.set_xlabel("Particle A energy [GeV]")
    ax.set_ylabel("Particle B energy [GeV]")
    ax.set_title("Beam Energy Density Map")

    _save(fig, plots_dir, "event_density")
    print("  📊  Plot saved → event_density.png")


# =====================================================================
# 5.  Final-state energy correlation
# =====================================================================
def plot_final_energy_correlation(df: pd.DataFrame, output_dir: str | Path) -> None:
    """Scatter / 2-D histogram of the two final-state energies.

    A tight anti-correlation indicates energy conservation in the CM frame.
    """
    plots_dir = Path(output_dir) / "results" / "plots"
    fig, ax = plt.subplots()

    h = ax.hist2d(
        df["final_energy_1"],
        df["final_energy_2"],
        bins=150,
        cmap="viridis",
        cmin=1,
    )
    fig.colorbar(h[3], ax=ax, label="Events")
    ax.set_xlabel("Final-state particle 1 energy [GeV]")
    ax.set_ylabel("Final-state particle 2 energy [GeV]")
    ax.set_title("Final-State Energy Correlation (energy conservation)")

    _save(fig, plots_dir, "final_energy_correlation")
    print("  📊  Plot saved → final_energy_correlation.png")


# =====================================================================
# 6.  Invariant mass spectrum
# =====================================================================
def plot_invariant_mass(df: pd.DataFrame, output_dir: str | Path) -> None:
    """Histogram of the reconstructed invariant mass √s."""
    plots_dir = Path(output_dir) / "results" / "plots"
    fig, ax = plt.subplots()

    # Prefer the true √s if available; fall back to E1 + E2
    if "sqrt_s" in df.columns:
        sqrt_s = df["sqrt_s"]
    else:
        sqrt_s = df["final_energy_1"] + df["final_energy_2"]
    ax.hist(sqrt_s, bins=150, alpha=0.7, color="#E91E63",
            edgecolor="white", linewidth=0.3)
    ax.set_xlabel("Invariant mass $\\sqrt{s}$ [GeV]")
    ax.set_ylabel("Events / bin")
    ax.set_title("Reconstructed Invariant Mass Spectrum")

    _save(fig, plots_dir, "invariant_mass")
    print("  📊  Plot saved → invariant_mass.png")


# =====================================================================
# Public driver
# =====================================================================
def generate_all_plots(df: pd.DataFrame, output_dir: str | Path) -> None:
    """Generate the full suite of standard plots.

    Parameters
    ----------
    df : pd.DataFrame
        Collision event dataset.
    output_dir : str or Path
        Project root directory.
    """
    print("\n  ──── Generating Plots ────")
    plot_energy_distribution(df, output_dir)
    plot_angular_distribution(df, output_dir)
    plot_momentum_scatter(df, output_dir)
    plot_event_density(df, output_dir)
    plot_final_energy_correlation(df, output_dir)
    plot_invariant_mass(df, output_dir)
    print("  ──── All plots saved ────\n")
