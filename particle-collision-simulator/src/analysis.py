"""
analysis.py — Statistical analysis of collision datasets.

Computes summary statistics, fits standard distributions to the data, and
writes a structured report to ``results/logs/``.

Distributions attempted
───────────────────────
• **Normal (Gaussian)** — ``scipy.stats.norm.fit``
• **Log-normal** — common for energy spectra that are strictly positive
• **Exponential** — expected when final-state energies follow a
  Boltzmann-like tail

The analysis results are printed to the console AND persisted as a plain-
text log file so they can be referenced or attached to publications.
"""

from __future__ import annotations

import datetime
from io import StringIO
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


# =============================================================================
# Summary statistics
# =============================================================================
def compute_summary_statistics(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Compute basic descriptive statistics for every numeric column.

    Returns
    -------
    dict  mapping ``column_name → {mean, std, var, min, max, median}``.
    """
    result: dict[str, dict[str, float]] = {}
    for col in df.select_dtypes(include=[np.number]).columns:
        data = df[col].to_numpy()
        result[col] = {
            "mean": float(np.mean(data)),
            "std": float(np.std(data)),
            "var": float(np.var(data)),
            "min": float(np.min(data)),
            "max": float(np.max(data)),
            "median": float(np.median(data)),
        }
    return result


# =============================================================================
# Distribution fitting
# =============================================================================
def fit_distributions(
    data: np.ndarray,
    label: str = "data",
) -> dict[str, dict[str, Any]]:
    """Fit several candidate distributions and return goodness-of-fit info.

    For each distribution we compute:
    • MLE parameters (via ``scipy.stats.<dist>.fit``)
    • Kolmogorov–Smirnov test statistic and *p*-value

    A higher *p*-value indicates a better fit (fail to reject H₀ that the
    data come from the fitted distribution).

    Parameters
    ----------
    data : 1-D array
    label : str – human-readable name (for logging)

    Returns
    -------
    dict  mapping ``dist_name → {params, ks_stat, p_value}``.
    """
    results: dict[str, dict[str, Any]] = {}
    candidates = {
        "normal": stats.norm,
        "lognormal": stats.lognorm,
        "exponential": stats.expon,
    }

    # Ensure positive data for lognormal / exponential fits
    positive_data = data[data > 0]

    for name, dist in candidates.items():
        try:
            fit_data = positive_data if name in ("lognormal", "exponential") else data
            params = dist.fit(fit_data)
            ks_stat, p_value = stats.kstest(fit_data, dist.cdf, args=params)
            results[name] = {
                "params": params,
                "ks_stat": float(ks_stat),
                "p_value": float(p_value),
            }
        except Exception as exc:  # noqa: BLE001
            results[name] = {"error": str(exc)}

    return results


# =============================================================================
# Full analysis pipeline
# =============================================================================
def run_analysis(df: pd.DataFrame, output_dir: str | Path) -> str:
    """Run the complete statistical analysis and return the report text.

    The report is also written to ``results/logs/analysis_report.txt``.

    Parameters
    ----------
    df : pd.DataFrame
        Collision dataset.
    output_dir : str or Path
        Project root.

    Returns
    -------
    str – the full analysis report (also saved to disk).
    """
    buf = StringIO()

    def _w(line: str = "") -> None:
        buf.write(line + "\n")

    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y-%m-%d %H:%M:%S UTC"
    )

    _w("=" * 68)
    _w("  PARTICLE COLLISION MONTE CARLO — STATISTICAL ANALYSIS REPORT")
    _w(f"  Generated: {timestamp}")
    _w("=" * 68)

    # ── Summary statistics ──────────────────────────────────────────────
    _w("\n1. DESCRIPTIVE STATISTICS")
    _w("-" * 68)
    summary = compute_summary_statistics(df)
    for col, s in summary.items():
        _w(f"\n  ▸ {col}")
        for stat_name, value in s.items():
            _w(f"      {stat_name:>8s} = {value:>14.4f}")

    # ── Distribution fits ───────────────────────────────────────────────
    _w("\n\n2. DISTRIBUTION FITS")
    _w("-" * 68)

    fit_columns = [
        ("particle_A_energy", "Particle A beam energy"),
        ("particle_B_energy", "Particle B beam energy"),
        ("collision_angle", "Collision angle θ"),
        ("final_energy_1", "Final-state energy 1"),
        ("final_energy_2", "Final-state energy 2"),
        ("sqrt_s", "Invariant mass √s"),
    ]

    for col, label in fit_columns:
        _w(f"\n  ▸ {label}  ({col})")
        fits = fit_distributions(df[col].to_numpy(), label=label)
        for dist_name, info in fits.items():
            if "error" in info:
                _w(f"      {dist_name:>12s}  → FAILED: {info['error']}")
            else:
                _w(f"      {dist_name:>12s}  KS = {info['ks_stat']:.6f}  "
                   f"p = {info['p_value']:.6f}  params = {info['params']}")

    # ── Energy conservation check ───────────────────────────────────────
    _w("\n\n3. ENERGY CONSERVATION CHECK")
    _w("-" * 68)
    total_beam = df["particle_A_energy"] + df["particle_B_energy"]
    total_final = df["final_energy_1"] + df["final_energy_2"]
    residual = total_beam - total_final
    _w(f"  Mean residual (E_beam − E_final) : {residual.mean():+.6e} GeV")
    _w(f"  Std  residual                    : {residual.std():.6e} GeV")
    _w(f"  Max  |residual|                  : {residual.abs().max():.6e} GeV")
    if "sqrt_s" in df.columns:
        _w(f"  Mean √s (invariant mass)         : {df['sqrt_s'].mean():.4f} GeV")
        _w(f"  Std  √s                          : {df['sqrt_s'].std():.4f} GeV")

    # ── Momentum balance ────────────────────────────────────────────────
    _w("\n\n4. TOTAL MOMENTUM BALANCE")
    _w("-" * 68)
    for comp in ("momentum_x", "momentum_y", "momentum_z"):
        _w(f"  <{comp:>12s}> = {df[comp].mean():+.4f} ± {df[comp].std():.4f} GeV/c")

    # ── Cross-correlations ──────────────────────────────────────────────
    _w("\n\n5. PEARSON CORRELATION MATRIX (selected columns)")
    _w("-" * 68)
    cols = ["particle_A_energy", "particle_B_energy", "collision_angle",
            "final_energy_1", "final_energy_2", "sqrt_s"]
    corr = df[cols].corr()
    _w(corr.to_string())

    _w("\n" + "=" * 68)
    _w("  END OF REPORT")
    _w("=" * 68)

    report = buf.getvalue()

    # ── Persist to disk ─────────────────────────────────────────────────
    log_dir = Path(output_dir) / "results" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "analysis_report.txt"
    log_path.write_text(report, encoding="utf-8")
    print(f"  📄  Analysis report saved → {log_path}")

    return report
