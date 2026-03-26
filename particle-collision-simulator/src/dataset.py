"""
dataset.py — Persistence layer for simulation data.

Handles saving (and loading) collision datasets in multiple formats:
  • CSV   — human-readable, easy to import in ROOT / Excel / R
  • NumPy — compact binary, fast I/O in Python

All files are written under ``data/raw/`` by default.

TODO 4: Support ROOT file export (HEP format) via ``uproot`` or
        ``root_numpy`` so the data integrates directly with CERN's
        ROOT / RDataFrame analysis ecosystem.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd


# =============================================================================
# Save helpers
# =============================================================================
def save_csv(df: pd.DataFrame, path: str | Path) -> Path:
    """Write the DataFrame to a gzip-compressed CSV.

    Parameters
    ----------
    df : pd.DataFrame
        Collision event dataset.
    path : str or Path
        Directory (not filename) where the file will be written.

    Returns
    -------
    Path – full path to the written file.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    filepath = path / "collision_events.csv.gz"
    df.to_csv(filepath, index=False, compression="gzip")
    print(f"  💾  CSV saved  → {filepath}  ({filepath.stat().st_size / 1e6:.2f} MB)")
    return filepath


def save_numpy(df: pd.DataFrame, path: str | Path) -> Path:
    """Write the DataFrame columns as a compressed ``.npz`` archive.

    Parameters
    ----------
    df : pd.DataFrame
    path : str or Path

    Returns
    -------
    Path – full path to the written file.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    filepath = path / "collision_events.npz"
    np.savez_compressed(
        filepath,
        **{col: df[col].to_numpy() for col in df.columns},
    )
    print(f"  💾  NPZ saved  → {filepath}  ({filepath.stat().st_size / 1e6:.2f} MB)")
    return filepath


def save_dataset(df: pd.DataFrame, output_dir: str | Path) -> None:
    """Convenience wrapper that writes both CSV and NumPy formats.

    Parameters
    ----------
    df : pd.DataFrame
        The full collision dataset.
    output_dir : str or Path
        Project root; files land in ``<output_dir>/data/raw/``.
    """
    raw_dir = Path(output_dir) / "data" / "raw"
    save_csv(df, raw_dir)
    save_numpy(df, raw_dir)


# =============================================================================
# Load helpers
# =============================================================================
def load_csv(path: str | Path) -> pd.DataFrame:
    """Load a previously saved CSV dataset."""
    return pd.read_csv(path)


def load_numpy(path: str | Path) -> dict[str, np.ndarray]:
    """Load a previously saved NPZ archive and return as dict."""
    data = np.load(path)
    return {key: data[key] for key in data.files}
