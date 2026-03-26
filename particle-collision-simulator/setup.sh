#!/usr/bin/env bash
# =============================================================================
# Particle Collision Simulator — Environment Setup
# =============================================================================
# Creates a Python virtual environment and installs all dependencies.
#
# Usage:
#   chmod +x setup.sh
#   ./setup.sh
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "╔══════════════════════════════════════════════════╗"
echo "║  Particle Collision Simulator — Setup            ║"
echo "╚══════════════════════════════════════════════════╝"
echo ""

# --- Create virtual environment ---
if [ ! -d "venv" ]; then
    echo "→ Creating Python virtual environment..."
    python3 -m venv venv
    echo "  ✔ Virtual environment created."
else
    echo "  ✔ Virtual environment already exists."
fi

# --- Activate ---
echo "→ Activating virtual environment..."
source venv/bin/activate

# --- Upgrade pip ---
echo "→ Upgrading pip..."
pip install --upgrade pip --quiet

# --- Install dependencies ---
echo "→ Installing dependencies from requirements.txt..."
pip install -r requirements.txt --quiet

echo ""
echo "═══════════════════════════════════════════════════"
echo "  ✔ Setup complete."
echo ""
echo "  Activate the environment with:"
echo "    source venv/bin/activate"
echo ""
echo "  Run the simulation with:"
echo "    python src/main.py --events 50000"
echo "═══════════════════════════════════════════════════"
