# Particle Collision Monte Carlo Simulator
![Python](https://img.shields.io/badge/Python-3.11%2B-blue)
![NumPy](https://img.shields.io/badge/NumPy-vectorized-green)
![MonteCarlo](https://img.shields.io/badge/Method-Monte%20Carlo-purple)
![License](https://img.shields.io/badge/License-MIT-yellow)

A production-quality Monte Carlo simulator for simplified high-energy particle collisions, built with Python and scientific computing libraries. Features a stunning **Next.js dashboard** with real-time particle collision animations, interactive simulations, and publication-quality visualizations.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [System Specifications](#system-specifications)
3. [Architecture](#architecture)
4. [Physics Background](#physics-background)
5. [Methodology](#methodology)
6. [How Monte Carlo Works](#how-monte-carlo-works)
7. [Experimental Configuration](#experimental-configuration)
8. [Reproducibility](#reproducibility)
9. [Validation and Verification](#validation-and-verification)
10. [Statistical Analysis](#statistical-analysis)
11. [Performance Benchmarks](#performance-benchmarks)
12. [Dashboard (Frontend)](#dashboard-frontend)
13. [Project Structure](#project-structure)
14. [Getting Started](#getting-started)
15. [Usage](#usage)
16. [Example Output](#example-output)
17. [Scientific Limitations](#scientific-limitations)
18. [Future Research Extensions](#future-research-extensions)
19. [Technology Stack](#technology-stack)
20. [Citation](#citation)
21. [Reproducible Research Workflow](#reproducible-research-workflow)
22. [Example Configuration File](#example-configuration-file)
23. [Dataset Schema Diagram](#dataset-schema-diagram)
24. [Computational Complexity](#computational-complexity)
25. [Relation to Real High-Energy Physics Tools](#relation-to-real-high-energy-physics-tools)
26. [Contributing](#contributing)


---

## Project Overview

This simulator models a **2→2 particle scattering** process: two beam particles collide and produce two final-state particles. The initial beam energies and collision angle are drawn randomly (Monte Carlo sampling), and the kinematics of the outgoing particles are computed analytically in the centre-of-mass frame using conservation of four-momentum.

**Key Features:**

- Generate **100,000+** collision events in seconds (fully vectorised with NumPy)
- Export datasets as **CSV** and **NumPy** binary files
- Produce six publication-quality plots (energy distributions, angular distributions, momentum scatter, density maps, invariant mass spectrum)
- Run automated **statistical analysis** with distribution fitting and Kolmogorov–Smirnov goodness-of-fit tests
- **Interactive Next.js dashboard** with real-time particle collision canvas animation, KaTeX formula rendering, and Recharts visualizations
- Browser-based simulation engine running entirely client-side

### Pipeline Overview

```
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│  Random Sampling │────▶│    Kinematics     │────▶│     Dataset      │────▶│  Plots + Analysis│
│   (MC engine)    │     │    (physics)      │     │    (Pandas)      │     │ (Matplotlib/Sci) │
└──────────────────┘     └──────────────────┘     └──────────────────┘     └──────────────────┘
         │                        │                        │                        │
    Monte Carlo             Energy–Momentum          CSV / NumPy            6 Publication
    Uniform + Isotropic     Conservation Laws        Binary Export          Quality Figures
    Sampling                4-Momentum Algebra       Metadata Attach        + KS Tests
```

---

## System Specifications

This project was developed and benchmarked on the following hardware and software stack:

### Hardware

| Component | Specification |
|---|---|
| **CPU** | Intel® Core™ i7-13650HX (14 cores / 20 threads) |
| **Architecture** | x86_64 |
| **CPU Frequency** | 800 MHz – 4900 MHz (turbo) |
| **L1d Cache** | 544 KiB (14 instances) |
| **L1i Cache** | 704 KiB (14 instances) |
| **L2 Cache** | 11.5 MiB (8 instances) |
| **L3 Cache** | 24 MiB (shared) |
| **RAM** | 16 GB DDR5 |
| **GPU (Integrated)** | Intel Raptor Lake-S UHD Graphics |
| **GPU (Discrete)** | NVIDIA GeForce RTX 3050 6GB Laptop GPU (GA107BM) |
| **Storage** | 159 GB LUKS-encrypted SSD (63 GB free) |
| **Hostname** | `desktop-bigboyaks` |

### Software Environment

| Software | Version |
|---|---|
| **Operating System** | Fedora Linux 43 (Workstation Edition) |
| **Kernel** | 6.18.8-200.fc43.x86_64 |
| **Python** | 3.14.3 |
| **Node.js** | 22.22.0 |
| **NumPy** | 1.26.0 |
| **SciPy** | 1.11.0 |
| **Matplotlib** | 3.8.0 |
| **Pandas** | 2.1.0 |
| **Next.js** | 14.2.x |
| **TypeScript** | 5.3.x |

### Performance Context

The 13th Gen Intel i7-13650HX is a high-performance hybrid processor with 6 P-cores (Performance) and 8 E-cores (Efficiency), making it well-suited for the vectorised NumPy operations that power this simulator. The 24 MB L3 cache allows large event arrays to remain in fast cache during computation, contributing to the measured throughput of **670,000+ events/second**.

---

## Architecture

### System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                              MONTE CARLO PROJECT                                     │
│                                                                                      │
│  ┌─────────────────────────────────────┐    ┌─────────────────────────────────────┐  │
│  │     SIMULATION ENGINE (Python)       │    │        DASHBOARD (Next.js)           │  │
│  │                                      │    │                                     │  │
│  │  ┌──────────┐    ┌──────────────┐   │    │  ┌──────────┐    ┌──────────────┐  │  │
│  │  │ main.py  │───▶│ simulation.py│   │    │  │ page.tsx │    │  Particle    │  │  │
│  │  │  (CLI)   │    │ (orchestrate)│   │    │  │  (Home)  │    │  Collision   │  │  │
│  │  └──────────┘    └──────┬───────┘   │    │  └──────────┘    │  Canvas      │  │  │
│  │                         │           │    │                  └──────────────┘  │  │
│  │       ┌─────────────────┼────────┐  │    │  ┌──────────┐    ┌──────────────┐  │  │
│  │       ▼                 ▼        ▼  │    │  │theory.tsx│    │ FormulaBlock │  │  │
│  │  ┌─────────┐  ┌──────────┐ ┌─────┐ │    │  │(Physics) │    │  (KaTeX)     │  │  │
│  │  │physics  │  │montecarlo│ │conf │ │    │  └──────────┘    └──────────────┘  │  │
│  │  │  .py    │  │   .py    │ │ig.py│ │    │                                     │  │
│  │  │Kinema-  │  │ Random   │ │Para-│ │    │  ┌──────────┐    ┌──────────────┐  │  │
│  │  │tics     │  │ Sampling │ │meter│ │    │  │sim.tsx   │    │ simulation   │  │  │
│  │  └─────────┘  └──────────┘ │s    │ │    │  │(Interact)│───▶│   .ts        │  │  │
│  │                            └─────┘ │    │  └──────────┘    │ (MC Engine)  │  │  │
│  │  ┌──────────────────────────────┐   │    │                  └──────────────┘  │  │
│  │  │        OUTPUT LAYER          │   │    │  ┌──────────┐    ┌──────────────┐  │  │
│  │  │  ┌─────────┐ ┌────────────┐  │   │    │  │results   │    │  Recharts    │  │  │
│  │  │  │dataset  │ │visualization│  │   │    │  │  .tsx    │───▶│  Plots       │  │  │
│  │  │  │  .py    │ │    .py      │  │   │    │  │(Viz)     │    │  (6 charts)  │  │  │
│  │  │  │CSV/NPZ  │ │ Matplotlib  │  │   │    │  └──────────┘    └──────────────┘  │  │
│  │  │  └─────────┘ └────────────┘  │   │    │                                     │  │
│  │  │  ┌─────────────────────────┐  │   │    └─────────────────────────────────────┘  │
│  │  │  │  analysis.py            │  │   │                                             │
│  │  │  │  KS Tests / Fitting     │  │   │                                             │
│  │  │  └─────────────────────────┘  │   │                                             │
│  │  └──────────────────────────────┘   │                                             │
│  └─────────────────────────────────────┘                                             │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

### Module Dependency Graph

```
                            ┌──────────┐
                            │ main.py  │
                            │  (CLI)   │
                            └────┬─────┘
                                 │
                            ┌────▼─────┐
                            │simulation│
                            │   .py    │
                            └────┬─────┘
                                 │
                  ┌──────────────┼──────────────┐
                  │              │              │
             ┌────▼─────┐  ┌────▼─────┐  ┌────▼─────┐
             │ physics  │  │montecarlo│  │ config   │
             │   .py    │  │   .py    │  │   .py    │
             │          │  │          │  │          │
             │ E²=p²+m² │  │ Sampling │  │ Defaults │
             │ √s calc  │  │ Engine   │  │ CLI args │
             │ CM frame │  │ RNG seed │  │ Ranges   │
             └──────────┘  └──────────┘  └──────────┘
                                 │
                  ┌──────────────┼──────────────┐
                  │              │              │
             ┌────▼─────┐  ┌────▼─────┐  ┌────▼─────┐
             │ dataset  │  │visualiza-│  │ analysis │
             │   .py    │  │ tion.py  │  │   .py    │
             │          │  │          │  │          │
             │ CSV/NPZ  │  │ 6 Plots  │  │ KS Test  │
             │ Export   │  │ Matplot  │  │ Fitting  │
             └──────────┘  └──────────┘  └──────────┘
```

### Event Processing Data Flow

```
    ┌───────────────────────────────────────────────────────────────────┐
    │                    EVENT GENERATION PIPELINE                       │
    │                                                                   │
    │  STAGE 1           STAGE 2           STAGE 3          STAGE 4    │
    │  ────────          ────────          ────────          ────────   │
    │                                                                   │
    │  ┌─────────┐      ┌─────────┐      ┌─────────┐     ┌─────────┐ │
    │  │ Sample  │      │ Compute │      │ Compute │     │ Split   │ │
    │  │ E_A,E_B │─────▶│  |p|    │─────▶│   √s    │────▶│ E₁, E₂ │ │
    │  │ θ, φ    │      │ px,py,pz│      │         │     │         │ │
    │  └─────────┘      └─────────┘      └─────────┘     └─────────┘ │
    │       │                │                │                │       │
    │  U(Emin,Emax)    √(E²−m²)        (EA+EB)²−|pA+pB|²   E·(0.5+δ)│
    │  U(cosθ)         Spherical→       Lorentz              δ~U(-0.1,│
    │  U(0,2π)         Cartesian        Invariant             0.1)    │
    │                                                                   │
    └───────────────────────────────────────────────────────────────────┘
```

### Dashboard Component Architecture

```
    ┌──────────────────────────────────────────────────────────────┐
    │                    NEXT.JS APP ROUTER                         │
    │                                                              │
    │  layout.tsx ──────────────────────────────────────────────── │
    │  ├── Navbar.tsx (client)                                     │
    │  ├── <main>                                                  │
    │  │   ├── / ──────────────── page.tsx (Home)                 │
    │  │   │   ├── ParticleCollisionCanvas.tsx  ←── Full-screen   │
    │  │   │   ├── AnimatedCounter.tsx              collision      │
    │  │   │   └── GlowCard.tsx                     animation     │
    │  │   │                                                       │
    │  │   ├── /theory ────────── theory/page.tsx                  │
    │  │   │   ├── FormulaBlock.tsx (KaTeX)                        │
    │  │   │   └── StepTimeline.tsx                                │
    │  │   │                                                       │
    │  │   ├── /simulation ────── simulation/page.tsx              │
    │  │   │   ├── Config Panel (sliders, inputs)                  │
    │  │   │   ├── Recharts (AreaChart, ScatterChart)              │
    │  │   │   └── Event Table Preview                             │
    │  │   │                                                       │
    │  │   └── /results ───────── results/page.tsx                 │
    │  │       ├── 5 Stat Cards (AnimatedCounter)                  │
    │  │       ├── 5 Charts (Bar, Area, Scatter)                   │
    │  │       └── Physics Validation Summary                      │
    │  │                                                           │
    │  └── Footer.tsx                                              │
    │                                                              │
    │  lib/simulation.ts ─── Browser-side MC engine                │
    │  (splitmix32 RNG, kinematics, histogram, correlation)        │
    └──────────────────────────────────────────────────────────────┘
```

---

## Physics Background

### Energy–Momentum Relation

In special relativity a particle of rest mass *m* and 3-momentum **p** has total energy:

```
E² = |p|²c² + m²c⁴
```

In **natural units** (ℏ = c = 1) this simplifies to `E² = p² + m²`.

### Centre-of-Mass Frame

For a collision of particles A and B the **invariant mass** (or CM energy) is:

```
√s = √[(E_A + E_B)² − |p_A + p_B|²]
```

This is the total energy available to create new particles. In the CM frame the total 3-momentum is zero, so the two outgoing particles share √s according to:

```
E₁* = (s + m₁² − m₂²) / (2√s)
E₂* = √s − E₁*
```

When the final-state particles are massless (e.g. photons) each simply gets √s / 2.

### Simplified Model

This simulator uses a **simplified but physically motivated** model:

1. Beam energies are drawn uniformly from a configurable range.
2. Collision angles follow an isotropic distribution (uniform in cos θ).
3. Momentum vectors are computed from spherical-to-Cartesian decomposition.
4. Final-state energies are derived from four-momentum conservation with a small stochastic asymmetry.

---

## Methodology

### Event Generation Workflow

```
    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
    │  Particle    │    │   Energy    │    │   Angle     │
    │  Initialize  │───▶│  Sampling   │───▶│  Generation │
    └─────────────┘    └─────────────┘    └──────┬──────┘
                                                  │
    ┌─────────────┐    ┌─────────────┐    ┌──────▼──────┐
    │  Dataset    │◀───│  Collision  │◀───│  Momentum   │
    │  Recording  │    │  Resolution │    │  Calculation│
    └─────────────┘    └─────────────┘    └─────────────┘
```

### Random Sampling Strategy

**Beam Energy Sampling**: `E ~ U(E_min, E_max)` — uniform coverage of the energy range.

**Angular Distribution**: `cos θ ~ U(-1, 1)` — isotropic in the CM frame. `φ ~ U(0, 2π)` — full azimuthal coverage.

### Collision Modeling Steps

| Step | Operation | Formula |
|---|---|---|
| 1 | Three-momentum magnitude | `|p| = √(E² − m²)` |
| 2 | Cartesian decomposition | `px = p·sinθ·cosφ, py = p·sinθ·sinφ, pz = p·cosθ` |
| 3 | Invariant mass | `s = (E_A + E_B)² − |p_A + p_B|²` |
| 4 | Final-state energies | `E₁ = (E_A + E_B)·(0.5 + δ), E₂ = (E_A + E_B) − E₁` |

---

## How Monte Carlo Works

**Monte Carlo methods** estimate deterministic quantities through repeated random sampling.

| Application | Description |
|---|---|
| **Event generation** | Produce pseudo-data mimicking detector output |
| **Phase-space integration** | Evaluate high-dimensional integrals numerically |
| **Uncertainty propagation** | Sample nuisance parameters to estimate systematic errors |

### Algorithm (this simulator)

```
for each event i = 1 … N:
    E_A  ~ Uniform(E_min, E_max)
    E_B  ~ Uniform(E_min, E_max)
    θ    ~ Isotropic (uniform in cos θ)
    φ    ~ Uniform(0, 2π)

    compute |p_A|, |p_B|  from E² = p² + m²
    compute (px, py, pz)  via spherical → Cartesian
    compute √s            (invariant mass)
    compute E₁*, E₂*      (final-state energies)
```

The full simulation is **vectorised** — all *N* events are computed simultaneously with NumPy, avoiding Python-level loops entirely.

### Statistical Convergence

```
    Error ──┐
            │  ╲
            │    ╲
            │      ╲───── σ/√N
            │         ╲
            │           ╲──────────────
            │
            └───────────────────────▶ N (events)
           10²    10³    10⁴    10⁵
```

The MC estimator uncertainty decreases as `1/√N` regardless of dimensionality — a key advantage over quadrature in high dimensions.

---

## Experimental Configuration

| Parameter | Flag | Default | Description |
|---|---|---|---|
| Events | `--events`, `-n` | 100,000 | Total collision events |
| Min energy | `--energy-min` | 1.0 GeV | Lower bound of beam energy |
| Max energy | `--energy-max` | 1000.0 GeV | Upper bound of beam energy |
| Seed | `--seed` | 42 | Random seed (−1 = non-deterministic) |
| Save data | `--save-dataset` | True | Export CSV + NPZ |
| Plots | `--plot` | True | Generate visualizations |

---

## Reproducibility

- **Deterministic seeds**: Same seed → identical output across runs
- **Pinned dependencies**: `requirements.txt` locks all package versions
- **Isolated environment**: `setup.sh` creates a fresh venv

```bash
python src/main.py --events 10000 --seed 42
python src/main.py --events 10000 --seed 42  # ← Identical output
```

---

## Validation and Verification

| Check | Method | Pass Criterion |
|---|---|---|
| Energy conservation | Direct subtraction | Residual < 10⁻¹² GeV |
| Momentum conservation | Vector magnitude | < 10⁻¹² GeV/c |
| cos(θ) uniformity | KS test | p-value > 0.01 |
| φ uniformity | KS test | p-value > 0.01 |
| Invariant mass | Cross-check | Relative diff < 10⁻¹⁰ |

### Conservation Law Verification

```
    E_A + E_B  ═══════════════  E₁ + E₂
    (initial)                   (final)

    Residual:  |ΔE| < 10⁻¹⁷ GeV  ✓  (machine precision)
```

---

## Statistical Analysis

### Distribution Fitting

| Distribution | Physical Context | Parameters |
|---|---|---|
| Gaussian | Central limit behaviour | μ, σ |
| Log-normal | Positive-definite quantities | μ, σ |
| Exponential | Decay processes | λ |
| Uniform | Phase-space sampling | a, b |

### Goodness-of-Fit

**Kolmogorov-Smirnov Test**: `D_n = sup_x |F_n(x) − F(x)|`

Quantifies agreement between empirical and theoretical distributions.

---

## Performance Benchmarks

*Measured on Intel i7-13650HX (14C/20T), 16 GB DDR5*

| Events | Time | Throughput | Memory |
|---|---|---|---|
| 1,000 | ~0.005 s | ~200,000 ev/s | ~1 MB |
| 10,000 | ~0.02 s | ~500,000 ev/s | ~5 MB |
| 100,000 | ~0.15 s | ~650,000 ev/s | ~40 MB |
| 1,000,000 | ~1.5 s | ~670,000 ev/s | ~350 MB |
| 10,000,000 | ~15 s | ~670,000 ev/s | ~3.5 GB |

### Performance Scaling

```
    Throughput (events/s)
    700k ─┐                    ┌────────────────────
          │                   ╱
    600k ─┤                 ╱
          │               ╱
    500k ─┤             ╱
          │           ╱
    200k ─┤         ╱
          │       ╱
      0 ──┴──────┴──────┬──────┬──────┬──────▶
          10²    10³   10⁴    10⁵    10⁶    10⁷
                        Events (N)
```

Throughput saturates around 670k events/s — limited by memory bandwidth at very high event counts. Scaling is linear: `T(N) ≈ a + bN`.

---

## Dashboard (Frontend)

The dashboard is a **Next.js 14** TypeScript application with a premium dark-mode UI:

### Pages

| Page | Route | Description |
|---|---|---|
| **Home** | `/` | Hero with full-screen particle collision canvas animation, animated stats, pipeline overview |
| **Theory** | `/theory` | 6 physics sections with KaTeX formula rendering, step-by-step algorithm timeline |
| **Simulation** | `/simulation` | Interactive config panel with sliders, real-time Monte Carlo engine, live Recharts plots |
| **Results** | `/results` | 5-chart visualization dashboard with 100,000 auto-generated events |

### Dashboard Tech Stack

| Library | Purpose |
|---|---|
| **Next.js 14** | React framework with App Router |
| **TypeScript** | Type safety |
| **Tailwind CSS** | Utility-first styling |
| **Framer Motion** | Scroll-reveal, entrance, and hover animations |
| **KaTeX** | LaTeX formula rendering |
| **Recharts** | Interactive charts (Area, Bar, Scatter) |
| **Lucide React** | Icon library |
| **Canvas API** | Full particle collision simulator animation |

### Key Animations

- **Particle Collision Canvas**: Full-viewport WebGL-style canvas with two beams of particles colliding at center, collision spark bursts, particle trails, ambient particles with mouse interaction
- **Scroll-Reveal**: All sections animate in on scroll with staggered delays
- **Animated Counters**: Number counting animations triggered on viewport entry
- **Glow Cards**: Glassmorphism cards with animated border glow on hover
- **Active Nav Pill**: Spring-animated indicator follows active route
- **Progress Bar**: Gradient-animated progress bar during simulation

---

## Project Structure

```
monte-carlo/
│
├── particle-collision-simulator/
│   ├── src/
│   │   ├── main.py              # CLI entry point
│   │   ├── config.py            # Simulation parameters & constants
│   │   ├── simulation.py        # High-level orchestrator
│   │   ├── physics.py           # Relativistic kinematics
│   │   ├── montecarlo.py        # Monte Carlo sampling engine
│   │   ├── dataset.py           # CSV / NumPy persistence
│   │   ├── visualization.py     # Matplotlib plot suite
│   │   └── analysis.py          # Statistical analysis & reporting
│   ├── data/
│   │   ├── raw/                 # Generated datasets
│   │   └── processed/           # Downstream analysis
│   ├── results/
│   │   ├── plots/               # PNG + PDF figures
│   │   └── logs/                # Analysis reports
│   ├── requirements.txt
│   └── setup.sh                 # One-command environment setup
│
├── dashboard/
│   ├── src/
│   │   ├── app/
│   │   │   ├── layout.tsx       # Root layout + metadata
│   │   │   ├── page.tsx         # Home page (hero + particle canvas)
│   │   │   ├── theory/page.tsx  # Theory & formulas
│   │   │   ├── simulation/      # Interactive simulation
│   │   │   │   └── page.tsx
│   │   │   ├── results/         # Visualization dashboard
│   │   │   │   └── page.tsx
│   │   │   └── globals.css      # Theme + glassmorphism + animations
│   │   ├── components/
│   │   │   ├── ParticleCollisionCanvas.tsx   # Full collision animation
│   │   │   ├── Navbar.tsx                     # Animated navigation
│   │   │   ├── Footer.tsx                     # Tech stack badges
│   │   │   ├── FormulaBlock.tsx               # KaTeX math blocks
│   │   │   ├── AnimatedCounter.tsx            # Number animations
│   │   │   ├── GlowCard.tsx                   # Glassmorphism card
│   │   │   └── StepTimeline.tsx               # Algorithm steps
│   │   └── lib/
│   │       └── simulation.ts    # Browser MC engine (TypeScript)
│   ├── tailwind.config.js
│   ├── next.config.js
│   ├── tsconfig.json
│   └── package.json
│
└── README.md                    # ← You are here
```

---

## Getting Started

### Prerequisites

- **Python 3.11+** (with pip)
- **Node.js 18+** (with npm)

### Simulator Setup

```bash
cd particle-collision-simulator
chmod +x setup.sh
./setup.sh
source venv/bin/activate
```

### Dashboard Setup

```bash
cd dashboard
npm install
npm run dev        # Development server → http://localhost:3000
npm run build      # Production build
npm run start      # Serve production build
```

---

## Usage

### Python Simulator

```bash
# Default run (100k events)
python src/main.py

# Custom configuration
python src/main.py --events 50000 --energy-min 10 --energy-max 500 --seed 42

# Data only (no plots)
python src/main.py --events 1000000 --no-plot
```

### Dashboard

Navigate to `http://localhost:3000` and explore:

- **Home**: Watch the particle collision animation
- **Theory**: Read the physics with LaTeX formulas
- **Simulation**: Configure and run simulations interactively
- **Results**: View auto-generated visualization dashboard

---

## Example Output

### Console Output (excerpt)

```
╔══════════════════════════════════════════════════════╗
║   Particle Collision Monte Carlo Simulator           ║
╚══════════════════════════════════════════════════════╝

  Events         :       100,000
  Energy range   : [1.0, 1000.0] GeV
  ⚙  MonteCarloEngine  seed=42

  → Sampling event parameters …
    ✔ 100,000 events generated in 0.15 s (650,000 events/s)

  💾  CSV saved  → data/raw/collision_events.csv.gz
  💾  NPZ saved  → data/raw/collision_events.npz

  📊  6 plots saved → results/plots/
```

### Generated Plots

| Plot | Description |
|---|---|
| `energy_distribution.png` | Beam + final-state energy histograms |
| `angular_distribution.png` | θ and cos θ distributions |
| `momentum_scatter.png` | px–py and px–pz scatter |
| `event_density.png` | 2-D heat map of (E_A, E_B) |
| `final_energy_correlation.png` | Anti-correlation from conservation |
| `invariant_mass.png` | Reconstructed √s spectrum |

### Dataset Columns

| Column | Unit | Description |
|---|---|---|
| `event_id` | — | Sequential event identifier |
| `particle_A_energy` | GeV | Beam energy of particle A |
| `particle_B_energy` | GeV | Beam energy of particle B |
| `collision_angle` | rad | Polar scattering angle θ |
| `momentum_x` | GeV/c | Total system px |
| `momentum_y` | GeV/c | Total system py |
| `momentum_z` | GeV/c | Total system pz |
| `final_energy_1` | GeV | Outgoing particle 1 energy |
| `final_energy_2` | GeV | Outgoing particle 2 energy |

---

## Scientific Limitations

| Limitation | Detail |
|---|---|
| **No QFT** | Classical kinematics only — no scattering amplitudes |
| **No detector** | No geometric acceptance, resolution smearing, or tracking |
| **Isotropic scattering** | No angular-dependent cross-sections |
| **No spin** | No spin correlations between initial/final states |
| **No radiation** | No QED/QCD radiative corrections |
| **2→2 only** | No multi-particle final states |

These simplifications are intentional — the simulator prioritises **conceptual clarity** of Monte Carlo methodology over physical accuracy.

---

## Future Research Extensions

| Extension | Description |
|---|---|
| **Lorentz boosts** | Full four-vector arithmetic and frame transformations |
| **Differential cross-sections** | Rutherford, Compton, pair production |
| **Detector simulation** | Energy resolution, angular smearing, GEANT4 integration |
| **ROOT export** | Standard HEP data format via `uproot` |
| **GPU acceleration** | CUDA kernels for massive parallelism |
| **Distributed MC** | Multi-core and grid computing support |
| **ML classification** | Neural network event classification |

---

## Technology Stack

### Simulation Engine

| Library | Role |
|---|---|
| **NumPy** | Vectorised numerics, random sampling |
| **SciPy** | Distribution fitting, KS tests |
| **Matplotlib** | Publication-quality figures |
| **Pandas** | Tabular data management |

### Dashboard

| Library | Role |
|---|---|
| **Next.js 14** | React framework, App Router |
| **TypeScript** | Type safety across all components |
| **Tailwind CSS 3** | Utility-first responsive styling |
| **Framer Motion** | Animation library |
| **KaTeX** | LaTeX math rendering |
| **Recharts** | Composable chart library |
| **Lucide React** | SVG icon set |

---

## Citation

If you use this simulator in research or academic work, please cite:

```bibtex
@software{singh2026montecarlo,
  author  = {Abhinav Kumar Singh},
  title   = {Particle Collision Monte Carlo Simulator},
  year    = {2026},
  url     = {https://github.com/abhinav29102005/monte-carlo}
}
```

This makes the repo **usable in academic publications**.

---

## Reproducible Research Workflow

Typical scientific workflow using this simulator:

1. Configure experiment parameters.
2. Generate collision events using Monte Carlo sampling.
3. Export datasets (CSV / NPZ).
4. Perform statistical analysis.
5. Visualize distributions and correlations.
6. Interpret physical results.

Because the simulator supports deterministic seeds and pinned dependencies,
all experiments can be reproduced exactly across machines.

---

## Example Configuration File

Instead of only CLI flags, add YAML config.

### Example Configuration

`simulation_config.yaml`

```yaml
events: 100000
energy_min: 1
energy_max: 1000
seed: 42
save_dataset: true
generate_plots: true
```

Run simulation:

```bash
python src/main.py --config simulation_config.yaml
```

Research projects typically rely on **configuration-driven experiments**.

---

## Dataset Schema Diagram

```text
Collision Dataset Schema

event_id
│
├── particle_A_energy (GeV)
├── particle_B_energy (GeV)
│
├── collision_angle (rad)
│
├── momentum
│   ├── px (GeV/c)
│   ├── py (GeV/c)
│   └── pz (GeV/c)
│
└── final_state
    ├── energy_1 (GeV)
    └── energy_2 (GeV)
```

---

## Computational Complexity

Because the simulation is fully vectorised with NumPy, the complexity is:

```text
O(N)
```

where N is the number of generated events.

No Python-level loops are used, allowing high throughput
(>650k events per second on modern CPUs).

---

## Relation to Real High-Energy Physics Tools

This simulator demonstrates simplified versions of techniques used in
modern particle physics frameworks such as:

• GEANT4 — detector simulation framework used by CERN  
• ROOT — data analysis framework for high-energy physics  
• Pythia — event generator for particle collisions  

While simplified, the architecture mirrors the workflow of real
HEP simulation pipelines.

This anchors your project to real research infrastructure.

---

## Contributing

Contributions are welcome.

Possible areas for improvement include:

• additional physics models  
• relativistic scattering processes  
• detector simulation  
• ROOT dataset export  
• GPU acceleration  

Please open an issue or submit a pull request.


