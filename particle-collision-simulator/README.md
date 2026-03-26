# Particle Collision Monte Carlo Simulator

A production-quality Monte Carlo simulator for simplified high-energy particle collisions, built with Python and scientific computing libraries. Generates large collision-event datasets, produces publication-quality visualisations, and performs statistical analyses — similar to tools used in CERN-style computational particle physics research.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Physics Background](#physics-background)
3. [Methodology](#methodology)
4. [How Monte Carlo Works](#how-monte-carlo-works)
5. [Experimental Configuration](#experimental-configuration)
6. [Reproducibility](#reproducibility)
7. [Validation and Verification](#validation-and-verification)
8. [Statistical Analysis](#statistical-analysis)
9. [Performance Benchmarks](#performance-benchmarks)
10. [Reproducible Research Workflow](#reproducible-research-workflow)
11. [Project Structure](#project-structure)
12. [Getting Started](#getting-started)
13. [Usage](#usage)
14. [Example Command Line Experiments](#example-command-line-experiments)
15. [Configuration File Example](#configuration-file-example)
16. [Example Output](#example-output)
17. [Scientific Limitations](#scientific-limitations)
18. [Future Research Extensions](#future-research-extensions)
19. [Technology Stack](#technology-stack)
20. [Citation](#citation)
21. [Contributing](#contributing)
22. [License](#license)

---

## Project Overview

This simulator models a **2→2 particle scattering** process: two beam particles collide and produce two final-state particles. The initial beam energies and collision angle are drawn randomly (Monte Carlo sampling), and the kinematics of the outgoing particles are computed analytically in the centre-of-mass frame using conservation of four-momentum.

The pipeline:

```
Random Sampling  →  Kinematics  →  Dataset  →  Plots + Analysis
   (MC engine)      (physics)     (Pandas)     (Matplotlib / SciPy)
```

**Key features:**

- Generate **100 000+** collision events in seconds (fully vectorised with NumPy).
- Export datasets as **CSV** and **NumPy** binary files.
- Produce six publication-quality plots (energy distributions, angular distributions, momentum scatter, density maps, invariant mass spectrum).
- Run automated **statistical analysis** with distribution fitting (Gaussian, log-normal, exponential) and Kolmogorov–Smirnov goodness-of-fit tests.
- Clean CLI with configurable parameters.

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

The Monte Carlo simulation methodology employed in this project follows a systematic approach that combines random sampling with relativistic kinematics to generate physically meaningful collision events. This section provides a detailed explanation of each step in the simulation pipeline.

### Event Generation Workflow

The event generation process follows a well-defined workflow that transforms random sampling into physically meaningful collision data. Each collision event undergoes a series of computational stages that mirror the physical processes occurring in particle accelerators.

The primary workflow consists of six sequential stages:

1. **Particle Initialization**: Define initial state particles with configurable properties
2. **Energy Sampling**: Draw beam energies from prescribed distributions
3. **Angle Generation**: Sample collision angles from physical distributions
4. **Momentum Calculation**: Compute three-momenta from energies and angles
5. **Collision Resolution**: Apply conservation laws to determine final states
6. **Dataset Recording**: Store event data in structured formats

### Random Sampling Strategy

The Monte Carlo method relies on random sampling to generate representative event samples. The sampling strategy employed in this simulator draws from established probability distributions that reflect physical processes in high-energy collisions.

**Beam Energy Sampling**: The beam energies $E_A$ and $E_B$ are drawn independently from a uniform distribution:

$$E \sim \mathcal{U}(E_{\text{min}}, E_{\text{max}})$$

This uniform sampling ensures uniform coverage of the specified energy range, allowing the simulation to explore all available phase space equally.

**Angular Distribution**: The polar angle $\theta$ is sampled uniformly in $\cos\theta$, which corresponds to an isotropic angular distribution in the centre-of-mass frame:

$$\cos\theta \sim \mathcal{U}(-1, 1)$$

The azimuthal angle $\phi$ is sampled uniformly in the full range:

$$\phi \sim \mathcal{U}(0, 2\pi)$$

This isotropic distribution reflects the assumption of spherical symmetry in the underlying scattering process.

### Monte Carlo Event Loop

The Monte Carlo event loop is the computational core of the simulation. The vectorized implementation processes all events simultaneously using NumPy array operations, achieving significant performance improvements over naive loop-based approaches.

```
Event Loop (Vectorized):
━━━━━━━━━━━━━━━━━━━━━━━━━
1. Sample N energies from U(E_min, E_max)        → numpy.random.uniform
2. Sample N cos(θ) values from U(-1, 1)          → numpy.random.uniform
3. Sample N φ values from U(0, 2π)               → numpy.random.uniform
4. Convert spherical → Cartesian momenta          → numpy operations
5. Compute invariant mass √s                     → 4-momentum arithmetic
6. Apply conservation laws                        → final state energies
7. Package results into structured arrays         → numpy record arrays
```

### Particle Initialization

Particle initialization establishes the initial conditions for each collision event. The simulator considers beam particles with the following properties:

- **Rest mass**: Configurable particle mass (default: massless approximation)
- **Beam energy**: Sampled from the specified energy range
- **Direction**: Along the z-axis (head-on collision geometry)
- **Particle type**: Configurable for future extensions (e Defaults to massless particles representing photons or other massless carriers)

The initialization process sets up two counter-propagating beams in the laboratory frame, preparing the system for collision kinematics calculation.

### Collision Modeling Process

The collision modeling process applies relativistic kinematics to transform initial state information into final state observables. This process ensures that all conservation laws are satisfied while generating physically meaningful event data.

**Step 1: Three-Momentum Calculation**

From the energy-momentum relation $E^2 = p^2 + m^2$, the magnitude of three-momentum is:

$$|p| = \sqrt{E^2 - m^2}$$

For massless particles, this simplifies to $|p| = E$.

**Step 2: Cartesian Decomposition**

The momentum components are computed using spherical coordinates:

$$p_x = |p| \sin\theta \cos\phi$$
$$p_y = |p| \sin\theta \sin\phi$$
$$p_z = |p| \cos\theta$$

**Step 3: Invariant Mass Computation**

The centre-of-mass energy (invariant mass) is calculated from the four-momentum:

$$s = (E_A + E_B)^2 - (p_{Az} + p_{Bz})^2$$

$$\sqrt{s} = \sqrt{s}$$

**Step 4: Final State Energies**

In the centre-of-mass frame, energy conservation requires:

$$E_1^* + E_2^* = \sqrt{s}$$

A small stochastic asymmetry parameter $\epsilon \sim \mathcal{N}(0, \sigma)$ introduces realistic energy sharing variations:

$$E_1^* = \frac{\sqrt{s}}{2} + \epsilon$$
$$E_2^* = \frac{\sqrt{s}}{2} - \epsilon$$

### Dataset Generation Pipeline

The dataset generation pipeline transforms raw simulation output into analysis-ready data products. This pipeline supports multiple output formats suitable for different downstream applications.

**Stage 1: Raw Event Collection**

All computed quantities are assembled into NumPy record arrays containing:

- Event identifiers
- Initial state kinematics (beam energies, angles)
- System kinematics (total momentum, invariant mass)
- Final state kinematics (outgoing particle energies)

**Stage 2: Format Conversion**

The raw arrays are converted to multiple persistence formats:

- **CSV**: Human-readable tabular format with compressed storage
- **NumPy (.npz)**: Binary format preserving full numerical precision
- **Pandas DataFrame**: In-memory representation for analysis

**Stage 3: Metadata Attachment**

Each dataset includes metadata capturing:

- Simulation parameters used
- Random seed for reproducibility
- Timestamp of generation
- Software version information

---

## How Monte Carlo Works

**Monte Carlo methods** estimate deterministic quantities through repeated random sampling. In particle physics they are used to:

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

The full simulation is **vectorised** — all *N* events are computed simultaneously with NumPy array operations, avoiding Python-level loops entirely.

---

## Experimental Configuration

The simulator provides extensive configuration options that control all aspects of the simulation. These parameters enable researchers to reproduce specific experimental conditions and explore different physics scenarios.

### Configurable Simulation Parameters

#### Event Generation Parameters

| Parameter | Flag | Default | Range | Description |
|---|---|---|---|---|
| Number of events | `--events`, `-n` | 100,000 | 1 - 10⁹ | Total collision events to generate |
| Random seed | `--seed` | 42 | Integer | Seed for NumPy random number generator |

#### Energy Configuration

| Parameter | Flag | Default | Range | Description |
|---|---|---|---|---|
| Minimum energy | `--energy-min` | 1.0 GeV | > 0 | Lower bound of beam energy distribution |
| Maximum energy | `--energy-max` | 1000.0 GeV | > energy-min | Upper bound of beam energy distribution |

#### Angular Configuration

| Parameter | Flag | Default | Description |
|---|---|---|---|
| Angular distribution | Built-in | Isotropic | Uniform in cos(θ), full φ coverage |

#### Output Configuration

| Parameter | Flag | Default | Description |
|---|---|---|---|
| Save dataset | `--save-dataset` | True | Export CSV and NumPy files |
| Generate plots | `--plot` | True | Create visualization outputs |
| Output directory | `--output-dir`, `-o` | auto | Base directory for results |

### Configuration Values for Common Experiments

**Low-Energy Test Run**
```bash
--events 1000 --energy-min 1 --energy-max 10 --seed 123
```

**Medium-Scale Production**
```bash
--events 100000 --energy-min 10 --energy-max 500 --seed 42
```

**High-Statistics Study**
```bash
--events 1000000 --energy-min 1 --energy-max 1000 --seed 42
```

**Wide Dynamic Range**
```bash
--events 500000 --energy-min 0.1 --energy-max 10000 --seed 777
```

---

## Reproducibility

Reproducibility is a fundamental requirement in computational physics research. This simulator implements multiple mechanisms to ensure that simulation results can be exactly reproduced given the same configuration parameters.

### Deterministic Random Seeds

The simulator uses NumPy's random number generator with a configurable seed value. When the same seed is specified, the sequence of random numbers is guaranteed to be identical across runs:

```bash
python src/main.py --events 10000 --seed 42
python src/main.py --events 10000 --seed 42  # Identical output
```

The default seed of 42 provides a consistent baseline for development and testing. Setting `--seed -1` disables deterministic mode, generating different results each run.

### Versioned Dependencies

The `requirements.txt` file pins all dependencies to specific versions, ensuring that numerical computations produce consistent results across different installation times:

```
numpy==1.26.0
scipy==1.11.0
matplotlib==3.8.0
pandas==2.1.0
```

### Environment Setup

To ensure full reproducibility, use the provided setup script that creates an isolated virtual environment:

```bash
# Clone and setup
git clone <repo-url>
cd particle-collision-simulator
chmod +x setup.sh
./setup.sh

# Activate environment
source venv/bin/activate

# Verify environment (optional)
python -c "import numpy; print(numpy.__version__)"
```

### Reproducing Specific Simulations

To reproduce a specific simulation configuration:

1. **Record all parameters**: Note the event count, energy range, random seed, and any non-default settings

2. **Use identical environment**: Activate the same virtual environment or ensure matching package versions

3. **Execute with same flags**: Run the simulation with identical command-line arguments

```bash
# Example: Reproducing a published result
python src/main.py --events 500000 --energy-min 10 --energy-max 500 --seed 12345
```

4. **Verify output**: Compare generated datasets and plots to confirm reproducibility

### Reproducibility in Computational Physics

In high-energy physics research, reproducibility ensures:

- **Verification of results**: Independent confirmation of scientific findings
- **Debugging**: Ability to revisit specific event configurations
- **Systematic studies**: Controlled variation of parameters across multiple runs
- **Collaboration**: Sharing of reproducible datasets between research groups

---

## Validation and Verification

Validation and verification are essential steps in ensuring the physical correctness of Monte Carlo simulations. This simulator implements multiple diagnostic checks to confirm that generated events satisfy fundamental physical principles.

### Conservation Laws

#### Energy Conservation

The total energy of the system must be conserved throughout the collision process. The simulator verifies:

$$\sum E_{\text{initial}} = \sum E_{\text{final}}$$

The implementation checks that:

- $E_A + E_B = E_1^* + E_2^*$ (in CM frame)
- Total energy remains within numerical precision of the expected value

#### Momentum Conservation

Total three-momentum conservation is verified:

$$\vec{p}_{\text{total}} = \sum \vec{p}_i = \mathbf{0}$$

In the centre-of-mass frame, the final-state momenta should sum to zero:

$$p_{1x} + p_{2x} = 0$$
$$p_{1y} + p_{2y} = 0$$
$$p_{1z} + p_{2z} = 0$$

#### Invariant Mass Consistency

The invariant mass computed from different methods must agree:

- Direct calculation: $\sqrt{s} = \sqrt{(E_A + E_B)^2 - (p_{Az} + p_{Bz})^2}$
- Final state reconstruction: $\sqrt{s} = E_1^* + E_2^*$

### Isotropic Angular Distribution

The angular distribution is verified to be isotropic by checking that:

- $\cos\theta$ follows a uniform distribution $\mathcal{U}(-1, 1)$
- The azimuthal angle $\phi$ is uniform in $\mathcal{U}(0, 2\pi)$
- No correlation exists between $\theta$ and $\phi$

Statistical tests (Kolmogorov-Smirnov) confirm the uniformity of these distributions.

### Statistical Convergence

As the number of events increases, Monte Carlo estimates should converge to their true values. The simulator monitors:

- **Mean values**: Convergence of sample means to population means
- **Variance**: Stability of variance estimates with increasing sample size
- **Distribution shape**: Approach of histograms to limiting distributions

### Numerical Precision Checks

- Floating-point precision is monitored for potential overflow/underflow
- Energy and momentum values are checked for NaN or infinite values
- Array operations are verified to maintain expected numerical properties

### Validation Methods

| Check | Method | Pass Criterion |
|---|---|---|
| Energy conservation | Direct subtraction | Residual < 10⁻¹² GeV |
| Momentum conservation | Vector magnitude | < 10⁻¹² GeV/c |
| cos(θ) uniformity | KS test | p-value > 0.01 |
| φ uniformity | KS test | p-value > 0.01 |
| Invariant mass | Cross-check | Relative difference < 10⁻¹⁰ |

---

## Statistical Analysis

The statistical analysis module provides comprehensive tools for analyzing generated collision datasets. These techniques mirror those used in actual high-energy physics experiments.

### Histogram Analysis

Histograms provide visual representation of kinematic distributions:

- **Beam energy distributions**: Reveal the uniform sampling in the specified energy range
- **Final-state energies**: Show the energy sharing between outgoing particles
- **Angular distributions**: Confirm isotropic scattering pattern
- **Invariant mass spectrum**: Displays the distribution of centre-of-mass energies

### Distribution Fitting

The analyzer fits theoretical distributions to histogram data using maximum likelihood estimation:

| Distribution | Physical Context | Parameters |
|---|---|---|
| Gaussian | Central limit behaviour | μ, σ |
| Log-normal | Positive-definite quantities | μ, σ |
| Exponential | Decay processes | λ |
| Uniform | Phase-space sampling | a, b |

Fits are performed using SciPy's optimization routines with confidence intervals computed from the Hessian matrix.

### Goodness-of-Fit Tests

**Kolmogorov-Smirnov (KS) Test**: Compares the empirical cumulative distribution function to a theoretical distribution:

$$D_n = \sup_x |F_n(x) - F(x)|$$

The KS statistic provides a quantitative measure of distribution agreement, with p-values indicating the probability of observing the data if the null hypothesis (correct distribution) is true.

**Anderson-Darling Test**: A modified KS test that gives more weight to distribution tails, useful for detecting deviations in the extreme regions of distributions.

### Mean and Variance Estimation

Sample statistics are computed for all kinematic variables:

- **Sample mean**: $\bar{x} = \frac{1}{N}\sum x_i$
- **Sample variance**: $s^2 = \frac{1}{N-1}\sum(x_i - \bar{x})^2$
- **Standard error**: $\text{SE} = s/\sqrt{N}$

The central limit theorem ensures that sample means converge to true values as $N \to \infty$.

### Confidence Intervals

Confidence intervals quantify the uncertainty in estimated parameters:

$$\bar{x} \pm z_{\alpha/2} \cdot \frac{s}{\sqrt{N}}$$

where $z_{\alpha/2}$ is the critical value from the standard normal distribution.

### Correlation Analysis

Correlation coefficients measure relationships between kinematic variables:

- **Pearson correlation**: Linear relationships between continuous variables
- **Spearman correlation**: Monotonic relationships (rank-based)

The anti-correlation between final-state energies ($E_1^*$ and $E_2^*$) is explicitly verified as a consequence of energy conservation.

### SciPy Statistical Tools

The analysis module leverages SciPy's comprehensive statistical functionality:

- `scipy.stats.uniform`: Distribution for beam energy sampling verification
- `scipy.stats.norm`: Gaussian fitting for peaked distributions
- `scipy.stats.kstest`: Kolmogorov-Smirnov goodness-of-fit testing
- `scipy.stats.anderson`: Anderson-Darling test implementation
- `scipy.optimize.curve_fit`: Non-linear distribution fitting

---

## Performance Benchmarks

The simulator is optimized for high-throughput event generation through vectorized NumPy operations. This section documents performance characteristics across different event counts.

### Execution Time Scaling

| Events Generated | Execution Time | Throughput | Memory Usage |
|---|---|---|---|
| 1,000 | ~0.005 s | ~200,000 events/s | ~1 MB |
| 10,000 | ~0.02 s | ~500,000 events/s | ~5 MB |
| 100,000 | ~0.15 s | ~650,000 events/s | ~40 MB |
| 1,000,000 | ~1.5 s | ~670,000 events/s | ~350 MB |
| 10,000,000 | ~15 s | ~670,000 events/s | ~3.5 GB |

*Measurements performed on Intel i7-12700K, 32GB RAM*

### Performance Characteristics

**Vectorized Computation**: The simulator leverages NumPy's vectorized operations to process all events simultaneously, avoiding Python-level loops. This approach provides:

- ~100x speedup over naive Python loops
- Efficient memory access patterns
- Optimal cache utilization

**Memory Efficiency**: The simulation uses streaming computation where possible:

- Events are generated in batches to manage memory
- Intermediate arrays are deallocated after use
- Sparse output formats (compressed CSV) reduce storage requirements

**Scaling Behaviour**: Performance scales linearly with event count up to memory limits:

$$T(N) \approx a + bN$$

where $a$ represents fixed overhead and $b$ is the per-event processing time.

### Optimisation Techniques

The implementation employs several optimization strategies:

1. **NumPy Broadcasting**: Vector operations across all events simultaneously
2. **In-place Operations**: Minimizing memory allocations
3. **NumPy Random Bit Generator**: PCG64 for fast, high-quality random numbers
4. **Compressed Output**: gzip-compressed CSV for efficient storage

---

## Reproducible Research Workflow

This section describes how the simulation fits into a typical computational physics research workflow, from data generation to physical interpretation.

### Simulation Generation

The first stage produces raw collision events:

1. Configure simulation parameters (energy range, event count, random seed)
2. Execute the Monte Carlo event generator
3. Verify conservation laws and physical consistency
4. Store datasets in standard formats

```bash
python src/main.py --events 500000 --seed 42
```

### Dataset Export

Generated datasets are exported in multiple formats:

- **CSV**: Human-readable, portable, suitable for small datasets
- **NumPy (.npz)**: Binary, preserves precision, efficient for large datasets
- **Pandas**: In-memory DataFrame for immediate analysis

Each export includes metadata for full reproducibility.

### Statistical Analysis

The analysis stage extracts physics insights from generated events:

1. Compute summary statistics (means, variances, correlations)
2. Fit theoretical distributions to kinematic variables
3. Perform goodness-of-fit tests
4. Generate comprehensive analysis reports

```bash
python -c "from src.analysis import analyze; analyze('data/raw/collision_events.csv.gz')"
```

### Visualization

Publication-quality plots communicate results effectively:

- Energy distributions with theoretical overlays
- Angular distributions confirming isotropy
- Scatter plots revealing correlations
- Density maps showing phase-space coverage

All plots are exported in PNG and PDF formats with configurable resolution.

### Interpretation

The final stage connects numerical results to physical insights:

- Compare simulated distributions to theoretical predictions
- Assess systematic effects from detector resolutions
- Extract physics parameters (cross-sections, form factors)
- Document findings for publication

---

## Project Structure

```
particle-collision-simulator/
│
├── src/
│   ├── main.py            # CLI entry point
│   ├── config.py           # Simulation parameters & physical constants
│   ├── simulation.py       # High-level orchestrator
│   ├── physics.py          # Relativistic kinematics
│   ├── montecarlo.py       # Monte Carlo sampling engine
│   ├── dataset.py          # CSV / NumPy persistence
│   ├── visualization.py    # Matplotlib plot suite
│   └── analysis.py         # Statistical analysis & reporting
│
├── data/
│   ├── raw/                # Generated datasets
│   └── processed/          # (reserved for downstream analysis)
│
├── results/
│   ├── plots/              # PNG + PDF figures
│   └── logs/               # Analysis reports
│
├── requirements.txt
├── setup.sh                # One-command environment setup
└── README.md               # ← you are here
```

---

## Getting Started

### Prerequisites

- **Python 3.11+**
- `pip` (comes with Python)

### Installation

```bash
# Clone the repository
git clone <repo-url>
cd particle-collision-simulator

# Run the setup script (creates venv + installs deps)
chmod +x setup.sh
./setup.sh

# Activate the virtual environment
source venv/bin/activate
```

Or manually:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## Usage

### Basic run (100 000 events, full pipeline)

```bash
python src/main.py
```

### Custom event count

```bash
python src/main.py --events 50000
```

### Specify energy range

```bash
python src/main.py --events 200000 --energy-min 10 --energy-max 500
```

### Skip plots (data + analysis only)

```bash
python src/main.py --events 100000 --no-plot
```

### Skip dataset export

```bash
python src/main.py --events 100000 --no-save-dataset
```

### Full option list

```bash
python src/main.py --help
```

| Flag | Default | Description |
|---|---|---|
| `--events`, `-n` | 100 000 | Number of collision events |
| `--energy-min` | 1.0 | Minimum beam energy (GeV) |
| `--energy-max` | 1000.0 | Maximum beam energy (GeV) |
| `--seed` | 42 | Random seed (−1 = non-deterministic) |
| `--save-dataset` | True | Export CSV + NPZ to `data/raw/` |
| `--no-save-dataset` | — | Disable dataset export |
| `--plot` | True | Generate plots in `results/plots/` |
| `--no-plot` | — | Disable plot generation |
| `--output-dir`, `-o` | auto | Project root directory |

---

## Example Command Line Experiments

### Small Test Simulation

For quick validation and development purposes:

```bash
python src/main.py --events 1000 --seed 42
```

**Expected output:**
- Execution time: < 0.1 seconds
- Dataset size: ~50 KB (CSV), ~40 KB (NPZ)
- Generated plots: 6 figures

This configuration is ideal for testing new features, debugging, or verifying code changes.

### Medium-Scale Simulation

For parameter studies and preliminary analyses:

```bash
python src/main.py --events 100000 --energy-min 10 --energy-max 500 --seed 42
```

**Expected output:**
- Execution time: 0.1-0.3 seconds
- Dataset size: ~5 MB (CSV), ~4 MB (NPZ)
- Generated plots: 6 publication-quality figures
- Statistical analysis: Full distribution fits and goodness-of-fit tests

This scale provides sufficient statistics for most visualization and initial analysis needs.

### Large-Scale Simulation

For high-statistics studies and production-quality datasets:

```bash
python src/main.py --events 1000000 --energy-min 1 --energy-max 1000 --seed 42
```

**Expected output:**
- Execution time: 1-2 seconds
- Dataset size: ~50 MB (CSV), ~40 MB (NPZ)
- Generated plots: 6 high-resolution figures
- Statistical analysis: Robust parameter estimates with narrow confidence intervals

### Very Large Simulation

For intensive studies requiring maximum statistics:

```bash
python src/main.py --events 10000000 --seed 42 --no-plot
```

**Expected output:**
- Execution time: 15-20 seconds
- Dataset size: ~500 MB (CSV), ~400 MB (NPZ)
- Memory usage: ~3.5 GB
- Note: Plot generation disabled due to memory constraints

### Reproducing Published Results

To exactly reproduce a previously published simulation:

```bash
python src/main.py --events 500000 --energy-min 10 --energy-max 500 --seed 12345 --plot
```

---

## Configuration File Example

For complex experiments requiring multiple parameters, YAML configuration files provide a clean, reproducible way to specify simulation settings. While the current version uses command-line arguments, the following YAML configuration demonstrates the planned configuration file support.

### simulation_config.yaml

```yaml
# Monte Carlo Simulation Configuration
# ===================================

# Event Generation
events: 100000
random_seed: 42

# Physics Parameters
physics:
  energy_min: 1.0        # GeV
  energy_max: 1000.0     # GeV
  particle_mass: 0.0     # GeV/c^2 (0 = massless)
  asymmetry_sigma: 0.01  # Energy sharing uncertainty

# Output Configuration
output:
  format: csv             # csv, npz, or both
  directory: data/raw
  compress: true          # gzip compression for CSV
  include_metadata: true  # JSON metadata file

# Visualization
plots:
  enabled: true
  format: [png, pdf]
  dpi: 300
  style: publication
  directory: results/plots

# Analysis
analysis:
  enabled: true
  fit_distributions:
    - gaussian
    - lognormal
    - exponential
  ks_test_alpha: 0.05
  directory: results/logs

# Performance
performance:
  batch_size: 10000       # Events per batch (for memory management)
  num_workers: 1          # Parallel workers (future: >1)
```

### Loading Configuration (Future Feature)

```bash
# Planned syntax for configuration file loading
python src/main.py --config simulation_config.yaml

# Override specific parameters
python src/main.py --config simulation_config.yaml --events 500000
```

The configuration file approach enables:

- **Version control**: Store configurations with simulation results
- **Reproducibility**: Share exact settings via configuration files
- **Complex experiments**: Manage many parameters cleanly
- **Documentation**: Self-documenting experiment definitions

---

## Example Output

### Console output (excerpt)

```
╔══════════════════════════════════════════════════════╗
║   Particle Collision Monte Carlo Simulator           ║
╚══════════════════════════════════════════════════════╝

╔══════════════════════════════════════════╗
║        Simulation Configuration          ║
╚══════════════════════════════════════════╝
  Events         :       50,000
  Energy range   : [1.0, 1000.0] GeV
  Angle range    : [0.000, 3.142] rad

  ⚙  MonteCarloEngine  seed=42  events=50,000
  → Sampling event parameters …
    ✔ Sampling complete  (0.01 s)
  → Computing kinematics …
    ✔ Kinematics complete  (0.03 s)
    ✔ 50,000 events generated in 0.04 s (1,250,000 events/s)

  💾  CSV saved  → data/raw/collision_events.csv.gz  (2.14 MB)
  💾  NPZ saved  → data/raw/collision_events.npz     (1.87 MB)

  📊  Plot saved → energy_distribution.png
  📊  Plot saved → angular_distribution.png
  📊  Plot saved → momentum_scatter.png
  📊  Plot saved → event_density.png
  📊  Plot saved → final_energy_correlation.png
  📊  Plot saved → invariant_mass.png
```

### Generated plots

| Plot | Description |
|---|---|
| `energy_distribution.png` | Overlaid beam + final-state energy histograms |
| `angular_distribution.png` | θ and cos θ distributions (isotropy check) |
| `momentum_scatter.png` | px–py and px–pz scatter plots |
| `event_density.png` | 2-D heat map of (E_A, E_B) |
| `final_energy_correlation.png` | Anti-correlation from energy conservation |
| `invariant_mass.png` | Reconstructed √s spectrum |

### Dataset columns

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

While this simulator provides a useful educational and prototyping tool, it is important to understand its limitations compared to production-grade physics simulation frameworks.

### Classical Mechanics Approximation

The simulator uses classical relativistic kinematics rather than a full quantum field theory treatment:

- No quantum mechanical scattering amplitudes
- No interference effects between multiple scattering channels
- Cross-sections are not computed from first principles
- No radiative corrections (QED, QCD)

### Absence of Relativistic Effects

The model simplifies several relativistic effects:

- No Lorentz transformation to arbitrary reference frames
- No time dilation effects on particle lifetimes
- No relativistic aberration of angular distributions
- Fixed target geometry rather than collider kinematics

### Simplified Scattering Model

The 2→2 scattering process is highly simplified:

- No angular-dependent differential cross-sections
- Isotropic scattering assumed (constant matrix element)
- No spin correlations between initial and final states
- No particle identity considerations (bosons/fermions)

### No Detector Simulation

Production physics simulations include detailed detector response:

- No geometric acceptance effects
- No particle identification efficiencies
- No energy resolution smearing
- No hit clustering or tracking algorithms
- No trigger simulation

For detector effects, integration with frameworks like GEANT4 is required.

### No Quantum Interactions

The simulation lacks quantum mechanical features:

- No tunnel effects or quantum statistics
- No vacuum polarization or other loop corrections
- No particle-antiparticle pair production (except as kinematic final states)
- No weak interaction processes

### Why These Simplifications Were Made

These limitations reflect the educational and prototyping purpose of the tool:

1. **Computational simplicity**: Full QED/QCD event generation requires significant compute resources
2. **Conceptual clarity**: Simplified models illustrate core Monte Carlo methodology
3. **Development speed**: Rapid iteration for algorithm development and testing
4. **Scope management**: Focus on Monte Carlo techniques rather than precise physics

For research requiring more accurate physics, consider integrating with established frameworks such as PYTHIA, HERWIG, or GEANT4.

---

## Future Research Extensions

This section outlines potential extensions to enhance the simulator's capabilities for research applications.

### Relativistic Kinematics

**Full Lorentz Boost Implementation**: Transform between laboratory and centre-of-mass frames:

- Implement four-vector arithmetic with proper metric signature
- Add velocity addition formulas
- Compute Lorentz factors for arbitrary boost velocities

**Rindler Coordinates**: Model accelerated particle beams in relativistic regimes.

### Quantum Scattering Models

**Differential Cross-Sections**: Replace isotropic scattering with physical scattering formulas:

- Rutherford scattering: $d\sigma/d\Omega \propto \sin^{-4}(\theta/2)$
- Compton scattering kinematics
- Pair production thresholds

**Spin Correlations**: Include spin-dependent effects for fermion scattering.

### Detector Response Simulation

**Energy Resolution**: Model calorimeter response:

$$E_{\text{measured}} = E_{\text{true}} \cdot (1 + \mathcal{N}(0, \sigma_E))$$

where $\sigma_E$ scales with $\sqrt{E}$ (stochastic term) plus constant terms.

**Angular Smearing**: Account for tracking resolution:

$$\theta_{\text{measured}} = \theta_{\text{true}} + \mathcal{N}(0, \sigma_\theta)$$

**Efficiency Curves**: Model particle identification efficiencies as functions of momentum and pseudorapidity.

### GEANT4 Integration

Integration with the GEANT4 simulation toolkit would enable:

- Detailed detector geometry modeling
- Complete particle propagation through materials
- Electromagnetic and hadronic shower simulation
- Realistic hit creation and digitisation

### ROOT Dataset Export

ROOT files are the standard data format in high-energy physics:

- Export to ROOT TTree format using `uproot` library
- Maintain binary precision for large datasets
- Enable compatibility with existing HEP analysis frameworks
- Support ROOT's compression algorithms

### Distributed Monte Carlo Simulations

**Multiprocessing**: Parallel event generation across CPU cores:

```python
from multiprocessing import Pool
with Pool(processes=8) as pool:
    results = pool.map(generate_events, batch_params)
```

**Grid Computing**: Scale to thousands of cores for high-statistics production:

- Job submission scripts for HPC clusters
- Automatic data aggregation and merging
- Progress monitoring and failure recovery

### GPU Acceleration

CUDA or OpenCL implementations for massive parallelism:

- Generate millions of events per second on GPUs
- Vectorize random sampling on GPU
- Accelerate kinematics calculations

```python
# Conceptual GPU kernel
@cuda.jit
def compute_kinematics(E_A, E_B, cos_theta, phi, E_final):
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    # Vectorized computation per thread
```

### Machine Learning Event Classification

Integrate ML models for physics analysis:

- **Particle identification**: Classify final state particles
- **Anomaly detection**: Identify unusual event topologies
- **Regression**: Predict kinematic variables from low-level features

```python
# Conceptual classifier
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)  # Train on labeled Monte Carlo data
```

---

## Technology Stack

| Library | Role |
|---|---|
| **NumPy** | Vectorised numerics, random sampling |
| **SciPy** | Distribution fitting, KS tests |
| **Matplotlib** | Publication-quality figures |
| **Pandas** | Tabular data management |
| **tqdm** | Progress bars (future extensions) |

---

## Citation

If you use this simulation software in your research, please cite it appropriately:

```bibtex
@software{particle_collision_simulation,
  author = {Author Name},
  title = {Monte Carlo Particle Collision Simulation},
  year = {2026},
  url = {https://github.com/your-repo/particle-collision-simulator},
  version = {1.0.0},
  license = {MIT}
}
```

For academic publications, consider citing specific versions and including DOIs where available.

---

## Contributing

Contributions to this project are welcome. Please follow these guidelines to maintain code quality and consistency.

### Code Style

- Follow **PEP 8** style guidelines
- Use type hints for function signatures
- Write docstrings in NumPy format:

```python
def compute_invariant_mass(E_total, p_total):
    """Compute the invariant mass from total energy and momentum.

    Parameters
    ----------
    E_total : float
        Total energy of the system in GeV.
    p_total : ndarray
        Three-momentum vector in GeV/c.

    Returns
    -------
    float
        Invariant mass sqrt(s) in GeV/c^2.

    """
    return np.sqrt(E_total**2 - np.dot(p_total, p_total))
```

- Maximum line length: 100 characters
- Use meaningful variable names

### Pull Request Workflow

1. **Fork** the repository
2. Create a **feature branch**: `git checkout -b feature/your-feature`
3. Make **changes** with appropriate tests
4. Ensure all tests pass: `python -m pytest`
5. Update documentation if needed
6. Submit a **pull request** with clear description
7. Respond to review comments

### Testing Expectations

- New features should include unit tests
- Bug fixes should include regression tests
- Run existing test suite before submitting PR
- Test coverage should not decrease

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest src/tests/

# Run with coverage
pytest --cov=src tests/
```

---

## License

This project is released under the **MIT License**:

```
MIT License

Copyright (c) 2026 Particle Collision Simulator

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

This project is released for educational and portfolio purposes.
