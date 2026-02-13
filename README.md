# SMTI: Sequential Moment Tensor Inversion

**SMTI** is a Bayesian Moment Tensor (MT) inversion framework with two powerful backends: **PyMC/PyTensor** for CPU-based sampling and **BlackJAX** for GPU-accelerated inference. It leverages Sequential Monte Carlo (SMC) samplers to provide robust global searching of the MT parameter space, handling multi-modal posteriors and providing rigorous uncertainty quantification.

## 🚀 Overview

Traditional MT inversion methods often struggle with non-linear parameter spaces and complex likelihood surfaces. SMTI addresses these challenges by:
- **Dual Backends**: Choose between PyMC (CPU, multi-core) or BlackJAX (GPU, JIT-compiled) depending on your hardware.
- **Bayesian Sampling**: Using adaptive tempered SMC to explore the full Tape parameter space.
- **GPU Acceleration**: BlackJAX backend with pure JAX implementation enables 10-100x speedups on NVIDIA GPUs.
- **High Performance**: Optimized computation graphs that escape the Python GIL for true parallelization.
- **Comprehensive Data Support**: Jointly inverting P/S polarities, absolute amplitudes, and amplitude ratios.
- **Rich Visualization**: Automated generation of Hudson diagrams, Kaverina plots with HDI contours, and beachball diagrams.

## ✨ Key Features

- **Dual Inversion Engines**:
  - `InversionPyTensor`: Pure PyTensor graph-based implementation for CPU multi-core parallelization.
  - `InversionBlackJAX`: Pure JAX/BlackJAX implementation for GPU acceleration with JIT compilation.
- **Quality Assessment**: Native support for **MT Quality Score (Q)**, combining convergence diagnostics (R-hat, ESS) with posterior concentration.
- **Uncertainty Handling**: Integrated support for station-angle location uncertainty and measurement noise.
- **Source Types**: Toggle between pure **Double-Couple (DC)** and **Full Moment Tensor** inversions.

## 📦 Installation

Clone the repository:
```bash
git clone https://github.com/amd-mohamadi/SMTI.git
cd SMTI
```

### Option 1: Pixi (Recommended)
[Pixi](https://pixi.sh) provides fast, reproducible environments. Once installed, simply run:
```bash
pixi install
pixi run python synthetic_test.py
```

### Option 2: Conda
```bash
conda env create -f environment.yml
conda activate smti
```

## 🛠️ Quick Start

### Run the Synthetic Test
The project includes a comprehensive synthetic test script that generates data for an event, performs the inversion, and saves result plots.

```bash
python synthetic_test.py
```

### Basic API Usage

#### Option 1: PyTensor (CPU, Multi-core)
```python
from src.inversion_pytensor import InversionPyTensor
from example_data import synthetic_event

# Load data
data = synthetic_event()

# Initialize inversion
inv = InversionPyTensor(
    data,
    inversion_options=['PPolarity', 'P/SHAmplitudeRatio'],
    draws=2000,
    chains=4,
    dc=False
)

# Run sampling
result = inv.forward()

# Access results
print(f"Posterior MT6 Mean:\n{result.mt6.mean(axis=1)}")
```

#### Option 2: BlackJAX (GPU-Accelerated)
```python
from src.inversion_blackjax import InversionBlackJAX
from src.data_loader import read_data

# Load data
data = read_data('path/to/datac.dat')

# Initialize inversion
inv = InversionBlackJAX(
    data,
    inversion_options=['PPolarity', 'PSHAmplitudeRatio'],
    num_particles=2000,
    dc=False,
    random_seed=42,
    mcmc_kernel='rmh',  # or 'nuts' for NUTS rejuvenation
    rmh_proposal_scale=0.02
)

# Run sampling (automatically uses GPU if available)
result = inv.forward()

# Access results
print(f"Posterior MT6 Median:\n{np.median(result.mt6, axis=1)}")
```

### GPU Setup for BlackJAX
To enable GPU acceleration, ensure you have JAX installed with CUDA support:
```bash
pip install --upgrade "jax[cuda12]"  # For CUDA 12.x
pip install blackjax
```

## 📂 Project Structure

- `src/`: The core package.
  - **PyTensor Backend (CPU)**:
    - `inversion_pytensor.py`: The main `InversionPyTensor` class for CPU-based SMC.
    - `tape_pytensor.py`: MT parameterization (Tape space) in PyTensor.
    - `forward_model_pytensor.py`: PyTensor-compiled forward modeling of radiation patterns.
  - **BlackJAX Backend (GPU)**:
    - `inversion_blackjax.py`: The `InversionBlackJAX` class for GPU-accelerated SMC.
    - `tape_jax.py`: MT parameterization (Tape space) in JAX.
  - **Shared Modules**:
    - `data_loader.py`: Data loading utilities for seismic event files.
    - `data_prep.py`: Station geometry, polarity matrices, and amplitude ratio processing.
    - `likelihoods.py`: Statistical distributions for polarities and ratios.
    - `utilities.py`: MAP estimation, quality scoring, and data generation.
    - `plot/`: Specialized plotting tools for MT analysis.
- **Runner Scripts**:
  - `cape_inversion.py`: PyTensor-based batch inversion for CAPE events.
  - `cape_inversion_blackjax.py`: BlackJAX-based batch inversion for CAPE events (GPU).
- `synthetic_test.py`: Top-level script for benchmarking and visualization.
- `example_data.py`: Pre-formatted datasets for testing.

## ⚠️ Status
This is an experimental research code. API stability is not guaranteed.

### Performance Notes
- **PyTensor backend**: Best for CPU-based workloads. Scales well with `chains` parameter (multi-core parallelization).
- **BlackJAX backend**: Best for GPU-accelerated workloads. Typically 10-100x faster than CPU for large particle counts (≥2000 particles). Requires CUDA-enabled GPU.

## 🙏 Acknowledgements

This project builds upon and is heavily inspired by [**MTfit**](https://github.com/djpugh/MTfit) by David J. Pugh. Key components derived from MTfit include:
- **Tape parameterization**: Moment tensor conversions using the Tape & Tape formulation.
- **Plotting utilities**: Beachball, and fault plane.
- **Forward modeling**: Radiation pattern calculations for P, SH, and SV phases.

We gratefully acknowledge the MTfit project for providing a solid foundation for seismic moment tensor analysis.

## 📖 Citation

If you use SMTI in your research, please also cite the original MTfit paper:

> Pugh, D.J. and White, R.S., 2018. MTfit: A Bayesian approach to seismic moment tensor inversion. *Seismological Research Letters*, 89(4), pp.1507-1513. [https://doi.org/10.1785/0220170258](https://doi.org/10.1785/0220170258)


---
Developed by **Ahmad Mohamadi**.
