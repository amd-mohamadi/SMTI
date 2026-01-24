# SMTI: Sequential Moment Tensor Inversion

**SMTI** is a Bayesian Moment Tensor (MT) inversion framework built on top of **PyMC** and **PyTensor**. It leverages the Sequential Monte Carlo (SMC) sampler to provide robust global searching of the MT parameter space, handling multi-modal posteriors and providing rigorous uncertainty quantification.

## 🚀 Overview

Traditional MT inversion methods often struggle with non-linear parameter spaces and complex likelihood surfaces. SMTI addresses these challenges by:
- **Bayesian Sampling**: Using PyMC's SMC sampler to explore the full Tape parameter space.
- **High Performance**: Optimized PyTensor graphs that escape the Python GIL for true multi-core parallelization.
- **Comprehensive Data Support**: Jointly inverting P/S polarities, absolute amplitudes, and amplitude ratios.
- **Rich Visualization**: Automated generation of Hudson diagrams, Kaverina plots with HDI contours, and beachball diagrams.

## ✨ Key Features

- **Inversion Engine**: `InversionPyTensor` provides a pure graph-based implementation using PyTensor for maximum speed, parallelization, and GIL-free execution.
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
idata, result = inv.forward()

# Access results
print(f"Posterior MT6 Mean:\n{result.mt6.mean(axis=1)}")
```

## 📂 Project Structure

- `src/`: The core package.
  - `inversion_pytensor.py`: The main `InversionPyTensor` class.
  - `forward_model_pytensor.py`: PyTensor-compiled forward modeling of radiation patterns.
  - `tape_pytensor.py`: MT parameterization (Tape space) in PyTensor.
  - `likelihoods.py`: Statistical distributions for polarities and ratios.
  - `utilities.py`: MAP estimation, quality scoring, and data generation.
  - `plot/`: Specialized plotting tools for MT analysis.
- `synthetic_test.py`: Top-level script for benchmarking and visualization.
- `example_data.py`: Pre-formatted datasets for testing.

## ⚠️ Status
This is an experimental research code. API stability is not guaranteed. 

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
