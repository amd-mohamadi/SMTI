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

Ensure you have [Pixi](https://pixi.sh) installed. SMTI uses Pixi to manage its environment and dependencies (including C++ compilers and BLAS/LAPACK for PyTensor).

### Setup with Pixi (Recommended)
```bash
pixi init
pixi add python=3.11 numpy=1.26.4 pymc=5.24.1 arviz=0.21.0 matplotlib=3.10.5 scipy=1.15.2 pandas=2.2.3 scikit-learn
pixi add --pypi pyrocko==2025.1.21
```

### Setup with Conda
If you prefer Conda, you can create an environment manually:
```bash
conda create -n smti python=3.11
conda activate smti
pip install pymc==5.24.1 arviz==0.21.0 matplotlib==3.10.5 pyrocko==2025.1.21 numpy==1.26.4 scipy==1.15.2 pandas==2.2.3
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

<!-- ## 📊 Outputs & Visualization

SMTI automatically generates a variety of diagnostic and scientific plots. Here are examples from a synthetic FORGE inversion:

<p align="center">
  <img src="synthetic_test/forge_1111911135_PPolarity_SHPolarity_PSHAmplitudeRatio_PSVAmplitudeRatio_mt/fault_planes_hdi90.png" width="30%" />
  <img src="synthetic_test/forge_1111911135_PPolarity_SHPolarity_PSHAmplitudeRatio_PSVAmplitudeRatio_mt/kaverina_hdi_90.png" width="30%" />
  <img src="synthetic_test/forge_1111911135_PPolarity_SHPolarity_PSHAmplitudeRatio_PSVAmplitudeRatio_mt/hudson_hdi_90.png" width="30%" />
</p>

*Left: Fault plane distribution (90% HDI). Center: Kaverina diagram. Right: Hudson plot.*

- **Hudson & Kaverina Diagrams**: With 90% HDI (Highest Density Interval) contours.
- **Beachballs**: Median and MAP (mode) solutions with station overlays.
- **Posterior Summaries**: Pairwise distribution of Tape parameters using ArviZ. -->

## ⚠️ Status
This is an experimental research code. API stability is not guaranteed. 

---
Developed by **Ahmad Mohamadi**.
