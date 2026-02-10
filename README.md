# Neural Network Solutions for the New Keynesian DSGE Model

This repository implements neural network-based solution methods for the three-equation New Keynesian Dynamic Stochastic General Equilibrium (DSGE) model. It replicates **Section 3.1 (Proof of Concept 1)** from:

> Kase, H., Melosi, L., & Rottner, M. (2025). *Estimating Nonlinear Heterogeneous Agent Models with Neural Networks.*

## Motivation

Traditional perturbation methods (e.g., linearization) break down when macroeconomic models feature occasionally binding constraints (e.g., the zero lower bound), strong nonlinearities, or heterogeneous agents. Neural networks provide a **global approximation** of policy functions that handles these challenges effectively.

This notebook focuses on the **solution step** using neural networks. The particle filter and likelihood estimation pipeline are covered in subsequent parts of the project.

## Model

The canonical three-equation New Keynesian model:

| Equation | Description |
|----------|-------------|
| **IS Curve** | $\hat{X}_t = \mathbb{E}_t[\hat{X}_{t+1}] - \sigma^{-1}(\phi_\pi \hat{\pi}_t + \phi_y \hat{X}_t - \mathbb{E}_t[\hat{\pi}_{t+1}] - \zeta_t)$ |
| **Phillips Curve** | $\hat{\pi}_t = \beta \mathbb{E}_t[\hat{\pi}_{t+1}] + \kappa \hat{X}_t$ |
| **Shock Process** | $\zeta_{t+1} = \rho \zeta_t + \varepsilon_{t+1} \cdot \sigma_\varepsilon \cdot \sigma \cdot (\rho - 1) \cdot \omega$ |

The neural network learns the mapping from the state variable $\zeta_t$ and structural parameters $\theta$ to the policy functions $(\hat{X}_t, \hat{\pi}_t)$.

## Neural Network Architecture

| Component | Specification |
|-----------|--------------|
| **Input** | 9-dimensional: $[\zeta, \beta, \sigma, \eta, \phi, \phi_\pi, \phi_y, \rho, \sigma_\varepsilon]$ |
| **Normalization** | Min-max scaling to $[-1, 1]$ |
| **Hidden Layers** | 5 layers, 64 neurons each, CELU activation |
| **Output** | 2-dimensional: $[\hat{X}, \hat{\pi}]$, scaled by $1/100$ |
| **Framework** | [Lux.jl](https://github.com/LuxDL/Lux.jl) (Julia) |

### Training

- **Loss function**: Weighted sum of squared Euler equation and Phillips curve residuals
- **Optimizer**: Adam with cosine annealing learning rate schedule
- **Sampling**: Each epoch draws fresh parameters from uniform priors and states from the ergodic distribution, with antithetic variates for variance reduction
- **Pre-trained model**: 100,000 epochs (included as `model_simple_internal_100k.jld2`)

## Results

The trained neural network closely matches the analytical (method of undetermined coefficients) solution across the full parameter space:

| Variable | Error at $\zeta = -1\sigma_\text{ergodic}$ |
|----------|------|
| Output gap $\hat{X}$ | 1.15% |
| Inflation $\hat{\pi}$ | 0.24% |

### Policy Function Comparisons

**Output Gap:**

![Output Gap Comparison](Policy%20Function%20Comparison-Output%20Gap.pdf)

**Inflation:**

![Inflation Comparison](Policy%20Function%20Comparison-Inflation.pdf)

## Repository Structure

```
.
├── README.md
├── NK Three Equation Model and Neural Network.ipynb   # Main notebook
├── model_simple_internal_100k.jld2                     # Pre-trained model weights
├── Policy Function Comparison-Output Gap.pdf           # Output gap figure
└── Policy Function Comparison-Inflation.pdf            # Inflation figure
```

## Getting Started

### Prerequisites

- [Julia](https://julialang.org/downloads/) (v1.9+)
- [IJulia](https://github.com/JuliaLang/IJulia.jl) for Jupyter notebook support

### Required Julia Packages

```julia
using Pkg
Pkg.add([
    "Lux", "Optimisers", "Zygote",    # Neural networks & AD
    "Distributions", "Sobol",          # Sampling
    "LinearAlgebra", "Statistics",     # Core math
    "Random", "ProgressMeter",        # Utilities
    "Printf", "JLD2",                 # I/O
    "Plots"                           # Visualization
])
```

### Running the Notebook

1. Clone this repository:
   ```bash
   git clone https://github.com/iamsurajkumar/Neural-Network-NK-Model.git
   cd Neural-Network-NK-Model
   ```

2. Launch the notebook:
   ```bash
   jupyter notebook "NK Three Equation Model and Neural Network.ipynb"
   ```

3. The notebook is configured to **load the pre-trained model** by default (`TRAIN_MODEL = false`). To train from scratch, set `TRAIN_MODEL = true` in the training cell.

## Calibration

Baseline parameters (calibrated to quarterly US data):

| Parameter | Value | Description |
|-----------|-------|-------------|
| $\beta$ | 0.97 | Discount factor (~3% annual real rate) |
| $\sigma$ | 2.0 | Relative risk aversion |
| $\eta$ | 1.125 | Inverse Frisch elasticity |
| $\phi$ | 0.7 | Calvo parameter (~3 quarter price duration) |
| $\phi_\pi$ | 1.875 | Taylor rule inflation response |
| $\phi_y$ | 0.25 | Taylor rule output response |
| $\rho$ | 0.875 | Shock persistence |
| $\sigma_\varepsilon$ | 0.06 | Shock standard deviation |

## Reference

```bibtex
@article{kase2025estimating,
  title={Estimating Nonlinear Heterogeneous Agent Models with Neural Networks},
  author={Kase, Hanno and Melosi, Leonardo and Rottner, Matthias},
  year={2025}
}
```

## License

This project is for academic and research purposes.
