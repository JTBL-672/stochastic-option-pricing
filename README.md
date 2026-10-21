# Stochastic Option Pricing Library

A modular, high-performance Python library built for European option pricing, comparing closed-form analytical models against numerical methods (Implicit Finite Difference Method) and stochastic simulations (Monte Carlo with Variance Reduction).

---

## Key Features

* **Analytical Baseline:** Closed-form Black-Scholes solver using `scipy.stats.norm` for precise reference pricing.
* **Implicit Finite Difference Method (FDM):** Solves the Black-Scholes partial differential equation (PDE) backward in time ($T \to 0$). Utilizes `scipy.linalg.solve_banded` ($O(N)$ Thomas algorithm) for efficient tridiagonal matrix solving with Dirichlet boundary conditions at $S = 0$ and $S = S_{\max}$.
* **Monte Carlo Simulation:** Simulates exact terminal asset prices under Geometric Brownian Motion (GBM). Features **Antithetic Variates** variance reduction and automatic standard error calculation.
* **Comprehensive Benchmarking:** Centralized script (`main.py`) measuring accuracy, convergence, and execution runtimes across all pricing engines.

---

## Repository Structure

```text
stochastic-option-pricing/
├── src/
│   ├── analytical.py      # Black-Scholes closed-form model
│   ├── pde_solver.py      # Implicit FDM solver (scipy.linalg.solve_banded)
│   └── monte_carlo.py     # Stochastic simulation with Antithetic Variates
├── main.py                # Central execution and benchmarking suite
├── requirements.txt       # Project dependencies
└── README.md              # Project documentation
