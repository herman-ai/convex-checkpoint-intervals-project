# Optimal Checkpoint Scheduling Under Failure Risk

**EE364B Course Project** — Convex Optimization

## Overview

This project solves the problem of optimally placing checkpoints during a long-running computation to minimize expected wall-clock time in the presence of time-dependent failures. Given a hazard rate $\lambda(t)$ describing failure risk as a function of useful work completed, we find the checkpoint schedule $\{T_k\}_{k=1}^{K}$ that minimizes:

$$\min_{\Delta} \sum_{k=1}^{K} \left[ e^{L_k} \cdot I_k + q_k \cdot (e^{L_k} - 1) \right] + \sum_k c_k$$

where $L_k = \int_{T_{k-1}}^{T_k} \lambda(t)\,dt$ is the cumulative hazard over interval $k$, $I_k = \int_{T_{k-1}}^{T_k} e^{-\Lambda(u)}\,du$ is the expected useful work, $q_k$ is recovery overhead, and $c_k$ is checkpoint write cost.

Both **nominal** (known hazard) and **robust** (uncertain hazard parameters) formulations are implemented.

---

## Problem Structure

### Hazard Function Families

| Family         | Description                                                      |
| -------------- | ---------------------------------------------------------------- |
| **Step**       | Piecewise-constant hazard — low risk early, high risk late       |
| **Polynomial** | Smooth quadratic increase, models gradual wear                   |
| **Power-law**  | Strongly accelerating — $\lambda(t) = \theta_0 (t/T)^{\theta_1}$ |

### Optimization Algorithms

| Algorithm          | Parameterization                    | Step Size                                        |
| ------------------ | ----------------------------------- | ------------------------------------------------ |
| **PGD**            | Internal knots $T$                  | Armijo backtracking from $\alpha_0 = 10^3$       |
| **Mirror Descent** | Interval lengths $\Delta$ (simplex) | $\alpha_t = \beta/\sqrt{t+1}$, $\beta = 10^{-2}$ |
| **ADMM**           | Consensus $x$/$z$ split             | Fixed $\rho = 5 \times 10^{-4}$                  |

All algorithms use multi-start with Dirichlet-sampled initializations and return the best solution found.

---

## Repository Structure

```
code/
├── generate_figure.py          # Nominal figures: hazard + checkpoints, convergence, K-sweep
├── generate_figure_robust.py   # Robust figures: uncertainty bands, MC comparison
├── robust/
│   ├── problems.py             # RobustStepHazardProblem, RobustPolynomialHazardProblem, RobustPowerLawHazardProblem
│   ├── optimizers.py           # optimize_pgd_internal_knots, optimize_mirror_descent, optimize_admm
│   ├── projection.py           # PAV isotonic regression (feasibility projection)
│   ├── multistart.py           # Diverse initializations + auto step-size selection
│   ├── monte_carlo.py          # MC evaluation under sampled θ uncertainty
│   ├── simulator_v2.py         # Event-driven failure simulator
│   ├── run_experiments.py      # Experiment runner: nominal vs robust comparison
│   └── utils/
│       ├── helpers.py          # UsefulWorkHazardProblem, objective/gradient, quadrature
│       └── hazard_functions.py # Hazard function templates
└── figures/
    ├── final/                  # Nominal experiment figures
    └── robust/                 # Robust experiment figures
```

---

## Running

### Nominal Experiments

```bash
python generate_figure.py
```

Generates in `figures/final/`:

- `{step,polynomial,powerlaw}_hazard.png` — Hazard rate with optimal checkpoint locations
- `{step,polynomial,powerlaw}_convergence.png` — PGD / MD / ADMM convergence across all starts
- `convergence_combined.png` — Side-by-side convergence for all three hazard types
- `K_sweep.png` — Optimal objective vs. number of checkpoints $K$
- `{step,polynomial,powerlaw}_optimal_K.png` — Hazard + checkpoints at optimal $K^*$

### Robust Experiments

```bash
python generate_figure_robust.py
```

Generates in `figures/robust/`:

- `robust_hazards.png` — Nominal, worst-case, and best-case hazard bands
- `{step,polynomial,powerlaw}_hazard_robust.png` — Robust-optimal checkpoint placement
- `{step,polynomial,powerlaw}_convergence_robust.png` — Algorithm convergence
- `convergence_combined_robust.png` — Combined convergence plot
- `mc_comparison.png` — Monte Carlo: robust vs nominal schedule under sampled uncertainty

---

## Key Parameters (set in `main()`)

| Parameter  | Value  | Description                                     |
| ---------- | ------ | ----------------------------------------------- |
| `K`        | 10     | Number of checkpoint intervals                  |
| `T`        | 48 h   | Total useful work                               |
| `epsilon`  | 0.5 h  | Minimum interval length (checkpoint write time) |
| `q`        | 600 s  | Recovery overhead per interval                  |
| `c`        | 1200 s | Checkpoint write cost (20 min)                  |
| `n_starts` | 10     | Random initializations per algorithm            |

---

## Dependencies

```
numpy
scipy
matplotlib
cvxpy
```

Install via:

```bash
pip install numpy scipy matplotlib cvxpy
```

Or with conda:

```bash
conda activate ee364b
```
