"""
Monte Carlo evaluation under theta uncertainty for step-hazard problems.
"""

from __future__ import annotations

import numpy as np

from simulator_v2 import simulate_schedule_useful_work_hazard, monte_carlo_schedule_useful_work_hazard
from problems import make_step_lambda_fn

__all__ = [
    "monte_carlo_uncertain_theta",
    "monte_carlo_schedule_useful_work_hazard",
]


def monte_carlo_uncertain_theta(
    delta: np.ndarray,
    tau: np.ndarray,
    theta_hat: np.ndarray,
    rho: np.ndarray,
    q: np.ndarray,
    num_trials: int = 1000,
    base_seed: int = 0,
) -> dict:
    """
    Monte Carlo evaluation where theta is drawn uniformly from the box
    [theta_hat - rho, theta_hat + rho] (clipped to non-negative) for each trial.

    Returns mean runtime, its standard error, and mean number of failures.
    """
    rng      = np.random.default_rng(base_seed)
    theta_lo = np.maximum(theta_hat - rho, 0.0)
    theta_hi = theta_hat + rho

    runtimes = []
    failures = []
    for i in range(num_trials):
        theta_sample = rng.uniform(theta_lo, theta_hi)
        lambda_fn    = make_step_lambda_fn(tau, theta_sample)
        result       = simulate_schedule_useful_work_hazard(
            delta=delta, lambda_fn=lambda_fn, q=q, seed=base_seed + i
        )
        runtimes.append(result["total_wall_clock"])
        failures.append(result["total_failures"])

    return {
        "mean_runtime":   float(np.mean(runtimes)),
        "stderr_runtime": float(np.std(runtimes, ddof=1) / np.sqrt(num_trials)),
        "mean_failures":  float(np.mean(failures)),
    }
