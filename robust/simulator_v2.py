from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from utils.helpers import integrate_lambda


def invert_hazard_on_interval(
    lambda_fn: Callable[[float], float],
    a: float,
    target_hazard: float,
    b: float,
    num_steps: int = 128,
    tol: float = 1e-6,
    max_bisect_iters: int = 60,
) -> float:
    """
    Solve for x in [a, b] such that ∫_a^x lambda(u) du = target_hazard.
    Uses bisection with numerical integration.
    """
    left, right = a, b
    for _ in range(max_bisect_iters):
        mid = 0.5 * (left + right)
        val = integrate_lambda(lambda_fn, a, mid, num_steps=num_steps)
        if abs(val - target_hazard) < tol:
            return mid
        if val < target_hazard:
            left = mid
        else:
            right = mid
    return 0.5 * (left + right)


def simulate_one_interval_useful_work_hazard(
    start_work: float,
    interval_work: float,
    lambda_fn: Callable[[float], float],
    recovery_overhead: float,
    checkpoint_cost: float,
    rng: random.Random,
    num_steps: int = 256,
) -> Dict[str, float]:
    """
    Simulate one interval under useful-work-dependent hazard.
    """
    a = start_work
    b = start_work + interval_work
    L = integrate_lambda(lambda_fn, a, b, num_steps=num_steps)

    wall_clock = 0.0
    failures = 0

    while True:
        E = rng.expovariate(1.0)
        if E >= L:
            wall_clock += interval_work + checkpoint_cost
            return {"wall_clock": wall_clock, "failures": failures}

        x_fail = invert_hazard_on_interval(
            lambda_fn=lambda_fn, a=a, target_hazard=E, b=b, num_steps=num_steps,
        )
        elapsed_runtime = x_fail - a
        wall_clock += elapsed_runtime + recovery_overhead
        failures += 1


def simulate_schedule_useful_work_hazard(
    delta: np.ndarray,
    lambda_fn: Callable[[float], float],
    q: np.ndarray,
    checkpoint_costs: Optional[np.ndarray] = None,
    seed: int = 0,
    num_steps: int = 256,
) -> Dict[str, float]:
    """
    Simulate a full checkpoint schedule under useful-work-dependent hazard.
    """
    rng = random.Random(seed)
    delta = np.asarray(delta, dtype=float)
    K = len(delta)

    if checkpoint_costs is None:
        checkpoint_costs = np.zeros(K, dtype=float)
    else:
        checkpoint_costs = np.asarray(checkpoint_costs, dtype=float)

    T_knots = np.zeros(K + 1, dtype=float)
    T_knots[1:] = np.cumsum(delta)

    total_wall_clock = 0.0
    total_failures = 0

    for k in range(K):
        result = simulate_one_interval_useful_work_hazard(
            start_work=T_knots[k],
            interval_work=delta[k],
            lambda_fn=lambda_fn,
            recovery_overhead=q[k],
            checkpoint_cost=checkpoint_costs[k],
            rng=rng,
            num_steps=num_steps,
        )
        total_wall_clock += result["wall_clock"]
        total_failures += result["failures"]

    return {"total_wall_clock": total_wall_clock, "total_failures": total_failures}


def monte_carlo_schedule_useful_work_hazard(
    delta: np.ndarray,
    lambda_fn: Callable[[float], float],
    q: np.ndarray,
    checkpoint_costs: Optional[np.ndarray] = None,
    num_trials: int = 200,
    base_seed: int = 0,
    num_steps: int = 256,
) -> Dict[str, float]:
    runtimes = []
    failures = []

    for i in range(num_trials):
        result = simulate_schedule_useful_work_hazard(
            delta=delta, lambda_fn=lambda_fn, q=q,
            checkpoint_costs=checkpoint_costs,
            seed=base_seed + i, num_steps=num_steps,
        )
        runtimes.append(result["total_wall_clock"])
        failures.append(result["total_failures"])

    mean_runtime = float(np.mean(runtimes))
    stderr_runtime = (
        float(np.std(runtimes, ddof=1) / np.sqrt(num_trials)) if num_trials > 1 else 0.0
    )
    return {
        "mean_runtime":   mean_runtime,
        "stderr_runtime": stderr_runtime,
        "mean_failures":  float(np.mean(failures)),
    }
