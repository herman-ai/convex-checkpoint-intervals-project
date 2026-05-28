import numpy as np
import cvxpy as cp


def solve_optimal_intervals_with_caps(
    total_useful_work: float,
    lambdas: np.ndarray,
    recoveries: np.ndarray,
    capacities: np.ndarray,
    min_interval: float = 0.0,
):
    """
    Solve

        minimize    sum_k (1/lambda_k + q_k) * (exp(lambda_k * Delta_k) - 1)
        subject to  sum_k Delta_k = total_useful_work
                    min_interval <= Delta_k <= capacities_k

    Parameters
    ----------
    total_useful_work : float
        Total useful work T.
    lambdas : np.ndarray
        lambda_k values, shape (K,)
    recoveries : np.ndarray
        q_k values, shape (K,)
    capacities : np.ndarray
        Upper bounds U_k on useful work allocated to stage k, shape (K,)
    min_interval : float
        Lower bound on each interval

    Returns
    -------
    dict
        status, objective_value, intervals
    """
    lambdas = np.asarray(lambdas, dtype=float)
    recoveries = np.asarray(recoveries, dtype=float)
    capacities = np.asarray(capacities, dtype=float)

    if lambdas.ndim != 1 or recoveries.ndim != 1 or capacities.ndim != 1:
        raise ValueError("lambdas, recoveries, capacities must be 1D arrays")
    if not (len(lambdas) == len(recoveries) == len(capacities)):
        raise ValueError("lambdas, recoveries, capacities must have same length")
    if np.any(lambdas <= 0):
        raise ValueError("all lambdas must be positive")
    if np.any(recoveries < 0):
        raise ValueError("all recoveries must be nonnegative")
    if np.any(capacities <= 0):
        raise ValueError("all capacities must be positive")
    if total_useful_work <= 0:
        raise ValueError("total_useful_work must be positive")
    if total_useful_work > np.sum(capacities):
        raise ValueError("total_useful_work exceeds total available capacity")

    K = len(lambdas)
    Delta = cp.Variable(K)

    coeffs = 1.0 / lambdas + recoveries
    objective = cp.sum(
        cp.multiply(coeffs, cp.exp(cp.multiply(lambdas, Delta)) - 1.0)
    )

    constraints = [
        cp.sum(Delta) == total_useful_work,
        Delta >= min_interval,
        Delta <= capacities,
    ]

    problem = cp.Problem(cp.Minimize(objective), constraints)
    problem.solve(solver=cp.SCS, verbose=False)

    return {
        "status": problem.status,
        "objective_value": problem.value,
        "intervals": None if Delta.value is None else np.array(Delta.value).ravel(),
    }


T = 24 * 3600  # 24 hours of useful work

# Suppose these are daily windows with different risk/recovery:
# 0: midnight-9am   (safe)
# 1: 9am-6pm        (risky)
# 2: 6pm-midnight   (medium)
# Repeat over 2 days -> 6 stages total

lambdas = np.array([
    1/(40*3600),  # safe
    1/(14*3600),  # risky
    1/(24*3600),  # medium
    1/(40*3600),  # safe
    1/(14*3600),  # risky
    1/(24*3600),  # medium
], dtype=float)

recoveries = np.array([
    5*60,
    30*60,
    10*60,
    5*60,
    30*60,
    10*60,
], dtype=float)

# Capacities correspond to the wall-clock length of each window
capacities = np.array([
    9*3600,   # midnight-9am
    9*3600,   # 9am-6pm
    6*3600,   # 6pm-midnight
    9*3600,
    9*3600,
    6*3600,
], dtype=float)

result = solve_optimal_intervals_with_caps(
    total_useful_work=T,
    lambdas=lambdas,
    recoveries=recoveries,
    capacities=capacities,
    min_interval=0.0,
)

print("status:", result["status"])
print("objective:", result["objective_value"])
print("intervals (hours):", result["intervals"] / 3600.0)
print("sum (hours):", np.sum(result["intervals"]) / 3600.0)

import matplotlib.pyplot as plt

intervals = result["intervals"] / 3600.0
caps = capacities / 3600.0
idx = np.arange(len(intervals))

plt.figure(figsize=(9, 4))
plt.bar(idx, intervals, label="Allocated useful work")
plt.plot(idx, caps, "ro--", label="Capacity")
plt.xlabel("Stage index")
plt.ylabel("Hours")
plt.title("Optimal allocation with per-stage capacities")
plt.legend()
plt.tight_layout()
plt.show()