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

def solve_optimal_intervals(
    total_useful_work: float,
    lambdas: np.ndarray,
    recoveries: np.ndarray,
    min_interval: float = 1e-6,
):
    """
    Solve

        minimize    sum_k (1/lambda_k + q_k) * (exp(lambda_k * Delta_k) - 1)
        subject to  Delta_k >= min_interval
                    sum_k Delta_k = total_useful_work

    Parameters
    ----------
    total_useful_work : float
        Total useful work T.
    lambdas : np.ndarray
        Array of lambda_k values, shape (K,).
    recoveries : np.ndarray
        Array of q_k values, shape (K,).
    min_interval : float
        Lower bound on each interval.

    Returns
    -------
    dict
        Contains optimal intervals, objective value, and solver status.
    """
    lambdas = np.asarray(lambdas, dtype=float)
    recoveries = np.asarray(recoveries, dtype=float)

    if lambdas.ndim != 1 or recoveries.ndim != 1:
        raise ValueError("lambdas and recoveries must be 1D arrays")
    if len(lambdas) != len(recoveries):
        raise ValueError("lambdas and recoveries must have the same length")
    if np.any(lambdas <= 0):
        raise ValueError("all lambdas must be positive")
    if np.any(recoveries < 0):
        raise ValueError("all recoveries must be nonnegative")
    if total_useful_work <= 0:
        raise ValueError("total_useful_work must be positive")

    K = len(lambdas)
    Delta = cp.Variable(K, pos=True)

    coeffs = 1.0 / lambdas + recoveries
    objective = cp.sum(cp.multiply(coeffs, cp.exp(cp.multiply(lambdas, Delta)) - 1.0))

    constraints = [
        cp.sum(Delta) == total_useful_work,
        Delta >= min_interval,
    ]

    problem = cp.Problem(cp.Minimize(objective), constraints)
    problem.solve(solver=cp.SCS, verbose=False)

    return {
        "status": problem.status,
        "objective_value": problem.value,
        "intervals": None if Delta.value is None else np.array(Delta.value).ravel(),
    }

# Example: 8 intervals, total useful work = 24 hours
T = 24 * 3600
K = 8

# Toy stage-dependent risk / recovery profile
lambdas = np.array([
    1/(200*3600),  # very safe
    1/(100*3600),
    1/(10*3600),   # moderately risky
    1/(2*3600),    # very risky
    1/(2*3600),    # very risky
    1/(5*3600),
    1/(50*3600),
    1/(200*3600),  # very safe
], dtype=float)

recoveries = np.array([
    10*60,
    10*60,
    60*60,
    4*3600,
    4*3600,
    90*60,
    10*60,
    10*60,
], dtype=float)

result = solve_optimal_intervals(T, lambdas, recoveries)

print("status:", result["status"])
print("objective:", result["objective_value"])
print("intervals (seconds):", result["intervals"])
print("intervals (hours):", result["intervals"] / 3600.0)
print("sum:", np.sum(result["intervals"]))


def evaluate_stage_objective(intervals: np.ndarray, lambdas: np.ndarray, recoveries: np.ndarray) -> float:
    intervals = np.asarray(intervals, dtype=float)
    lambdas = np.asarray(lambdas, dtype=float)
    recoveries = np.asarray(recoveries, dtype=float)
    coeffs = 1.0 / lambdas + recoveries
    return np.sum(coeffs * (np.exp(lambdas * intervals) - 1.0))


equal_intervals = np.full(K, T / K)

print("equal objective:", evaluate_stage_objective(equal_intervals, lambdas, recoveries))
print("optimized objective:", evaluate_stage_objective(result["intervals"], lambdas, recoveries))


import matplotlib.pyplot as plt

idx = np.arange(K)

plt.figure(figsize=(8, 4))
plt.bar(idx - 0.2, equal_intervals / 3600.0, width=0.4, label="Equal spacing")
plt.bar(idx + 0.2, result["intervals"] / 3600.0, width=0.4, label="Optimized")
plt.xlabel("Interval index")
plt.ylabel("Useful work in interval (hours)")
plt.title("Equal vs optimized work intervals")
plt.legend()
plt.tight_layout()
plt.show()