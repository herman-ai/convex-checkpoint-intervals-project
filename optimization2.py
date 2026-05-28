import numpy as np
import cvxpy as cp


SECONDS_PER_HOUR = 3600.0
SECONDS_PER_DAY = 24.0 * SECONDS_PER_HOUR

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


def build_repeating_daily_stages(
    num_days: int,
    stage_hours,
    lambda_values,
    recovery_values,
):
    """
    Build repeated daily stages for the convex approximation model.

    Parameters
    ----------
    num_days : int
        Number of days to repeat the daily pattern.
    stage_hours : list[float]
        Length of each stage within one day, in hours.
        Example: [9, 9, 6]
    lambda_values : list[float]
        Hazard values for each stage in one day.
    recovery_values : list[float]
        Recovery overhead values for each stage in one day, in seconds.

    Returns
    -------
    dict with:
        - lambdas: np.ndarray
        - recoveries: np.ndarray
        - capacities: np.ndarray
        - stage_labels: list[str]
        - start_times: np.ndarray
    """
    stage_hours = np.asarray(stage_hours, dtype=float)
    lambda_values = np.asarray(lambda_values, dtype=float)
    recovery_values = np.asarray(recovery_values, dtype=float)

    if num_days <= 0:
        raise ValueError("num_days must be positive")
    if not (len(stage_hours) == len(lambda_values) == len(recovery_values)):
        raise ValueError("stage_hours, lambda_values, recovery_values must have same length")
    if np.any(stage_hours <= 0):
        raise ValueError("all stage_hours must be positive")
    if abs(np.sum(stage_hours) - 24.0) > 1e-9:
        raise ValueError("stage_hours must sum to 24 hours")

    num_stages_per_day = len(stage_hours)
    total_stages = num_days * num_stages_per_day

    lambdas = np.tile(lambda_values, num_days)
    recoveries = np.tile(recovery_values, num_days)
    capacities = np.tile(stage_hours * SECONDS_PER_HOUR, num_days)

    stage_labels = []
    start_times = []

    cumulative_day_hours = np.concatenate(([0.0], np.cumsum(stage_hours)))
    for d in range(num_days):
        day_offset = d * SECONDS_PER_DAY
        for k in range(num_stages_per_day):
            start_hour = cumulative_day_hours[k]
            end_hour = cumulative_day_hours[k + 1]
            stage_labels.append(f"day{d+1}_h{start_hour:.0f}-{end_hour:.0f}")
            start_times.append(day_offset + start_hour * SECONDS_PER_HOUR)

    return {
        "lambdas": lambdas,
        "recoveries": recoveries,
        "capacities": capacities,
        "stage_labels": stage_labels,
        "start_times": np.array(start_times, dtype=float),
    }


# Example usage
if __name__ == "__main__":
    stage_hours = [9, 9, 6]  # 12am-9am, 9am-6pm, 6pm-12am

    lambda_values = [
        1 / (40 * SECONDS_PER_HOUR),  # safer overnight
        1 / (14 * SECONDS_PER_HOUR),  # riskier peak
        1 / (24 * SECONDS_PER_HOUR),  # medium evening
    ]

    recovery_values = [
        5 * 60,   # 5 min
        30 * 60,  # 30 min
        10 * 60,  # 10 min
    ]

    data = build_repeating_daily_stages(
        num_days=3,
        stage_hours=stage_hours,
        lambda_values=lambda_values,
        recovery_values=recovery_values,
    )

    print("lambdas:", data["lambdas"])
    print("recoveries:", data["recoveries"])
    print("capacities (hours):", data["capacities"] / SECONDS_PER_HOUR)
    print("stage labels:", data["stage_labels"])
    print("start times (hours):", data["start_times"] / SECONDS_PER_HOUR)


T = 48 * SECONDS_PER_HOUR  # total useful work

data = build_repeating_daily_stages(
    num_days=3,
    stage_hours=[9, 9, 6],
    lambda_values=[
        1 / (40 * SECONDS_PER_HOUR),
        1 / (14 * SECONDS_PER_HOUR),
        1 / (24 * SECONDS_PER_HOUR),
    ],
    recovery_values=[
        5 * 60,
        30 * 60,
        10 * 60,
    ],
)

result = solve_optimal_intervals_with_caps(
    total_useful_work=T,
    lambdas=data["lambdas"],
    recoveries=data["recoveries"],
    capacities=data["capacities"],
    min_interval=0.0,
)

print("status:", result["status"])
print("objective:", result["objective_value"])
print("intervals (hours):", result["intervals"] / SECONDS_PER_HOUR)
print("sum (hours):", np.sum(result["intervals"]) / SECONDS_PER_HOUR)


import matplotlib.pyplot as plt

intervals = result["intervals"] / SECONDS_PER_HOUR
caps = data["capacities"] / SECONDS_PER_HOUR
labels = data["stage_labels"]

plt.figure(figsize=(12, 4))
x = np.arange(len(intervals))
plt.bar(x, intervals, label="Allocated useful work")
plt.plot(x, caps, "ro--", label="Capacity")
plt.xticks(x, labels, rotation=45, ha="right")
plt.ylabel("Hours")
plt.title("Optimal interval allocation across repeated daily stages")
plt.legend()
plt.tight_layout()
plt.show()