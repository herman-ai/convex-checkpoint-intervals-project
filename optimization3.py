import numpy as np
from simulator import DailyEnvironment

SECONDS_PER_HOUR = 3600.0
SECONDS_PER_DAY = 24.0 * SECONDS_PER_HOUR

def integrate_over_interval(
    fn,
    start_time: float,
    duration: float,
    dt: float = 300.0,
) -> float:
    """
    Numerically integrate fn(t) over [start_time, start_time + duration]
    using a simple left Riemann sum.
    """
    if duration < 0:
        raise ValueError("duration must be nonnegative")
    if duration == 0:
        return 0.0
    if dt <= 0:
        raise ValueError("dt must be positive")

    total = 0.0
    elapsed = 0.0
    while elapsed < duration:
        step = min(dt, duration - elapsed)
        t = start_time + elapsed
        total += fn(t) * step
        elapsed += step
    return total


def cumulative_hazard_over_interval(
    start_time: float,
    useful_work: float,
    env,
    dt: float = 300.0,
) -> float:
    return integrate_over_interval(env.hazard, start_time, useful_work, dt=dt)


def average_hazard_over_interval(
    start_time: float,
    useful_work: float,
    env,
    dt: float = 300.0,
) -> float:
    if useful_work <= 0:
        raise ValueError("useful_work must be positive")
    H = cumulative_hazard_over_interval(start_time, useful_work, env, dt=dt)
    return H / useful_work


def average_recovery_over_interval(
    start_time: float,
    useful_work: float,
    env,
    dt: float = 300.0,
) -> float:
    if useful_work <= 0:
        raise ValueError("useful_work must be positive")
    integral_q = integrate_over_interval(env.recovery_overhead, start_time, useful_work, dt=dt)
    return integral_q / useful_work


def approx_interval_runtime_path(
    start_time: float,
    useful_work: float,
    env,
    dt: float = 300.0,
) -> float:
    """
    Path-based approximation:
        H(s,Δ) = ∫ λ(u) du
        λ_bar = H / Δ
        q_bar = (1/Δ) ∫ q(u) du

        V_path(s,Δ) = (1/λ_bar + q_bar) * (exp(H) - 1)

    If H is tiny, use a stable small-H approximation.
    """
    if useful_work <= 0:
        return 0.0

    H = cumulative_hazard_over_interval(start_time, useful_work, env, dt=dt)
    lam_bar = H / useful_work
    q_bar = average_recovery_over_interval(start_time, useful_work, env, dt=dt)

    if lam_bar <= 0:
        return useful_work

    # Numerically stable for small H
    return (1.0 / lam_bar + q_bar) * np.expm1(H)


def compute_start_times_path(schedule, env, dt: float = 300.0):
    """
    Compute approximate wall-clock start times using the path-based interval runtime.
    """
    schedule = np.asarray(schedule, dtype=float)
    K = len(schedule)
    start_times = np.zeros(K, dtype=float)
    current_time = 0.0

    for k, work in enumerate(schedule):
        start_times[k] = current_time
        runtime = approx_interval_runtime_path(current_time, work, env, dt=dt)
        checkpoint = env.checkpoint_cost(current_time + runtime)
        current_time = current_time + runtime + checkpoint

    return start_times

def frozen_stage_parameters_from_schedule(schedule, env, dt: float = 300.0):
    """
    For the current schedule, compute:
      - start times
      - average hazard over each interval path
      - average recovery over each interval path

    These frozen values can be used in the convex subproblem.
    """
    schedule = np.asarray(schedule, dtype=float)
    K = len(schedule)
    start_times = np.zeros(K, dtype=float)
    lambdas = np.zeros(K, dtype=float)
    recoveries = np.zeros(K, dtype=float)

    current_time = 0.0
    for k, work in enumerate(schedule):
        start_times[k] = current_time

        lam_bar = average_hazard_over_interval(current_time, work, env, dt=dt)
        q_bar = average_recovery_over_interval(current_time, work, env, dt=dt)

        lambdas[k] = lam_bar
        recoveries[k] = q_bar

        runtime = approx_interval_runtime_path(current_time, work, env, dt=dt)
        checkpoint = env.checkpoint_cost(current_time + runtime)
        current_time = current_time + runtime + checkpoint

    return {
        "start_times": start_times,
        "lambdas": lambdas,
        "recoveries": recoveries,
    }


def solve_frozen_convex_subproblem_path(
    total_useful_work: float,
    schedule,
    env,
    min_interval: float = 0.0,
    dt: float = 300.0,
):
    """
    Build frozen parameters from the current schedule using the path-based approximation,
    then solve the convex subproblem.
    """
    try:
        import cvxpy as cp
    except ImportError as e:
        raise ImportError("cvxpy is required") from e

    frozen = frozen_stage_parameters_from_schedule(schedule, env, dt=dt)
    lambdas = frozen["lambdas"]
    recoveries = frozen["recoveries"]

    if np.any(lambdas <= 0):
        raise ValueError("Frozen average hazards must be positive")

    K = len(lambdas)
    Delta = cp.Variable(K)

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
        "start_times": frozen["start_times"],
        "frozen_lambdas": lambdas,
        "frozen_recoveries": recoveries,
    }


def optimize_schedule_fixed_point_path(
    total_useful_work: float,
    num_intervals: int,
    env,
    max_iters: int = 20,
    damping: float = 0.3,
    tol: float = 1e-4,
    min_interval: float = 0.0,
    dt: float = 300.0,
    initial_schedule=None,
):
    if not (0.0 < damping <= 1.0):
        raise ValueError("damping must be in (0, 1]")
    if num_intervals <= 0:
        raise ValueError("num_intervals must be positive")

    if initial_schedule is None:
        schedule = np.full(num_intervals, total_useful_work / num_intervals, dtype=float)
    else:
        schedule = np.asarray(initial_schedule, dtype=float)
        if len(schedule) != num_intervals:
            raise ValueError("initial_schedule length must equal num_intervals")
        if abs(np.sum(schedule) - total_useful_work) > 1e-6:
            raise ValueError("initial_schedule must sum to total_useful_work")

    history = []

    for it in range(max_iters):
        subproblem = solve_frozen_convex_subproblem_path(
            total_useful_work=total_useful_work,
            schedule=schedule,
            env=env,
            min_interval=min_interval,
            dt=dt,
        )

        if subproblem["intervals"] is None:
            return {
                "status": subproblem["status"],
                "schedule": None,
                "history": history,
            }

        new_schedule = np.array(subproblem["intervals"], dtype=float)
        updated_schedule = (1.0 - damping) * schedule + damping * new_schedule
        updated_schedule *= total_useful_work / np.sum(updated_schedule)

        diff = np.linalg.norm(updated_schedule - schedule)

        history.append(
            {
                "iter": it,
                "schedule": schedule.copy(),
                "start_times": subproblem["start_times"].copy(),
                "frozen_lambdas": subproblem["frozen_lambdas"].copy(),
                "frozen_recoveries": subproblem["frozen_recoveries"].copy(),
                "subproblem_objective": subproblem["objective_value"],
                "update_norm": diff,
            }
        )

        schedule = updated_schedule
        if diff < tol:
            break

    final_start_times = compute_start_times_path(schedule, env, dt=dt)
    approx_total_runtime = 0.0
    current_time = 0.0
    for work in schedule:
        runtime = approx_interval_runtime_path(current_time, work, env, dt=dt)
        checkpoint = env.checkpoint_cost(current_time + runtime)
        approx_total_runtime += runtime + checkpoint
        current_time += runtime + checkpoint

    return {
        "status": "converged" if len(history) < max_iters or history[-1]["update_norm"] < tol else "max_iters_reached",
        "schedule": schedule,
        "start_times": final_start_times,
        "approx_total_runtime": approx_total_runtime,
        "history": history,
    }


T = 48 * SECONDS_PER_HOUR
K = 6

T = 24 * SECONDS_PER_HOUR
K = 8

env = DailyEnvironment(
    lambda_offpeak=1.0 / (48.0 * SECONDS_PER_HOUR),
    lambda_peak=1.0 / (16.0 * SECONDS_PER_HOUR),
    recovery_offpeak=5.0 * 60.0,
    recovery_peak=30.0 * 60.0,
    checkpoint_cost_offpeak=2.0 * 60.0,
    checkpoint_cost_peak=8.0 * 60.0,
    peak_start_hour=9.0,
    peak_end_hour=18.0,
)

result = optimize_schedule_fixed_point_path(
    total_useful_work=T,
    num_intervals=K,
    env=env,
    max_iters=20,
    damping=0.3,
    tol=1e-3,
)

print("status:", result["status"])
print("schedule (hours):", result["schedule"] / SECONDS_PER_HOUR)
print("start_times (hours):", result["start_times"] / SECONDS_PER_HOUR)
print("approx_total_runtime (hours):", result["approx_total_runtime"] / SECONDS_PER_HOUR)

result = optimize_schedule_fixed_point_path(
    total_useful_work=T,
    num_intervals=K,
    env=env,
    max_iters=20,
    damping=0.3,
    tol=1e-3,
    dt=300.0,
)

print("status:", result["status"])
print("schedule (hours):", result["schedule"] / SECONDS_PER_HOUR)
print("start_times (hours):", result["start_times"] / SECONDS_PER_HOUR)
print("approx_total_runtime (hours):", result["approx_total_runtime"] / SECONDS_PER_HOUR)

import numpy as np
import matplotlib.pyplot as plt


def plot_schedule_comparison(result, total_useful_work):
    schedule = np.asarray(result["schedule"], dtype=float)
    K = len(schedule)
    equal_schedule = np.full(K, total_useful_work / K)

    x = np.arange(K)

    plt.figure(figsize=(9, 4))
    plt.bar(x - 0.2, equal_schedule / 3600.0, width=0.4, label="Equal spacing")
    plt.bar(x + 0.2, schedule / 3600.0, width=0.4, label="Optimized")
    plt.xlabel("Interval index")
    plt.ylabel("Useful work in interval (hours)")
    plt.title("Equal vs optimized checkpoint schedule")
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_start_times(result):
    starts = np.asarray(result["start_times"], dtype=float)
    x = np.arange(len(starts))

    plt.figure(figsize=(9, 4))
    plt.plot(x, starts / 3600.0, "o-")
    plt.xlabel("Interval index")
    plt.ylabel("Start time (hours)")
    plt.title("Approximate wall-clock start times")
    plt.tight_layout()
    plt.show()

def plot_start_times(result):
    starts = np.asarray(result["start_times"], dtype=float)
    x = np.arange(len(starts))

    plt.figure(figsize=(9, 4))
    plt.plot(x, starts / 3600.0, "o-")
    plt.xlabel("Interval index")
    plt.ylabel("Start time (hours)")
    plt.title("Approximate wall-clock start times")
    plt.tight_layout()
    plt.show()

def plot_convergence(result):
    history = result["history"]
    if len(history) == 0:
        raise ValueError("No history found in result")

    iters = [h["iter"] for h in history]
    update_norms = [h["update_norm"] for h in history]
    objectives = [h["subproblem_objective"] for h in history]

    fig, ax1 = plt.subplots(figsize=(9, 4))

    ax1.plot(iters, update_norms, "o-", label="Update norm")
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Update norm")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(iters, objectives, "s--", label="Frozen subproblem objective")
    ax2.set_ylabel("Subproblem objective")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")

    plt.title("Fixed-point convergence")
    plt.tight_layout()
    plt.show()


def plot_schedule_timeline(result):
    schedule = np.asarray(result["schedule"], dtype=float)
    starts = np.asarray(result["start_times"], dtype=float)

    plt.figure(figsize=(11, 2.5))

    for k, (s, w) in enumerate(zip(starts, schedule)):
        plt.barh(
            y=0,
            width=w / 3600.0,
            left=s / 3600.0,
            height=0.5,
            align="center",
            alpha=0.8,
            edgecolor="black",
        )
        plt.text((s + w / 2) / 3600.0, 0, f"{k}", ha="center", va="center", fontsize=9)

    plt.xlabel("Wall-clock time (hours)")
    plt.yticks([])
    plt.title("Optimized interval schedule on wall-clock timeline")
    plt.tight_layout()
    plt.show()

def plot_schedule_timeline_with_peak_windows(result, peak_start_hour=9.0, peak_end_hour=18.0):
    schedule = np.asarray(result["schedule"], dtype=float)
    starts = np.asarray(result["start_times"], dtype=float)

    end_time = np.max(starts + schedule)
    total_hours = end_time / 3600.0
    num_days = int(np.ceil(total_hours / 24.0))

    plt.figure(figsize=(12, 3))

    # Shade peak windows
    for d in range(num_days):
        left = 24.0 * d + peak_start_hour
        right = 24.0 * d + peak_end_hour
        plt.axvspan(left, right, alpha=0.15)

    # Plot intervals
    for k, (s, w) in enumerate(zip(starts, schedule)):
        left = s / 3600.0
        width = w / 3600.0
        plt.barh(0, width=width, left=left, height=0.5, edgecolor="black")
        plt.text(left + width / 2, 0, str(k), ha="center", va="center", fontsize=9)

    plt.xlabel("Wall-clock time (hours)")
    plt.yticks([])
    plt.title("Optimized schedule with shaded peak windows")
    plt.tight_layout()
    plt.show()

def plot_runtime_comparison(equal_stats, optimized_stats):
    labels = ["Equal spacing", "Optimized"]
    means = [
        equal_stats["mean_runtime"] / 3600.0,
        optimized_stats["mean_runtime"] / 3600.0,
    ]
    errors = [
        equal_stats["stderr_runtime"] / 3600.0,
        optimized_stats["stderr_runtime"] / 3600.0,
    ]

    x = np.arange(len(labels))

    plt.figure(figsize=(6, 4))
    plt.bar(x, means, yerr=errors, capsize=5)
    plt.xticks(x, labels)
    plt.ylabel("Mean runtime (hours)")
    plt.title("Simulation runtime comparison")
    plt.tight_layout()
    plt.show()


plot_schedule_comparison(result, T)
plot_start_times(result)
# plot_final_frozen_parameters(result)
plot_convergence(result)
plot_schedule_timeline(result)
plot_schedule_timeline_with_peak_windows(result, peak_start_hour=9.0, peak_end_hour=18.0)