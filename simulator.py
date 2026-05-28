from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Callable, List, Tuple, Dict, Optional


# -----------------------------
# Time-dependent environment
# -----------------------------

HazardFn = Callable[[float], float]
RecoveryFn = Callable[[float], float]
CheckpointCostFn = Callable[[float], float]


SECONDS_PER_HOUR = 3600.0
SECONDS_PER_DAY = 24.0 * SECONDS_PER_HOUR


def mod_day(t: float) -> float:
    """Return time-of-day in seconds."""
    return t % SECONDS_PER_DAY


@dataclass
class DailyEnvironment:
    """
    Simple piecewise-constant daily environment.

    Peak hours are specified in hours in [0, 24). For example:
        peak_start_hour = 9.0
        peak_end_hour = 18.0
    means 9am to 6pm local cluster time.
    """

    lambda_offpeak: float
    lambda_peak: float
    recovery_offpeak: float
    recovery_peak: float
    checkpoint_cost_offpeak: float
    checkpoint_cost_peak: float
    peak_start_hour: float = 9.0
    peak_end_hour: float = 18.0

    def _is_peak(self, t: float) -> bool:
        h = mod_day(t) / SECONDS_PER_HOUR
        return self.peak_start_hour <= h < self.peak_end_hour

    def hazard(self, t: float) -> float:
        return self.lambda_peak if self._is_peak(t) else self.lambda_offpeak

    def recovery_overhead(self, t: float) -> float:
        return self.recovery_peak if self._is_peak(t) else self.recovery_offpeak

    def checkpoint_cost(self, t: float) -> float:
        return self.checkpoint_cost_peak if self._is_peak(t) else self.checkpoint_cost_offpeak


# -----------------------------
# Schedule representation
# -----------------------------


def equal_work_schedule(total_useful_work: float, num_intervals: int) -> List[float]:
    if num_intervals <= 0:
        raise ValueError("num_intervals must be positive")
    delta = total_useful_work / num_intervals
    return [delta for _ in range(num_intervals)]


# -----------------------------
# Nonhomogeneous Poisson simulation
# -----------------------------


def simulate_failure_time_piecewise_constant_hazard(
    start_time: float,
    max_run_time: float,
    env: DailyEnvironment,
    rng: random.Random,
    dt: float = 300.0,
) -> Optional[float]:
    """
    Simulate the elapsed running time until failure under a time-varying hazard.

    We use a simple piecewise-constant approximation over small time steps dt.
    Returns:
        - elapsed time to failure in [0, max_run_time], if failure occurs
        - None, if the process survives the whole max_run_time

    This is a starter simulator, not an optimized exact method.
    """
    elapsed = 0.0
    while elapsed < max_run_time:
        step = min(dt, max_run_time - elapsed)
        t = start_time + elapsed
        lam = env.hazard(t)
        if lam < 0:
            raise ValueError("hazard must be nonnegative")

        # Failure probability within this step under constant hazard approximation.
        p_fail = 1.0 - math.exp(-lam * step)
        if rng.random() < p_fail:
            # Sample failure location inside the step, approximately uniformly.
            # For small dt this is a reasonable starter approximation.
            return elapsed + rng.uniform(0.0, step)

        elapsed += step

    return None


# -----------------------------
# Interval and job simulation
# -----------------------------


@dataclass
class IntervalResult:
    start_time: float
    useful_work: float
    completion_time: float
    checkpoint_time: float
    num_failures: int


@dataclass
class JobRunResult:
    total_runtime: float
    interval_results: List[IntervalResult]
    total_failures: int


def simulate_one_interval(
    start_time: float,
    useful_work: float,
    env: DailyEnvironment,
    rng: random.Random,
    dt: float = 300.0,
) -> IntervalResult:
    """
    Simulate one interval under restart-until-success dynamics.

    Assumption for now:
    - 1 unit of useful work takes 1 unit of running time when the job is progressing.
    - If a failure occurs before the interval finishes, all progress in the interval is lost.
    - After a failure at wall-clock time t, the restart overhead is env.recovery_overhead(t).
    - After successful completion, checkpoint cost is env.checkpoint_cost(t_checkpoint).
    """
    current_time = start_time
    failures = 0

    while True:
        failure_elapsed = simulate_failure_time_piecewise_constant_hazard(
            start_time=current_time,
            max_run_time=useful_work,
            env=env,
            rng=rng,
            dt=dt,
        )

        if failure_elapsed is None:
            # Success
            current_time += useful_work
            checkpoint_time = env.checkpoint_cost(current_time)
            current_time += checkpoint_time
            return IntervalResult(
                start_time=start_time,
                useful_work=useful_work,
                completion_time=current_time,
                checkpoint_time=checkpoint_time,
                num_failures=failures,
            )

        # Failure and retry
        failures += 1
        fail_time = current_time + failure_elapsed
        recovery = env.recovery_overhead(fail_time)
        current_time = fail_time + recovery


def simulate_job_run(
    interval_workloads: List[float],
    env: DailyEnvironment,
    seed: int = 0,
    dt: float = 300.0,
) -> JobRunResult:
    rng = random.Random(seed)
    interval_results: List[IntervalResult] = []
    current_time = 0.0
    total_failures = 0

    for work in interval_workloads:
        result = simulate_one_interval(
            start_time=current_time,
            useful_work=work,
            env=env,
            rng=rng,
            dt=dt,
        )
        interval_results.append(result)
        current_time = result.completion_time
        total_failures += result.num_failures

    return JobRunResult(
        total_runtime=current_time,
        interval_results=interval_results,
        total_failures=total_failures,
    )


# -----------------------------
# Monte Carlo evaluation
# -----------------------------


def monte_carlo_runtime(
    interval_workloads: List[float],
    env: DailyEnvironment,
    num_trials: int = 100,
    base_seed: int = 0,
    dt: float = 300.0,
) -> Dict[str, float]:
    runtimes = []
    failures = []
    for i in range(num_trials):
        result = simulate_job_run(
            interval_workloads=interval_workloads,
            env=env,
            seed=base_seed + i,
            dt=dt,
        )
        runtimes.append(result.total_runtime)
        failures.append(result.total_failures)

    mean_runtime = sum(runtimes) / len(runtimes)
    mean_failures = sum(failures) / len(failures)
    if len(runtimes) > 1:
        var = sum((x - mean_runtime) ** 2 for x in runtimes) / (len(runtimes) - 1)
        stderr = math.sqrt(var / len(runtimes))
    else:
        stderr = 0.0

    return {
        "mean_runtime": mean_runtime,
        "stderr_runtime": stderr,
        "mean_failures": mean_failures,
        "num_trials": float(num_trials),
    }


# -----------------------------
# Simple schedule heuristics
# -----------------------------


def front_load_safe_periods(
    total_useful_work: float,
    num_intervals: int,
    safe_fraction: float = 0.6,
) -> List[float]:
    """
    A toy non-equal schedule heuristic.

    Earlier intervals get more work, later intervals get less.
    This is just a starter baseline for experimentation.
    """
    if num_intervals <= 0:
        raise ValueError("num_intervals must be positive")
    if not (0.0 < safe_fraction < 1.0):
        raise ValueError("safe_fraction must be in (0, 1)")

    weights = [safe_fraction ** i for i in range(num_intervals)]
    total_weight = sum(weights)
    return [total_useful_work * w / total_weight for w in weights]


# -----------------------------
# Example usage
# -----------------------------


if __name__ == "__main__":
    env = DailyEnvironment(
        lambda_offpeak=1.0 / (48.0 * SECONDS_PER_HOUR),   # safer overnight
        lambda_peak=1.0 / (16.0 * SECONDS_PER_HOUR),      # riskier during busy hours
        recovery_offpeak=5.0 * 60.0,
        recovery_peak=30.0 * 60.0,
        checkpoint_cost_offpeak=2.0 * 60.0,
        checkpoint_cost_peak=8.0 * 60.0,
        peak_start_hour=9.0,
        peak_end_hour=18.0,
    )

    total_useful_work = 24.0 * SECONDS_PER_HOUR
    num_intervals = 8

    equal_schedule = equal_work_schedule(total_useful_work, num_intervals)
    unequal_schedule = front_load_safe_periods(total_useful_work, num_intervals, safe_fraction=0.8)

    equal_stats = monte_carlo_runtime(equal_schedule, env, num_trials=200, base_seed=123)
    unequal_stats = monte_carlo_runtime(unequal_schedule, env, num_trials=200, base_seed=123)

    print("Equal schedule:")
    print(equal_schedule)
    print(equal_stats)

    print("\nUnequal schedule:")
    print(unequal_schedule)
    print(unequal_stats)

