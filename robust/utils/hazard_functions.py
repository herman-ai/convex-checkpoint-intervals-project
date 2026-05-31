
from utils.helpers import SECONDS_PER_HOUR


def lambda_step(t: float, switch_time: float, lambda_low: float, lambda_high: float) -> float:
    return lambda_low if t < switch_time else lambda_high

T = 48 * SECONDS_PER_HOUR

def lambda_fn_step(t: float) -> float:
    return lambda_step(
        t=t,
        switch_time=24 * SECONDS_PER_HOUR,
        lambda_low=1.0 / (40 * SECONDS_PER_HOUR),
        lambda_high=1.0 / (6 * SECONDS_PER_HOUR),
    )

def lambda_fn_example(t: float) -> float:
    """
    Example hazard as a function of useful work completed.
    Units: if useful work is measured in seconds, lambda is per second.
    """
    base = 1.0 / (24.0 * SECONDS_PER_HOUR)   # baseline MTBF ~ 24 hours
    slope = 1.0 / ((200.0 * SECONDS_PER_HOUR) ** 2)
    return base + slope * t


def lambda_linear(t: float, T: float, lambda_start: float, lambda_end: float) -> float:
    alpha = min(max(t / T, 0.0), 1.0)
    return lambda_start + (lambda_end - lambda_start) * alpha

def lambda_fn_linear(t: float) -> float:
    return lambda_linear(
        t=t,
        T=T,
        lambda_start=1.0 / (36 * SECONDS_PER_HOUR),
        lambda_end=1.0 / (4 * SECONDS_PER_HOUR),
    )

def lambda_convex(t: float, T: float, lambda_start: float, lambda_end: float, power: float = 2.0) -> float:
    alpha = min(max(t / T, 0.0), 1.0)
    return lambda_start + (lambda_end - lambda_start) * (alpha ** power)

def lambda_fn_convex(t: float) -> float:
    return lambda_convex(
        t=t,
        T=T,
        lambda_start=1.0 / (40 * SECONDS_PER_HOUR),
        lambda_end=1.0 / (3 * SECONDS_PER_HOUR),
        power=2.5,
    )


def lambda_three_phase(
    t: float,
    tau1: float,
    tau2: float,
    lambda1: float,
    lambda2: float,
    lambda3: float,
) -> float:
    if t < tau1:
        return lambda1
    elif t < tau2:
        return lambda2
    return lambda3


def lambda_fn_three_phase(t: float) -> float:
    return lambda_three_phase(
        t=t,
        tau1=16 * SECONDS_PER_HOUR,
        tau2=36 * SECONDS_PER_HOUR,
        lambda1=1.0 / (50 * SECONDS_PER_HOUR),
        lambda2=1.0 / (18 * SECONDS_PER_HOUR),
        lambda3=1.0 / (4 * SECONDS_PER_HOUR),
    )

def q_step(t: float, switch_time: float, q_low: float, q_high: float) -> float:
    return q_low if t < switch_time else q_high

def q_linear(t: float, T: float, q_start: float, q_end: float) -> float:
    alpha = min(max(t / T, 0.0), 1.0)
    return q_start + (q_end - q_start) * alpha