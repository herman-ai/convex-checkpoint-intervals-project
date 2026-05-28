import math
import random
from dataclasses import dataclass
from typing import Callable, Optional, Dict, List

import numpy as np

SECONDS_PER_HOUR = 3600.0


# -----------------------------
# Numerical integration helpers
# -----------------------------

def integrate_lambda(lambda_fn: Callable[[float], float], a: float, b: float, num_steps: int = 256) -> float:
    """
    Numerically integrate lambda_fn over [a, b] using trapezoidal rule.
    """
    if b < a:
        raise ValueError("b must be >= a")
    if a == b:
        return 0.0
    xs = np.linspace(a, b, num_steps + 1)
    ys = np.array([lambda_fn(x) for x in xs], dtype=float)
    return np.trapezoid(ys, xs)


def cumulative_hazard_segments(lambda_fn: Callable[[float], float], T_knots: np.ndarray, num_steps: int = 256) -> np.ndarray:
    """
    Compute L_k = ∫_{T_{k-1}}^{T_k} lambda(u) du for each interval.
    """
    K = len(T_knots) - 1
    L = np.zeros(K, dtype=float)
    for k in range(K):
        L[k] = integrate_lambda(lambda_fn, T_knots[k], T_knots[k + 1], num_steps=num_steps)
    return L


def interval_work_integral(lambda_fn: Callable[[float], float], a: float, b: float, num_steps: int = 256) -> float:
    """
    Compute I_k = ∫_a^b exp(-Λ_k(u)) du  where  Λ_k(u) = ∫_a^u lambda(v) dv.

    Uses the trapezoidal rule to build the cumulative hazard incrementally,
    then integrates exp(-Λ_k(u)) with the trapezoidal rule.
    """
    if a == b:
        return 0.0
    xs = np.linspace(a, b, num_steps + 1)
    lam_vals = np.array([lambda_fn(x) for x in xs], dtype=float)
    dx = (b - a) / num_steps
    cum_hazard = np.zeros(num_steps + 1, dtype=float)
    for j in range(1, num_steps + 1):
        cum_hazard[j] = cum_hazard[j - 1] + 0.5 * (lam_vals[j - 1] + lam_vals[j]) * dx
    integrand = np.exp(-cum_hazard)
    return float(np.trapezoid(integrand, xs))


def interval_work_integrals(lambda_fn: Callable[[float], float], T_knots: np.ndarray, num_steps: int = 256) -> np.ndarray:
    """
    Compute I_k for each interval k.
    """
    K = len(T_knots) - 1
    I = np.zeros(K, dtype=float)
    for k in range(K):
        I[k] = interval_work_integral(lambda_fn, T_knots[k], T_knots[k + 1], num_steps=num_steps)
    return I


# -----------------------------
# Stable scalar helpers
# -----------------------------

def phi_of_L(L: np.ndarray) -> np.ndarray:
    """
    phi(L) = (e^L - 1) / L, with stable small-L handling.
    """
    L = np.asarray(L, dtype=float)
    out = np.empty_like(L)
    small = np.abs(L) < 1e-8
    out[small] = 1.0 + 0.5 * L[small] + (L[small] ** 2) / 6.0
    out[~small] = np.expm1(L[~small]) / L[~small]
    return out


def psi_of_L(L: np.ndarray) -> np.ndarray:
    """
    psi(L) = (L e^L - (e^L - 1)) / L^2, with stable small-L handling.
    """
    L = np.asarray(L, dtype=float)
    out = np.empty_like(L)
    small = np.abs(L) < 1e-8
    # Series: 1/2 + L/3 + L^2/8 + ...
    out[small] = 0.5 + L[small] / 3.0 + (L[small] ** 2) / 8.0
    out[~small] = (L[~small] * np.exp(L[~small]) - np.expm1(L[~small])) / (L[~small] ** 2)
    return out


# -----------------------------
# Objective and gradients
# -----------------------------

@dataclass
class UsefulWorkHazardProblem:
    total_useful_work: float
    num_intervals: int
    epsilon: float
    lambda_fn: Callable[[float], float]
    q: np.ndarray  # shape (K,)
    checkpoint_costs: Optional[np.ndarray] = None  # optional shape (K,)

    def __post_init__(self):
        self.q = np.asarray(self.q, dtype=float)
        if self.checkpoint_costs is not None:
            self.checkpoint_costs = np.asarray(self.checkpoint_costs, dtype=float)

        if len(self.q) != self.num_intervals:
            raise ValueError("q must have length num_intervals")
        if self.checkpoint_costs is not None and len(self.checkpoint_costs) != self.num_intervals:
            raise ValueError("checkpoint_costs must have length num_intervals")
        if self.total_useful_work <= 0:
            raise ValueError("total_useful_work must be positive")
        if self.epsilon < 0:
            raise ValueError("epsilon must be nonnegative")
        if self.total_useful_work <= self.num_intervals * self.epsilon:
            raise ValueError("total_useful_work must exceed K * epsilon")

    def delta_to_knots(self, delta: np.ndarray) -> np.ndarray:
        delta = np.asarray(delta, dtype=float)
        if len(delta) != self.num_intervals:
            raise ValueError("delta has wrong length")
        T_knots = np.zeros(self.num_intervals + 1, dtype=float)
        T_knots[1:] = np.cumsum(delta)
        return T_knots

    def internal_knots_to_full(self, T_internal: np.ndarray) -> np.ndarray:
        T_internal = np.asarray(T_internal, dtype=float)
        if len(T_internal) != self.num_intervals - 1:
            raise ValueError("T_internal has wrong length")
        T_knots = np.zeros(self.num_intervals + 1, dtype=float)
        T_knots[1:-1] = T_internal
        T_knots[-1] = self.total_useful_work
        return T_knots

    def full_knots_to_delta(self, T_knots: np.ndarray) -> np.ndarray:
        return np.diff(T_knots)

    def objective_from_delta(self, delta: np.ndarray, num_steps: int = 256) -> float:
        T_knots = self.delta_to_knots(delta)
        L = cumulative_hazard_segments(self.lambda_fn, T_knots, num_steps=num_steps)
        I = interval_work_integrals(self.lambda_fn, T_knots, num_steps=num_steps)

        val = np.sum(np.exp(L) * I + self.q * np.expm1(L))
        if self.checkpoint_costs is not None:
            val += np.sum(self.checkpoint_costs)
        return float(val)

    def objective_from_internal_knots(self, T_internal: np.ndarray, num_steps: int = 256) -> float:
        T_knots = self.internal_knots_to_full(T_internal)
        delta = self.full_knots_to_delta(T_knots)
        return self.objective_from_delta(delta, num_steps=num_steps)

    def gradient_internal_knots(self, T_internal: np.ndarray, num_steps: int = 256) -> np.ndarray:
        """
        Gradient with respect to T_1, ..., T_{K-1}.

        Exact formula:
            df/dT_k = lambda(T_k) * H_k + 1 - lambda(T_k) * H_{k+1} - exp(L_{k+1})
        where H_k = exp(L_k) * (I_k + q_k).
        """
        T_knots = self.internal_knots_to_full(T_internal)
        L = cumulative_hazard_segments(self.lambda_fn, T_knots, num_steps=num_steps)
        I = interval_work_integrals(self.lambda_fn, T_knots, num_steps=num_steps)

        exp_L = np.exp(L)
        # H_k = exp(L_k) * (I_k + q_k)
        H = exp_L * (I + self.q)

        grad = np.zeros(self.num_intervals - 1, dtype=float)
        for i in range(self.num_intervals - 1):
            lam_t = self.lambda_fn(T_knots[i + 1])
            grad[i] = lam_t * H[i] + 1.0 - exp_L[i + 1] * (1.0 + lam_t * self.q[i + 1])
        return grad

    def gradient_delta(self, delta: np.ndarray, num_steps: int = 256) -> np.ndarray:
        """
        Gradient in the Delta-parameterization using suffix sums.
        """
        T_knots = self.delta_to_knots(delta)
        T_internal = T_knots[1:-1]
        gT = self.gradient_internal_knots(T_internal, num_steps=num_steps)

        gDelta = np.zeros(self.num_intervals, dtype=float)
        # By the partner's formula
        gDelta[-1] = 0.0
        for j in range(self.num_intervals - 2, -1, -1):
            if j == self.num_intervals - 2:
                gDelta[j] = gT[j]
            else:
                gDelta[j] = gDelta[j + 1] + gT[j]
        return gDelta
    