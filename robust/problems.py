"""
Robust checkpoint-scheduling problem classes.

Each class wraps a specific hazard-rate family and exposes the same
interface (objective_from_delta, gradient_delta, …) that the generic
optimizers (PGD, MD, ADMM) expect.
"""

from __future__ import annotations

import numpy as np

from utils.helpers import UsefulWorkHazardProblem


# ── helper ────────────────────────────────────────────────────────────────────

def make_step_lambda_fn(tau: np.ndarray, theta: np.ndarray):
    """Return a scalar-valued step-hazard function lambda(t) for given tau/theta."""
    tau   = np.asarray(tau,   dtype=float)
    theta = np.asarray(theta, dtype=float)

    def lambda_fn(t: float) -> float:
        j = np.searchsorted(tau, t, side="right") - 1
        j = min(max(j, 0), len(theta) - 1)
        return float(theta[j])

    return lambda_fn


# ── RobustStepHazardProblem ───────────────────────────────────────────────────

class RobustStepHazardProblem:
    """
    Robust checkpoint scheduling with piecewise-constant (step) hazard rate.

    Uncertainty set: theta in [theta_hat - rho, theta_hat + rho], theta >= 0.
    Worst case: theta* = max(theta_hat + rho, 0)  (upper corner).

    All objective and gradient computations are performed exactly (no quadrature)
    by splitting intervals across phase boundaries.
    """

    def __init__(
        self,
        total_useful_work: float,
        num_intervals: int,
        epsilon: float,
        tau: np.ndarray,           # shape (n+1,), includes 0 and T
        theta_hat: np.ndarray,     # shape (n,)
        rho: np.ndarray | float,   # per-coordinate half-width, shape (n,) or scalar
        q: np.ndarray,             # shape (K,)
        checkpoint_costs: np.ndarray | None = None,
    ):
        self.T = float(total_useful_work)
        self.K = int(num_intervals)
        self.epsilon = float(epsilon)
        self.tau = np.asarray(tau, dtype=float)
        self.theta_hat = np.asarray(theta_hat, dtype=float)
        _rho = np.asarray(rho, dtype=float)
        self.rho = np.broadcast_to(_rho, self.theta_hat.shape).copy()
        self.q = np.asarray(q, dtype=float)
        self.checkpoint_costs = (
            None if checkpoint_costs is None
            else np.asarray(checkpoint_costs, dtype=float)
        )

        if self.tau[0] != 0.0 or abs(self.tau[-1] - self.T) > 1e-9:
            raise ValueError("tau must start at 0 and end at total useful work T")
        if len(self.theta_hat) != len(self.tau) - 1:
            raise ValueError("theta_hat must have length len(tau)-1")
        if len(self.q) != self.K:
            raise ValueError("q must have length K")
        if self.checkpoint_costs is not None and len(self.checkpoint_costs) != self.K:
            raise ValueError("checkpoint_costs must have length K")
        if self.T <= self.K * self.epsilon:
            raise ValueError("Need T > K*epsilon")
        if np.any(self.theta_hat < 0):
            raise ValueError("theta_hat must be nonnegative")
        if np.any(self.rho < 0):
            raise ValueError("rho must be nonnegative")

    # ── aliases so generic optimizers can use this problem ────────────────────

    @property
    def num_intervals(self) -> int:
        return self.K

    @property
    def total_useful_work(self) -> float:
        return self.T

    @property
    def theta_worst(self) -> np.ndarray:
        """Worst-case corner of the additive box Θ = {θ : |θ - θ̂| ≤ ρ, θ ≥ 0}."""
        return np.maximum(self.theta_hat + self.rho, 0.0)

    # ── knot / delta conversions ──────────────────────────────────────────────

    def delta_to_knots(self, delta: np.ndarray) -> np.ndarray:
        delta = np.asarray(delta, dtype=float)
        T_knots = np.zeros(self.K + 1, dtype=float)
        T_knots[1:] = np.cumsum(delta)
        return T_knots

    def internal_knots_to_full(self, T_internal: np.ndarray) -> np.ndarray:
        T_internal = np.asarray(T_internal, dtype=float)
        T_knots = np.zeros(self.K + 1, dtype=float)
        T_knots[1:-1] = T_internal
        T_knots[-1] = self.T
        return T_knots

    def full_knots_to_delta(self, T_knots: np.ndarray) -> np.ndarray:
        return np.diff(T_knots)

    # ── geometry helpers ──────────────────────────────────────────────────────

    def overlap_lengths(self, T_knots: np.ndarray) -> np.ndarray:
        """s_k^j = length of interval k lying in phase j.  Shape (K, n_phases)."""
        K = self.K
        n = len(self.theta_hat)
        s = np.zeros((K, n), dtype=float)
        for k in range(K):
            a, b = T_knots[k], T_knots[k + 1]
            for j in range(n):
                left  = max(a, self.tau[j])
                right = min(b, self.tau[j + 1])
                s[k, j] = max(0.0, right - left)
        return s

    def cumulative_hazards(
        self, T_knots: np.ndarray, theta: np.ndarray | None = None
    ) -> np.ndarray:
        """L_k(theta) = sum_j theta_j s_k^j."""
        if theta is None:
            theta = self.theta_worst
        return self.overlap_lengths(T_knots) @ theta

    def lambda_at(self, t: float, theta: np.ndarray | None = None) -> float:
        """Step hazard lambda(t; theta)."""
        if theta is None:
            theta = self.theta_worst
        j = np.searchsorted(self.tau, t, side="right") - 1
        j = min(max(j, 0), len(theta) - 1)
        return float(theta[j])

    # ── exact objective / gradient ────────────────────────────────────────────

    def expected_useful_work_term(
        self, a: float, b: float, theta: np.ndarray | None = None
    ) -> float:
        """Compute exp(L_k) * ∫_a^b exp(-Λ_k(u)) du exactly for piecewise-constant hazard."""
        if theta is None:
            theta = self.theta_worst

        pieces: list = []
        L_total = 0.0
        for j in range(len(theta)):
            left  = max(a, self.tau[j])
            right = min(b, self.tau[j + 1])
            if right > left:
                pieces.append((left, right, theta[j]))
                L_total += theta[j] * (right - left)

        val = 0.0
        prefix = 0.0
        for left, right, lam in pieces:
            seg_len = right - left
            if lam < 1e-12:
                val += np.exp(L_total - prefix) * seg_len
            else:
                val += np.exp(L_total - prefix) * (1.0 - np.exp(-lam * seg_len)) / lam
            prefix += lam * seg_len
        return float(val)

    def interval_cost(self, a: float, b: float, qk: float, ck: float = 0.0) -> float:
        theta_star = self.theta_worst
        L = float(sum(
            theta_star[j] * max(0.0, min(b, self.tau[j + 1]) - max(a, self.tau[j]))
            for j in range(len(theta_star))
        ))
        useful_term = self.expected_useful_work_term(a, b, theta_star)
        return useful_term + qk * (np.exp(L) - 1.0) + ck

    def objective_from_delta(self, delta: np.ndarray, **kwargs) -> float:
        T_knots = self.delta_to_knots(delta)
        val = sum(
            self.interval_cost(T_knots[k], T_knots[k + 1], self.q[k])
            for k in range(self.K)
        )
        if self.checkpoint_costs is not None:
            val += float(np.sum(self.checkpoint_costs))
        return float(val)

    def objective_from_internal_knots(self, T_internal: np.ndarray, **kwargs) -> float:
        T_knots = self.internal_knots_to_full(T_internal)
        return self.objective_from_delta(self.full_knots_to_delta(T_knots))

    def gradient_delta(self, delta: np.ndarray, **kwargs) -> np.ndarray:
        """Gradient w.r.t. delta via suffix sum of gradient_internal_knots."""
        T_knots = self.delta_to_knots(delta)
        gT = self.gradient_internal_knots(T_knots[1:-1])
        gDelta = np.zeros(self.K, dtype=float)
        for j in range(self.K - 2, -1, -1):
            gDelta[j] = gT[j] if j == self.K - 2 else gDelta[j + 1] + gT[j]
        return gDelta

    def gradient_internal_knots(self, T_internal: np.ndarray, **kwargs) -> np.ndarray:
        """
        Robust gradient via Danskin's theorem: evaluate at theta* = theta_hat + rho.
        """
        T_knots    = self.internal_knots_to_full(T_internal)
        theta_star = self.theta_worst
        L          = self.cumulative_hazards(T_knots, theta_star)
        grad       = np.zeros(self.K - 1, dtype=float)

        useful_terms = np.array([
            self.expected_useful_work_term(T_knots[k], T_knots[k + 1], theta_star)
            for k in range(self.K)
        ])

        for k in range(1, self.K):
            lam_t = self.lambda_at(T_knots[k], theta_star)
            grad[k - 1] = (
                lam_t * np.exp(L[k - 1]) * (useful_terms[k - 1] + self.q[k - 1])
                + 1.0
                - (1.0 + self.q[k] * lam_t) * np.exp(L[k])
            )
        return grad


# ── RobustPolynomialHazardProblem ─────────────────────────────────────────────

class RobustPolynomialHazardProblem:
    """
    Robust checkpoint scheduling with polynomial hazard rate:

        lambda(t; theta) = sum_{j=0}^{n} theta_j * (t/T)^j,   theta in R^{n+1}

    Uncertainty set: theta in [theta_hat - rho, theta_hat + rho], lambda >= 0.
    Worst case: theta* = max(theta_hat + rho, 0)  (upper corner).

    Delegates objective/gradient to UsefulWorkHazardProblem evaluated at theta*.
    """

    def __init__(
        self,
        total_useful_work: float,
        num_intervals: int,
        epsilon: float,
        theta_hat: np.ndarray,    # shape (n+1,), polynomial coefficients
        rho: np.ndarray | float,  # per-coordinate half-width, shape (n+1,) or scalar
        q: np.ndarray,            # shape (K,)
        checkpoint_costs: np.ndarray | None = None,
    ):
        self.T = float(total_useful_work)
        self.K = int(num_intervals)
        self.epsilon = float(epsilon)
        self.theta_hat = np.asarray(theta_hat, dtype=float)
        _rho = np.asarray(rho, dtype=float)
        self.rho = np.broadcast_to(_rho, self.theta_hat.shape).copy()
        self.q = np.asarray(q, dtype=float)

        theta_star = np.maximum(self.theta_hat + self.rho, 0.0)
        self._worst_problem = UsefulWorkHazardProblem(
            total_useful_work=self.T,
            num_intervals=self.K,
            epsilon=self.epsilon,
            lambda_fn=self._make_lambda(theta_star),
            q=self.q,
            checkpoint_costs=checkpoint_costs,
        )

    def _make_lambda(self, theta: np.ndarray):
        T  = self.T
        js = np.arange(len(theta), dtype=float)

        def lambda_fn(t):
            t = np.asarray(t, dtype=float)
            scalar = t.ndim == 0
            t = np.atleast_1d(t)
            result = (t[:, np.newaxis] / T) ** js @ theta
            return float(result[0]) if scalar else result

        return lambda_fn

    @property
    def num_intervals(self) -> int:
        return self.K

    @property
    def total_useful_work(self) -> float:
        return self.T

    def delta_to_knots(self, delta):
        return self._worst_problem.delta_to_knots(delta)

    def internal_knots_to_full(self, T_internal):
        return self._worst_problem.internal_knots_to_full(T_internal)

    def full_knots_to_delta(self, T_knots):
        return self._worst_problem.full_knots_to_delta(T_knots)

    def objective_from_delta(self, delta, **kwargs):
        return self._worst_problem.objective_from_delta(delta, **kwargs)

    def objective_from_internal_knots(self, T_internal, **kwargs):
        return self._worst_problem.objective_from_internal_knots(T_internal, **kwargs)

    def gradient_internal_knots(self, T_internal, **kwargs):
        return self._worst_problem.gradient_internal_knots(T_internal, **kwargs)

    def gradient_delta(self, delta, **kwargs):
        return self._worst_problem.gradient_delta(delta, **kwargs)


# ── RobustPowerLawHazardProblem ───────────────────────────────────────────────

class RobustPowerLawHazardProblem:
    """
    Robust checkpoint scheduling with power-law hazard rate:

        lambda(t; theta) = theta_0 * (t/T)^{theta_1},
        theta_0 >= 0,  theta_1 > -1.

    Uncertainty set: theta in [theta_hat - rho, theta_hat + rho] intersected
    with the feasibility constraints above.

    Worst case: theta* = (theta_hat_0 + rho_0,  theta_hat_1 - rho_1).
      - Larger theta_0 scales up all rates   → higher objective.
      - Smaller theta_1 flattens the profile → higher cumulative hazard.

    Delegates objective/gradient to UsefulWorkHazardProblem evaluated at theta*.
    """

    def __init__(
        self,
        total_useful_work: float,
        num_intervals: int,
        epsilon: float,
        theta_hat: np.ndarray,  # shape (2,): [scale theta_0, exponent theta_1]
        rho: np.ndarray,        # shape (2,): [rho_0, rho_1]
        q: np.ndarray,          # shape (K,)
        checkpoint_costs: np.ndarray | None = None,
    ):
        self.T = float(total_useful_work)
        self.K = int(num_intervals)
        self.epsilon = float(epsilon)
        self.theta_hat = np.asarray(theta_hat, dtype=float)
        self.rho = np.asarray(rho, dtype=float)
        self.q = np.asarray(q, dtype=float)

        assert len(self.theta_hat) == 2 and len(self.rho) == 2, \
            "theta_hat and rho must each have length 2 for power-law hazard"

        theta0_star = self.theta_hat[0] + self.rho[0]
        theta1_star = max(self.theta_hat[1] - self.rho[1], -1 + 1e-9)
        self.theta_star = np.array([theta0_star, theta1_star])

        self._worst_problem = UsefulWorkHazardProblem(
            total_useful_work=self.T,
            num_intervals=self.K,
            epsilon=self.epsilon,
            lambda_fn=self._make_lambda(self.theta_star),
            q=self.q,
            checkpoint_costs=checkpoint_costs,
        )

    def _make_lambda(self, theta: np.ndarray):
        T              = self.T
        theta0, theta1 = float(theta[0]), float(theta[1])

        def lambda_fn(t):
            t = np.asarray(t, dtype=float)
            scalar = t.ndim == 0
            t = np.atleast_1d(np.maximum(t, 1e-12))
            result = theta0 * (t / T) ** theta1
            return float(result[0]) if scalar else result

        return lambda_fn

    @property
    def num_intervals(self) -> int:
        return self.K

    @property
    def total_useful_work(self) -> float:
        return self.T

    def delta_to_knots(self, delta):
        return self._worst_problem.delta_to_knots(delta)

    def internal_knots_to_full(self, T_internal):
        return self._worst_problem.internal_knots_to_full(T_internal)

    def full_knots_to_delta(self, T_knots):
        return self._worst_problem.full_knots_to_delta(T_knots)

    def objective_from_delta(self, delta, **kwargs):
        return self._worst_problem.objective_from_delta(delta, **kwargs)

    def objective_from_internal_knots(self, T_internal, **kwargs):
        return self._worst_problem.objective_from_internal_knots(T_internal, **kwargs)

    def gradient_internal_knots(self, T_internal, **kwargs):
        return self._worst_problem.gradient_internal_knots(T_internal, **kwargs)

    def gradient_delta(self, delta, **kwargs):
        return self._worst_problem.gradient_delta(delta, **kwargs)
