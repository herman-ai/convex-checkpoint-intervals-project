from typing import Dict, List

import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

from utils.hazard_functions import (
    lambda_fn_step,
    lambda_fn_linear,
    lambda_fn_convex,
    lambda_fn_three_phase,
)
from utils.helpers import SECONDS_PER_HOUR, UsefulWorkHazardProblem
from simulator_v2 import (
    monte_carlo_schedule_useful_work_hazard,
    simulate_schedule_useful_work_hazard,
)

import numpy as np


class RobustStepHazardProblem:
    def __init__(
        self,
        total_useful_work: float,
        num_intervals: int,
        epsilon: float,
        tau: np.ndarray,          # shape (n+1,), includes 0 and T
        theta_hat: np.ndarray,    # shape (n,)
        rho: np.ndarray | float,  # per-coordinate half-width (additive), shape (n,) or scalar
        q: np.ndarray,            # shape (K,)
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
        self.checkpoint_costs = None if checkpoint_costs is None else np.asarray(checkpoint_costs, dtype=float)

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

    # ── aliases so generic optimizers (PGD, MD) can use this problem ──────
    @property
    def num_intervals(self) -> int:
        return self.K

    @property
    def total_useful_work(self) -> float:
        return self.T

    @property
    def theta_worst(self) -> np.ndarray:
        # Worst-case corner of the additive box Θ = {θ : |θ - θ̂| ≤ ρ, θ ≥ 0}
        return np.maximum(self.theta_hat + self.rho, 0.0)

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

    def overlap_lengths(self, T_knots: np.ndarray) -> np.ndarray:
        """
        s_k^j = length of interval k lying in phase j.
        Returns array shape (K, n_phases).
        """
        K = self.K
        n = len(self.theta_hat)
        s = np.zeros((K, n), dtype=float)

        for k in range(K):
            a = T_knots[k]
            b = T_knots[k + 1]
            for j in range(n):
                left = max(a, self.tau[j])
                right = min(b, self.tau[j + 1])
                s[k, j] = max(0.0, right - left)
        return s

    def cumulative_hazards(self, T_knots: np.ndarray, theta: np.ndarray | None = None) -> np.ndarray:
        """
        L_k(theta) = sum_j theta_j s_k^j
        """
        if theta is None:
            theta = self.theta_worst
        s = self.overlap_lengths(T_knots)
        return s @ theta

    def objective_from_delta(self, delta: np.ndarray, **kwargs) -> float:
        """
        Exact robust objective using expected_useful_work_term for each interval.
        The single-phase shortcut (delta/L + q)*(exp(L)-1) is only correct when
        lambda is constant over the whole interval; this handles phase crossings.
        """
        T_knots = self.delta_to_knots(delta)
        val = 0.0
        for k in range(self.K):
            val += self.interval_cost(T_knots[k], T_knots[k + 1], self.q[k])
        if self.checkpoint_costs is not None:
            val += float(np.sum(self.checkpoint_costs))
        return val

    def objective_from_internal_knots(self, T_internal: np.ndarray, **kwargs) -> float:
        T_knots = self.internal_knots_to_full(T_internal)
        delta = self.full_knots_to_delta(T_knots)
        return self.objective_from_delta(delta)

    def lambda_at(self, t: float, theta: np.ndarray | None = None) -> float:
        """
        Step hazard lambda(t; theta)
        """
        if theta is None:
            theta = self.theta_worst
        j = np.searchsorted(self.tau, t, side="right") - 1
        j = min(max(j, 0), len(theta) - 1)
        return float(theta[j])

    def expected_useful_work_term(self, a: float, b: float, theta: np.ndarray | None = None) -> float:
        """
        Computes exp(L_k) * ∫_a^b exp(-Lambda_k(u)) du exactly for piecewise-constant hazard.
        """
        if theta is None:
            theta = self.theta_worst

        val = 0.0
        L_total = 0.0
        pieces = []

        # Split interval [a,b] across phases
        for j in range(len(theta)):
            left = max(a, self.tau[j])
            right = min(b, self.tau[j + 1])
            if right > left:
                pieces.append((left, right, theta[j]))
                L_total += theta[j] * (right - left)

        prefix = 0.0
        for left, right, lam in pieces:
            seg_len = right - left
            if lam < 1e-12:
                contrib = np.exp(L_total - prefix) * seg_len
            else:
                contrib = np.exp(L_total - prefix) * (1.0 - np.exp(-lam * seg_len)) / lam
            val += contrib
            prefix += lam * seg_len

        return float(val)

    def interval_cost(self, a: float, b: float, qk: float, ck: float = 0.0) -> float:
        theta_star = self.theta_worst
        # L_k = integral of lambda over [a,b]; compute directly (not via cumulative_hazards
        # which expects the full K+1 knot array, not a 2-element array)
        L = float(sum(
            theta_star[j] * max(0.0, min(b, self.tau[j + 1]) - max(a, self.tau[j]))
            for j in range(len(theta_star))
        ))
        useful_term = self.expected_useful_work_term(a, b, theta_star)
        return useful_term + qk * (np.exp(L) - 1.0) + ck

    def gradient_delta(self, delta: np.ndarray, **kwargs) -> np.ndarray:
        """
        Gradient in the Delta-parameterization via suffix sum of gradient_internal_knots.
        """
        T_knots = self.delta_to_knots(delta)
        T_internal = T_knots[1:-1]
        gT = self.gradient_internal_knots(T_internal)
        gDelta = np.zeros(self.K, dtype=float)
        for j in range(self.K - 2, -1, -1):
            gDelta[j] = (gT[j] if j == self.K - 2 else gDelta[j + 1] + gT[j])
        return gDelta

    def gradient_internal_knots(self, T_internal: np.ndarray, **kwargs) -> np.ndarray:
        """
        Robust gradient via Danskin: evaluate gradient at theta* = (1+rho)*theta_hat.
        By Danskin's theorem, the subgradient of max_theta f(Delta, theta) w.r.t.
        Delta equals the gradient of f at the maximising theta*.
        """
        # **kwargs absorbs num_steps from generic callers (not used; exact formula)
        T_knots = self.internal_knots_to_full(T_internal)
        K = self.K
        theta_star = self.theta_worst

        L = self.cumulative_hazards(T_knots, theta_star)
        grad = np.zeros(K - 1, dtype=float)

        # precompute interval integrals
        useful_terms = np.zeros(K, dtype=float)
        for k in range(K):
            useful_terms[k] = self.expected_useful_work_term(T_knots[k], T_knots[k + 1], theta_star)

        for k in range(1, K):
            t = T_knots[k]
            lam_t = self.lambda_at(t, theta_star)

            grad[k - 1] = (
                lam_t * np.exp(L[k - 1]) * (useful_terms[k - 1] + self.q[k - 1])
                + 1.0
                - (1.0 + self.q[k] * lam_t) * np.exp(L[k])
            )

        return grad

def make_step_lambda_fn(tau: np.ndarray, theta: np.ndarray):
    tau = np.asarray(tau, dtype=float)
    theta = np.asarray(theta, dtype=float)

    def lambda_fn(t: float) -> float:
        j = np.searchsorted(tau, t, side="right") - 1
        j = min(max(j, 0), len(theta) - 1)
        return float(theta[j])

    return lambda_fn

##############################################################################

def project_internal_knots_cvxpy(v: np.ndarray, total_useful_work: float, epsilon: float) -> np.ndarray:
    """
    Project v onto:
        0 <= T_1 <= ... <= T_{K-1} <= T
        with gaps T_k - T_{k-1} >= epsilon and T - T_{K-1} >= epsilon
    """
    import cvxpy as cp

    n = len(v)
    x = cp.Variable(n)

    constraints = []
    prev = 0.0
    for i in range(n):
        constraints.append(x[i] - prev >= epsilon)
        prev = x[i]
    constraints.append(total_useful_work - prev >= epsilon)

    problem = cp.Problem(cp.Minimize(cp.sum_squares(x - v)), constraints)
    problem.solve(solver=cp.SCS, verbose=False)

    if x.value is None:
        raise RuntimeError("Projection failed")
    return np.array(x.value).ravel()


class RobustPolynomialHazardProblem:
    """
    Robust checkpoint scheduling with polynomial hazard rate:

        lambda(t; theta) = sum_{j=0}^{n} theta_j * (t/T)^j,   theta in R^{n+1}

    Uncertainty set: theta in [theta_hat - rho, theta_hat + rho], lambda >= 0.
    Worst case: theta* = max(theta_hat + rho, 0) (upper corner), because each
    monomial (t/T)^j >= 0 so f is increasing in every theta_j.

    Delegates objective/gradient to UsefulWorkHazardProblem evaluated at theta*.
    """

    def __init__(
        self,
        total_useful_work: float,
        num_intervals: int,
        epsilon: float,
        theta_hat: np.ndarray,   # shape (n+1,), polynomial coefficients
        rho: np.ndarray | float, # per-coordinate half-width, shape (n+1,) or scalar
        q: np.ndarray,           # shape (K,)
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
        T = self.T
        js = np.arange(len(theta), dtype=float)
        def lambda_fn(t):
            # Accepts both scalars and numpy arrays
            t = np.asarray(t, dtype=float)
            scalar = t.ndim == 0
            t = np.atleast_1d(t)
            result = (t[:, np.newaxis] / T) ** js @ theta   # shape (len(t),)
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


class RobustPowerLawHazardProblem:
    """
    Robust checkpoint scheduling with power-law hazard rate:

        lambda(t; theta) = theta_0 * (t/T)^{theta_1},
        theta_0 >= 0,  theta_1 > -1.

    Uncertainty set: theta in [theta_hat - rho, theta_hat + rho] intersected
    with the feasibility constraints above.

    Worst case: theta* = (theta_hat_0 + rho_0,  theta_hat_1 - rho_1).
      - Larger theta_0 scales up all rates   → higher objective.
      - Smaller theta_1 flattens the profile → total cumulative hazard
        integral(0,T) = theta_0*T/(theta_1+1) is larger → higher objective.

    Delegates objective/gradient to UsefulWorkHazardProblem evaluated at theta*.
    """

    def __init__(
        self,
        total_useful_work: float,
        num_intervals: int,
        epsilon: float,
        theta_hat: np.ndarray,   # shape (2,): [scale theta_0, exponent theta_1]
        rho: np.ndarray,         # shape (2,): [rho_0, rho_1]
        q: np.ndarray,           # shape (K,)
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

        theta0_star = self.theta_hat[0] + self.rho[0]                  # scale: upper bound
        theta1_star = max(self.theta_hat[1] - self.rho[1], -1 + 1e-9)  # exponent: lower bound, clamped > -1
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
        T = self.T
        theta0, theta1 = float(theta[0]), float(theta[1])
        def lambda_fn(t):
            # Accepts both scalars and numpy arrays
            t = np.asarray(t, dtype=float)
            scalar = t.ndim == 0
            t = np.atleast_1d(np.maximum(t, 1e-12))  # avoid singularity for theta1 < 0
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


def project_internal_knots_pav(v: np.ndarray, total_useful_work: float, epsilon: float) -> np.ndarray:
    """
    Project candidate internal knots v = [T_1, ..., T_{K-1}] onto the feasible set

        0 = T_0 <= T_1 <= ... <= T_{K-1} <= T_K = total_useful_work
        T_k - T_{k-1} >= epsilon  for all k

    using the Pool-Adjacent Violators (PAV) algorithm.

    Steps:
      1. Shift S_hat[k] = T_k - k*epsilon, with fixed endpoints S_hat[0]=0,
         S_hat[K] = total_useful_work - K*epsilon.  The constraint becomes S non-decreasing.
      2. Run PAV (isotonic regression) with the fixed endpoints anchored by
         infinite weight so they are never displaced.
      3. Recover T_k* = S_k* + k*epsilon and return the internal knots T_1*,...,T_{K-1}*.
    """
    n = len(v)       # number of internal knots = K - 1
    K = n + 1        # number of intervals
    T_tilde = total_useful_work - K * epsilon

    if T_tilde < 0:
        raise ValueError("total_useful_work < K * epsilon: problem is infeasible")

    # Build full S_hat sequence (length K+1) including fixed endpoints
    S_hat = np.empty(K + 1, dtype=float)
    S_hat[0] = 0.0
    for k in range(1, K):
        S_hat[k] = v[k - 1] - k * epsilon
    S_hat[K] = T_tilde

    # PAV on S_hat with fixed endpoints (infinite weight)
    # Each block: {'mean': float, 'weight': float, 'start': int, 'end': int}
    INF = float("inf")
    blocks = [
        {"mean": float(S_hat[k]), "weight": INF if (k == 0 or k == K) else 1.0,
         "start": k, "end": k}
        for k in range(K + 1)
    ]

    i = 1
    while i < len(blocks):
        if blocks[i]["mean"] < blocks[i - 1]["mean"]:
            mu_a, w_a = blocks[i - 1]["mean"], blocks[i - 1]["weight"]
            mu_b, w_b = blocks[i]["mean"],     blocks[i]["weight"]

            if w_a == INF:
                new_mu, new_w = mu_a, INF
            elif w_b == INF:
                new_mu, new_w = mu_b, INF
            else:
                new_w  = w_a + w_b
                new_mu = (w_a * mu_a + w_b * mu_b) / new_w

            blocks[i - 1] = {"mean": new_mu, "weight": new_w,
                              "start": blocks[i - 1]["start"], "end": blocks[i]["end"]}
            blocks.pop(i)
            i = max(1, i - 1) # The next iteration will check the new block against its predecessor, so move back if possible
        else:
            i += 1

    # Expand blocks back to per-index values
    S_star = np.empty(K + 1, dtype=float)
    for block in blocks:
        S_star[block["start"] : block["end"] + 1] = block["mean"]

    # Recover T* = S* + k*epsilon; return only internal knots
    T_star = S_star + epsilon * np.arange(K + 1)
    return T_star[1:-1]



def optimize_pgd_internal_knots(
    problem: UsefulWorkHazardProblem,
    max_iters: int = 500,
    step_size: float = 1e3,
    num_steps: int = 256,
    init_delta: np.ndarray | None = None,
) -> Dict[str, object]:
    """
    PGD in the T-parameterization with Armijo backtracking line search.

    Important: because each iterate is projected back onto the feasible set,
    the actual step is d = T_new - T_internal, not -alpha * grad. Therefore the
    sufficient-decrease test must use the projected Armijo condition

        f(T + d) <= f(T) + sigma * grad^T d,

    rather than the unconstrained rule based on alpha * ||grad||^2.
    """
    K = problem.num_intervals
    if init_delta is None:
        delta0 = np.full(K, problem.total_useful_work / K)
    else:
        delta0 = np.asarray(init_delta, dtype=float).copy()
    T_knots = problem.delta_to_knots(delta0)
    T_internal = T_knots[1:-1].copy()

    history = []
    sigma = 1e-4

    for it in range(max_iters):
        grad = problem.gradient_internal_knots(T_internal, num_steps=num_steps)
        obj_curr = problem.objective_from_internal_knots(T_internal, num_steps=num_steps)

        alpha = step_size
        T_new = T_internal.copy()
        obj_new = obj_curr

        for _ in range(30):
            candidate = T_internal - alpha * grad
            T_try = project_internal_knots_pav(
                candidate,
                total_useful_work=problem.total_useful_work,
                epsilon=problem.epsilon,
            )
            d = T_try - T_internal
            obj_try = problem.objective_from_internal_knots(T_try, num_steps=num_steps)

            if obj_try <= obj_curr + sigma * float(np.dot(grad, d)):
                T_new = T_try
                obj_new = obj_try
                break

            alpha *= 0.5

        update_norm = np.linalg.norm(T_new - T_internal)
        history.append({"iter": it, "objective": obj_new, "update_norm": update_norm})
        T_internal = T_new
        if update_norm < 1e-6:
            break

    T_full = problem.internal_knots_to_full(T_internal)
    delta = problem.full_knots_to_delta(T_full)

    return {
        "T_internal": T_internal,
        "delta": delta,
        "objective": problem.objective_from_delta(delta, num_steps=num_steps),
        "history": history,
    }

def optimize_mirror_descent(
    problem: UsefulWorkHazardProblem,
    max_iters: int = 1000,
    step_size: float = 1e-2,
    num_steps: int = 256,
    init_delta: np.ndarray | None = None,
) -> Dict[str, object]:
    """
    Mirror descent with negative-entropy mirror map (exponentiated gradient)
    in the shifted simplex coordinates Delta_tilde_k = Delta_k - epsilon.

    Update rule:
        Delta_tilde^{n+1}_k = T_tilde * Delta_tilde^n_k * exp(-alpha * g_k)
                               / sum_j Delta_tilde^n_j * exp(-alpha * g_j)

    where g = grad_Delta f and T_tilde = T - K * epsilon.
    By construction every iterate satisfies Delta_tilde >= 0 and sum = T_tilde.

    Step-size note: grad_delta is dimensionless (seconds/seconds) with
    scale that depends on the hazard family and robustness setting. In the
    experiments below we therefore auto-scale the initial MD step size from
    the infinity norm of the initial gradient, rather than using one fixed
    value for all problems.
    """
    K = problem.num_intervals
    T_tilde = problem.total_useful_work - K * problem.epsilon

    # Initialize from provided delta or default to equal spacing
    if init_delta is None:
        delta_tilde = np.full(K, T_tilde / K, dtype=float)
    else:
        d0 = np.asarray(init_delta, dtype=float)
        delta_tilde = np.maximum(d0 - problem.epsilon, 1e-10)
        delta_tilde *= T_tilde / delta_tilde.sum()

    # Best-iterate tracking: return the best delta seen across all iterates.
    # Combined with the diminishing step schedule this guarantees convergence
    # even for non-smooth objectives (e.g. step hazard).
    # Seed with the initial (equal) schedule so we never return something worse
    # than the starting point (important when equal is near-optimal).
    delta_init_full  = delta_tilde + problem.epsilon
    best_obj         = problem.objective_from_delta(delta_init_full, num_steps=num_steps)
    best_delta_tilde = delta_tilde.copy()

    history = []
    for it in range(max_iters):
        delta = delta_tilde + problem.epsilon
        grad = problem.gradient_delta(delta, num_steps=num_steps)

        # Diminishing step schedule: alpha_t = step_size / sqrt(t+1).
        # This is the standard subgradient schedule for non-smooth objectives
        # (e.g. step hazard) where a fixed step would oscillate indefinitely
        # around kinks. Best-iterate tracking ensures we keep the best seen so far.
        alpha = step_size / np.sqrt(it + 1)

        # Exponentiated gradient update (log-sum-exp for numerical stability)
        log_weights = np.log(delta_tilde) - alpha * grad
        log_weights -= log_weights.max()              # shift for stability
        weights = np.exp(log_weights)
        delta_tilde_new = T_tilde * weights / weights.sum()

        update_norm = np.linalg.norm(delta_tilde_new - delta_tilde)

        # Evaluate at the new iterate and track the best seen so far
        delta_new = delta_tilde_new + problem.epsilon
        obj_new = problem.objective_from_delta(delta_new, num_steps=num_steps)
        if obj_new < best_obj:
            best_obj         = obj_new
            best_delta_tilde = delta_tilde_new.copy()

        # Record running-minimum so the convergence curve is monotone
        history.append({"iter": it, "objective": best_obj, "update_norm": update_norm})

        delta_tilde = delta_tilde_new
        if update_norm < 1e-6:
            break

    delta = best_delta_tilde + problem.epsilon
    return {
        "delta": delta,
        "objective": best_obj,
        "history": history,
    }



def _diverse_inits(K: int, T: float, epsilon: float) -> list:
    """
    Three deterministic starting schedules for multi-start optimization:
      - equal:        T/K per interval
      - back-loaded:  linearly increasing intervals (weights 1, 2, ..., K)
      - front-loaded: linearly decreasing intervals (weights K, K-1, ..., 1)

    Back-loaded is a useful warm start when later intervals are safer (e.g.
    step hazard with early rate >> late rate); front-loaded covers the
    opposite case.  Equal is always included as the baseline.
    """
    T_tilde = T - K * epsilon
    equal = np.full(K, T / K)
    w_up = np.arange(1, K + 1, dtype=float)
    back  = epsilon + T_tilde * w_up / w_up.sum()
    w_dn  = np.arange(K, 0, -1, dtype=float)
    front = epsilon + T_tilde * w_dn / w_dn.sum()
    return [equal, back, front]


def _best_pgd(problem, **kwargs) -> Dict[str, object]:
    """PGD from three diverse initializations; returns the best result."""
    inits = _diverse_inits(problem.num_intervals, problem.total_useful_work, problem.epsilon)
    results = [optimize_pgd_internal_knots(problem, init_delta=d, **kwargs) for d in inits]
    return min(results, key=lambda r: r["objective"])


def _best_md(problem, **kwargs) -> Dict[str, object]:
    """Mirror descent from three diverse initializations; returns the best result."""
    inits = _diverse_inits(problem.num_intervals, problem.total_useful_work, problem.epsilon)
    results = [optimize_mirror_descent(problem, init_delta=d, **kwargs) for d in inits]
    return min(results, key=lambda r: r["objective"])


# ── ADMM helpers ─────────────────────────────────────────────────────────────

def _interval_h_grad_and_hess(problem, k: int, a: float, b: float, num_steps: int = 64) -> tuple:
    """
    Returns (dh/da, dh/db, d²h/da², d²h/db²) for interval k with endpoints a, b.

    Exact analytic formulas (same for all three hazard types):
        dh/da   = -exp(L) * (1 + q * lambda(a))
        dh/db   =  lambda(b) * (ut + q * exp(L)) + 1
        d²h/da² =  lambda(a) * exp(L) * (1 + q * lambda(a))
        d²h/db² =  lambda(b)^2 * (ut + q * exp(L)) + lambda(b)

    where  L = ∫_a^b lambda,  ut = exp(L) * ∫_a^b exp(-Λ(u)) du.
    For RobustStepHazardProblem these are exact; for _worst_problem, L and ut
    are obtained via numerical quadrature.
    """
    from utils.helpers import integrate_lambda, interval_work_integral
    q_k = float(problem.q[k])

    if isinstance(problem, RobustStepHazardProblem):
        theta = problem.theta_worst
        L = float(sum(
            theta[j] * max(0.0, min(b, problem.tau[j + 1]) - max(a, problem.tau[j]))
            for j in range(len(theta))
        ))
        exp_L = np.exp(L)
        ut = problem.expected_useful_work_term(a, b, theta)
        lam_a = problem.lambda_at(a, theta)
        lam_b = problem.lambda_at(b, theta)
    elif hasattr(problem, "_worst_problem"):
        wp = problem._worst_problem
        L = integrate_lambda(wp.lambda_fn, a, b, num_steps=num_steps)
        exp_L = np.exp(L)
        I = interval_work_integral(wp.lambda_fn, a, b, num_steps=num_steps)
        ut = exp_L * I
        lam_a = wp.lambda_fn(a)
        lam_b = wp.lambda_fn(b)
    else:
        raise NotImplementedError(f"_interval_h_grad_and_hess: unsupported type {type(problem)}")

    dh_da   = -exp_L * (1.0 + q_k * lam_a)
    dh_db   = lam_b * (ut + q_k * exp_L) + 1.0
    d2h_da2 = lam_a * exp_L * (1.0 + q_k * lam_a)
    d2h_db2 = lam_b ** 2 * (ut + q_k * exp_L) + lam_b
    return float(dh_da), float(dh_db), float(d2h_da2), float(d2h_db2)


def optimize_admm(
    problem,
    max_iters: int = 300,
    rho: float = 1.0,
    inner_iters: int = 3,
    num_steps: int = 64,
    tol: float = 1e-6,
    init_delta: np.ndarray | None = None,
) -> Dict[str, object]:
    """
    ADMM for checkpoint scheduling with hazard rate uncertainty.

    Variables:
        x[k] = (x_k^-, x_k^+) ∈ R²  — local interval endpoints  (shape K×2)
        z[m] ∈ R                      — global consensus checkpoints (shape K+1)
    Constraint:  x_k^- = z[k], x_k^+ = z[k+1]  (i.e. x = Mz, M = [e_k, e_{k+1}]^T per row)

    Augmented Lagrangian:
        L_ρ(x,z,y) = Σ_k h(x_k; θ*) + y^T(x-Mz) + ρ/2 ‖x-Mz‖²

    x-step: K independent 2-D subproblems solved via diagonal Newton steps:
            Δa = -∇F/da / (d²h/da² + ρ),  Δb = -∇F/db / (d²h/db² + ρ).
            This is the exact curvature-scaled GD step; inner_iters=2-3 suffices.
    z-step: closed-form unconstrained minimiser then PAV projection onto T.
            v[m] = (x[m-1,1] + x[m,0])/2 + (y[m-1,1] + y[m,0])/(2ρ)  for m=1..K-1
    Dual:   y ← y + ρ(x - Mz).

    Convergence note: ρ should be set to 1/(2 * pgd_step_size) via _auto_admm_rho
    so that the ADMM z-updates have effective step size comparable to PGD.
    """
    K = problem.num_intervals
    T = problem.total_useful_work
    epsilon = problem.epsilon

    # ── Initialise ────────────────────────────────────────────────────────────
    if init_delta is None:
        delta0 = np.full(K, T / K)
    else:
        delta0 = np.asarray(init_delta, dtype=float).copy()
    z = problem.delta_to_knots(delta0).copy()      # shape (K+1,)
    x = np.stack([z[:-1], z[1:]], axis=1).copy()  # shape (K, 2)
    y = np.zeros((K, 2), dtype=float)             # dual variables, shape (K, 2)

    history: List[Dict] = []

    for it in range(max_iters):
        Mz = np.stack([z[:-1], z[1:]], axis=1)   # current Mz target, shape (K, 2)

        # ── x-step: K independent 2-D Newton subproblems ─────────────────────
        # Newton step: Δ = -∇F / (∇²h + ρ)  — one step converges for near-quadratic h
        for k in range(K):
            v_a, v_b = Mz[k, 0], Mz[k, 1]
            y_a, y_b = y[k, 0], y[k, 1]
            a, b = x[k, 0], x[k, 1]
            for _ in range(inner_iters):
                b = max(b, a + 1e-6)   # keep interval positive-length
                dh_da, dh_db, d2h_da2, d2h_db2 = _interval_h_grad_and_hess(
                    problem, k, a, b, num_steps)
                # Newton step with exact diagonal curvature: 1/(d²h + ρ)
                H_a = d2h_da2 + rho
                H_b = d2h_db2 + rho
                a -= (dh_da + y_a + rho * (a - v_a)) / H_a
                b -= (dh_db + y_b + rho * (b - v_b)) / H_b
            x[k, 0] = a
            x[k, 1] = max(b, a + 1e-6)

        # ── z-step: unconstrained minimiser → PAV projection ─────────────────
        # Interior z[m], m=1..K-1:  d/dz[m] L_ρ = 0  →  v[m-1] below
        v_int = (x[:-1, 1] + x[1:, 0]) / 2.0 + (y[:-1, 1] + y[1:, 0]) / (2.0 * rho)
        z[1:-1] = project_internal_knots_pav(v_int, T, epsilon)

        # ── Dual update ───────────────────────────────────────────────────────
        Mz_new = np.stack([z[:-1], z[1:]], axis=1)
        residual = x - Mz_new
        y += rho * residual

        # ── Diagnostics ───────────────────────────────────────────────────────
        primal_res = float(np.linalg.norm(residual))
        delta = np.diff(z)
        obj = problem.objective_from_delta(delta, num_steps=num_steps)
        history.append({"iter": it, "objective": obj, "primal_res": primal_res})

        if primal_res < tol:
            break

    delta = np.diff(z)
    return {
        "delta":     delta,
        "z":         z,
        "objective": problem.objective_from_delta(delta),
        "history":   history,
    }


def _best_admm(problem, **kwargs) -> Dict[str, object]:
    """ADMM from three diverse initializations; returns the best result."""
    inits = _diverse_inits(problem.num_intervals, problem.total_useful_work, problem.epsilon)
    results = [optimize_admm(problem, init_delta=d, **kwargs) for d in inits]
    return min(results, key=lambda r: r["objective"])


def _auto_pgd_step(prob, target_frac=0.05) -> float:
    """
    Set PGD step size so the first gradient step moves ~target_frac * (T - K*eps) / K.
    Scale-independent: problems with larger gradients get smaller steps.
    """
    delta0     = np.full(prob.num_intervals, prob.total_useful_work / prob.num_intervals)
    T_internal = prob.delta_to_knots(delta0)[1:-1]
    grad       = prob.gradient_internal_knots(T_internal)
    g_inf      = np.max(np.abs(grad))
    if g_inf < 1e-15:
        return 1e3
    T_tilde = prob.total_useful_work - prob.num_intervals * prob.epsilon
    target  = target_frac * T_tilde / prob.num_intervals
    return target / g_inf


def _auto_md_step(prob, target_logit=0.05, max_alpha=5.0) -> float:
    """
    Set the initial MD step size (alpha_0) so the first EG update shifts
    log-weights by about `target_logit` nats for the highest-gradient component.

    The EG update is: log(w_k^new) = log(w_k^old) - alpha * g_k + const,
    so alpha * g_inf is the log-weight shift of the steepest component, i.e.
    a multiplicative change of exp(-alpha * g_inf) in that component's weight.
    target_logit=0.05 → ~5% fractional weight change per step, matching
    _auto_pgd_step's target_frac=0.05 in Euclidean space.  Larger values
    (e.g. 0.5) cause EG to overshoot near-optimal starting points because a
    0.5-nat shift = exp(-0.5) ≈ 0.6× change in one step.
    Capped at `max_alpha` to prevent huge steps when gradient is near-zero.
    """
    delta0 = np.full(prob.num_intervals, prob.total_useful_work / prob.num_intervals)
    grad   = prob.gradient_delta(delta0)
    g_inf  = np.max(np.abs(grad))
    if g_inf < 1e-15:
        return max_alpha
    return min(target_logit / g_inf, max_alpha)


def _auto_admm_rho(prob) -> float:
    """
    Set ADMM penalty ρ so that the effective ADMM z-update step size matches
    the PGD step size:  ρ = 1 / (2 * pgd_step).

    Analysis: with Newton x-step, each ADMM outer iteration moves z by
    ≈ (df/dz) / (2*(d²h + ρ)).  Setting ρ << d²h gives step ≈ 1/(2*d²h)
    (Newton scale); setting ρ >> d²h gives step ≈ 1/(2*ρ).  By choosing
    ρ = 1/(2*pgd_step) we ensure the z-updates are on the same scale as PGD.
    """
    pgd_step = _auto_pgd_step(prob)
    return 1.0 / (2.0 * pgd_step)


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
    MC where theta is drawn uniformly from the box [theta_hat - rho, theta_hat + rho]
    (clipped to non-negative) independently for each trial.
    """
    rng = np.random.default_rng(base_seed)
    theta_lo = np.maximum(theta_hat - rho, 0.0)
    theta_hi = theta_hat + rho
    runtimes = []
    failures = []
    for i in range(num_trials):
        theta_sample = rng.uniform(theta_lo, theta_hi)
        lambda_fn = make_step_lambda_fn(tau, theta_sample)
        result = simulate_schedule_useful_work_hazard(
            delta=delta, lambda_fn=lambda_fn, q=q, seed=base_seed + i)
        runtimes.append(result["total_wall_clock"])
        failures.append(result["total_failures"])
    mean_rt = float(np.mean(runtimes))
    return {
        "mean_runtime":   mean_rt,
        "stderr_runtime": float(np.std(runtimes, ddof=1) / np.sqrt(num_trials)),
        "mean_failures":  float(np.mean(failures)),
    }


if __name__ == "__main__":
    import os
    os.makedirs("figures", exist_ok=True)

    RUN_MC  = False   # set True to run Monte Carlo (slow)

    K       = 8
    T       = 48 * 3600.0
    epsilon = 0.5 * 3600.0
    q = np.array([5*60, 5*60, 10*60, 10*60, 15*60, 15*60, 20*60, 20*60], dtype=float)
    equal_delta = np.full(K, T / K)
    N_MC = 100  # MC trials per (schedule, lambda) pair

    def run_comparison(label, nominal_prob, robust_prob, lambda_nom, lambda_wc):
        """Run PGD, MD, and ADMM on both problems; print full table + MC; return results dict."""
        sep = "─" * 65
        print(f"\n{sep}\n  {label}\n{sep}")

        nom_pgd  = _best_pgd( nominal_prob, max_iters=1000, step_size=_auto_pgd_step(nominal_prob))
        nom_md   = _best_md(  nominal_prob, max_iters=2000, step_size=_auto_md_step(nominal_prob))
        nom_admm = _best_admm(nominal_prob, max_iters=500,  rho=_auto_admm_rho(nominal_prob))
        rob_pgd  = _best_pgd( robust_prob,  max_iters=1000, step_size=_auto_pgd_step(robust_prob))
        rob_md   = _best_md(  robust_prob,  max_iters=2000, step_size=_auto_md_step(robust_prob))
        rob_admm = _best_admm(robust_prob,  max_iters=500,  rho=_auto_admm_rho(robust_prob))

        sched_keys = ["equal", "nom_pgd", "nom_md", "nom_admm", "rob_pgd", "rob_md", "rob_admm"]
        sched_lbl  = {
            "equal"   : "equal       ",
            "nom_pgd" : "nominal PGD ",
            "nom_md"  : "nominal MD  ",
            "nom_admm": "nominal ADMM",
            "rob_pgd" : "robust  PGD ",
            "rob_md"  : "robust  MD  ",
            "rob_admm": "robust  ADMM",
        }
        deltas = {
            "equal"   : equal_delta,
            "nom_pgd" : nom_pgd["delta"],
            "nom_md"  : nom_md["delta"],
            "nom_admm": nom_admm["delta"],
            "rob_pgd" : rob_pgd["delta"],
            "rob_md"  : rob_md["delta"],
            "rob_admm": rob_admm["delta"],
        }

        print("\n  Schedules (h):")
        for k in sched_keys:
            print(f"    {sched_lbl[k]}: {np.round(deltas[k] / 3600, 2)}")

        # ── Analytic objectives ──────────────────────────────────────────────
        print(f"\n  {'':24s}  {'at theta_hat':>14s}  {'at theta*':>12s}")
        analytic = {}
        for k in sched_keys:
            at_nom = nominal_prob.objective_from_delta(deltas[k])
            at_wc  = robust_prob.objective_from_delta(deltas[k])
            analytic[k] = (at_nom, at_wc)
            print(f"  {sched_lbl[k]}           {at_nom:>14.1f}s  {at_wc:>12.1f}s")

        for opt in ("pgd", "md", "admm"):
            nom_nom, nom_wc = analytic[f"nom_{opt}"]
            rob_nom, rob_wc = analytic[f"rob_{opt}"]
            price  = rob_nom - nom_nom
            saving = nom_wc  - rob_wc
            print(f"\n  [{opt.upper():4s}]  Price of robustness: +{price:.1f}s (+{100*price/nom_nom:.2f}%)"
                  f"   Worst-case savings: -{saving:.1f}s (-{100*saving/nom_wc:.2f}%)")

        # ── Monte Carlo ──────────────────────────────────────────────────────
        mc = {}
        if RUN_MC:
            print(f"\n  Monte Carlo ({N_MC} trials, nominal lambda / worst-case lambda):")
            for k in sched_keys:
                r_n = monte_carlo_schedule_useful_work_hazard(
                    delta=deltas[k], lambda_fn=lambda_nom, q=q, num_trials=N_MC)
                r_w = monte_carlo_schedule_useful_work_hazard(
                    delta=deltas[k], lambda_fn=lambda_wc,  q=q, num_trials=N_MC)
                mc[k] = (r_n, r_w)
                print(f"    {sched_lbl[k]}:  nom {r_n['mean_runtime']:.0f}s (\u00b1{r_n['stderr_runtime']:.0f})"
                      f"   wc {r_w['mean_runtime']:.0f}s (\u00b1{r_w['stderr_runtime']:.0f})")

        return {
            "label":        label,
            "nom_pgd":      nom_pgd,
            "nom_md":       nom_md,
            "nom_admm":     nom_admm,
            "rob_pgd":      rob_pgd,
            "rob_md":       rob_md,
            "rob_admm":     rob_admm,
            "deltas":       deltas,
            "analytic":     analytic,
            "mc":           mc,
            "nominal_prob": nominal_prob,
            "robust_prob":  robust_prob,
        }

    # ── 1. Step hazard (3-phase, ordering reversal) ──────────────────────────
    # At theta_hat phase ordering (safest→riskiest): 2 < 1 < 3
    #   theta_hat = [0.10, 0.05, 0.20] /h  → nominal concentrates work in phase 2 (16–32h)
    # At theta_worst phase ordering:          1 < 3 < 2  (complete reversal for phases 1&2)
    #   theta_worst ≈ [0.105, 0.30, 0.21] /h → robust concentrates work in phase 1 (0–16h)
    # Large uncertainty on phase 2 (rho = 500% of nominal) drives the reversal.
    # Both nominal and robust schedules improve clearly over equal; their optimal
    # strategies are visually distinct, illustrating the value of robust optimization.
    tau      = np.array([0, 16*3600, 32*3600, 48*3600], dtype=float)
    th_step  = np.array([1/(10*3600), 1/(20*3600), 1/(5*3600)], dtype=float)
    rho_step = np.array([0.05, 5.0, 0.05]) * th_step
    # theta_worst = [0.105, 0.30, 0.21] /h
    step_res = run_comparison(
        "Step hazard  [3-phase ordering reversal; phase 2 safe nominally but risky at theta*]",
        RobustStepHazardProblem(T, K, epsilon, tau, th_step, np.zeros_like(th_step), q),
        RobustStepHazardProblem(T, K, epsilon, tau, th_step, rho_step, q),
        lambda_nom=make_step_lambda_fn(tau, th_step),
        lambda_wc =make_step_lambda_fn(tau, th_step + rho_step),
    )

    # ── 2. Polynomial hazard, degree 2 ──────────────────────────────────────
    # lambda(t) = a0 + a2*(t/T)^2;  large uncertainty on a2 (late-job coefficient).
    # Nominal: quadratic term dominates late → front-load.
    # rho_a2=80% → worst-case a2 much steeper → robust front-loads even more than nominal.
    th_poly  = np.array([1/(40*3600), 0.0, 3/(10*3600)], dtype=float)
    rho_poly = np.array([0.05, 0.0, 0.80]) * th_poly
    nom_poly = RobustPolynomialHazardProblem(T, K, epsilon, th_poly, np.zeros_like(th_poly), q)
    rob_poly = RobustPolynomialHazardProblem(T, K, epsilon, th_poly, rho_poly, q)
    poly_res = run_comparison(
        "Polynomial hazard, deg-2  [rho_a2 = 80%]",
        nom_poly, rob_poly,
        lambda_nom=nom_poly._worst_problem.lambda_fn,
        lambda_wc =rob_poly._worst_problem.lambda_fn,
    )

    # ── 3. Power-law hazard ─────────────────────────────────────────────────
    # lambda(t) = theta0*(t/T)^theta1;  nominal strongly accelerating (theta1=2.5).
    # rho1=2.0 → worst-case theta1=0.5 (near-flat); rho0=30% scales up overall rate.
    # Nominal over-front-loads for flat worst-case; robust hedges toward equal spacing.
    th_pl  = np.array([1/(8*3600), 2.5], dtype=float)
    rho_pl = np.array([0.3 * th_pl[0], 2.0])
    nom_pl = RobustPowerLawHazardProblem(T, K, epsilon, th_pl, np.zeros_like(rho_pl), q)
    rob_pl = RobustPowerLawHazardProblem(T, K, epsilon, th_pl, rho_pl, q)
    pl_res = run_comparison(
        "Power-law hazard  [theta1=2.5, rho0=30%, rho1=2.0  ->  wc theta1=0.5]",
        nom_pl, rob_pl,
        lambda_nom=nom_pl._worst_problem.lambda_fn,
        lambda_wc =rob_pl._worst_problem.lambda_fn,
    )

    all_results   = [step_res, poly_res, pl_res]
    short_labels  = ["Step", "Polynomial", "Power-law"]
    sched_keys    = ["equal", "nom_pgd", "nom_md", "nom_admm", "rob_pgd", "rob_md", "rob_admm"]
    sched_disp    = ["equal", "nom\nPGD", "nom\nMD", "nom\nADMM", "rob\nPGD", "rob\nMD", "rob\nADMM"]
    bar_clrs7     = ["steelblue", "tomato", "#ff9999", "darkorange",
                     "seagreen",  "#90ee90", "#2e7d32"]
    k_idx         = np.arange(K)
    bar_w         = 0.18
    x7            = np.arange(7)

    # ── Plot 1: Schedules 3×2 ──────────────────────────────────────────────
    fig1, axes1 = plt.subplots(3, 2, figsize=(14, 12))
    fig1.suptitle("Robust vs nominal optimal schedules (PGD, MD, ADMM)", fontsize=13)
    for row, (res, slabel) in enumerate(zip(all_results, short_labels)):
        for col, (title, d_keys, clrs, lbls) in enumerate([
            ("Nominal problem (theta = theta_hat)",
             ["equal", "nom_pgd", "nom_md", "nom_admm"],
             ["steelblue", "tomato", "#ff9999", "darkorange"],
             ["equal", "PGD", "MD", "ADMM"]),
            ("Robust problem (theta = theta*)",
             ["equal", "rob_pgd", "rob_md", "rob_admm"],
             ["steelblue", "seagreen", "#90ee90", "#2e7d32"],
             ["equal", "PGD", "MD", "ADMM"]),
        ]):
            ax = axes1[row, col]
            n_bars = len(d_keys)
            offsets = np.arange(n_bars) - (n_bars - 1) / 2.0
            for i, (dk, color, lbl) in enumerate(zip(d_keys, clrs, lbls)):
                ax.bar(k_idx + offsets[i] * bar_w, res["deltas"][dk] / 3600,
                       width=bar_w, label=lbl, color=color, alpha=0.85)
            ax.set_title(f"{slabel} — {title}", fontsize=9)
            ax.set_xlabel("Interval k"); ax.set_ylabel("delta_k (h)")
            ax.set_xticks(k_idx); ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig("figures/schedules_robust.png", dpi=150, bbox_inches="tight")
    print("\nSaved figures/schedules_robust.png")

    # ── Plot 2: Convergence 3×2 ────────────────────────────────────────────
    fig2, axes2 = plt.subplots(3, 2, figsize=(14, 12))
    fig2.suptitle("Convergence: objective vs iteration (PGD, MD, ADMM)", fontsize=13)
    for row, (res, slabel) in enumerate(zip(all_results, short_labels)):
        for col, (pgd_k, md_k, admm_k, prob_k, title) in enumerate([
            ("nom_pgd", "nom_md", "nom_admm", "nominal_prob", "Nominal problem"),
            ("rob_pgd", "rob_md", "rob_admm", "robust_prob",  "Robust problem"),
        ]):
            ax = axes2[row, col]
            ph = res[pgd_k]["history"]
            mh = res[md_k]["history"]
            ah = res[admm_k]["history"]
            ax.plot([h["iter"] for h in ph], [h["objective"] for h in ph],
                    color="tomato",     lw=1.5, label="PGD")
            ax.plot([h["iter"] for h in mh], [h["objective"] for h in mh],
                    color="seagreen",   lw=1.5, label="MD (EG)")
            ax.plot([h["iter"] for h in ah], [h["objective"] for h in ah],
                    color="darkorange", lw=1.5, label="ADMM")
            eq_obj = res[prob_k].objective_from_delta(equal_delta)
            ax.axhline(eq_obj, color="steelblue", ls="--", lw=1.0, label="Equal schedule")
            ax.set_title(f"{slabel} — {title}", fontsize=9)
            ax.set_xlabel("Iteration"); ax.set_ylabel("Objective (s)")
            ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig("figures/convergence_robust.png", dpi=150, bbox_inches="tight")
    print("Saved figures/convergence_robust.png")

    # ── Plot 3: Analytic objective 3×2 ────────────────────────────────────
    fig3, axes3 = plt.subplots(3, 2, figsize=(14, 12))
    fig3.suptitle("Analytic objective at theta_hat and theta*", fontsize=13)
    for row, (res, slabel) in enumerate(zip(all_results, short_labels)):
        for col, (theta_idx, theta_lbl) in enumerate([(0, "at theta_hat"), (1, "at theta*")]):
            ax = axes3[row, col]
            vals = [res["analytic"][k][theta_idx] for k in sched_keys]
            ax.bar(x7, vals, color=bar_clrs7, alpha=0.85)
            eq_val = vals[0]
            for xi in range(1, 7):
                impr = 100 * (eq_val - vals[xi]) / eq_val
                ax.text(xi, vals[xi] * 1.01, f"{impr:+.1f}%", ha="center", fontsize=7)
            ax.set_title(f"{slabel} — {theta_lbl}", fontsize=9)
            ax.set_xticks(x7); ax.set_xticklabels(sched_disp, fontsize=8)
            ax.set_ylabel("Objective (s)")
    plt.tight_layout()
    plt.savefig("figures/analytic_objective_robust.png", dpi=150, bbox_inches="tight")
    print("Saved figures/analytic_objective_robust.png")

    # ── Plot 4: MC runtime 3×2 ─────────────────────────────────────────────
    if RUN_MC:
        fig4, axes4 = plt.subplots(3, 2, figsize=(14, 12))
        fig4.suptitle(f"Monte Carlo mean runtime ({N_MC} trials)", fontsize=13)
        for row, (res, slabel) in enumerate(zip(all_results, short_labels)):
            for col, (mc_idx, lam_lbl) in enumerate([(0, "nominal lambda"), (1, "worst-case lambda")]):
                ax = axes4[row, col]
                vals = [res["mc"][k][mc_idx]["mean_runtime"]  for k in sched_keys]
                errs = [res["mc"][k][mc_idx]["stderr_runtime"] for k in sched_keys]
                ax.bar(x7, vals, yerr=errs, color=bar_clrs7, alpha=0.85, capsize=4)
                eq_val = vals[0]
                for xi in range(1, 7):
                    impr = 100 * (eq_val - vals[xi]) / eq_val
                    ax.text(xi, (vals[xi] + errs[xi]) * 1.02, f"{impr:+.1f}%", ha="center", fontsize=7)
                ax.set_title(f"{slabel} — {lam_lbl}", fontsize=9)
                ax.set_xticks(x7); ax.set_xticklabels(sched_disp, fontsize=8)
                ax.set_ylabel("Mean runtime (s)")
        plt.tight_layout()
        plt.savefig("figures/mc_runtime_robust.png", dpi=150, bbox_inches="tight")
        print("Saved figures/mc_runtime_robust.png")

    # plt.show()