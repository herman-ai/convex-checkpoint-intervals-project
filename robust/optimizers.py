"""
Optimization algorithms for robust checkpoint scheduling.

  - optimize_pgd_internal_knots  — PGD with Armijo backtracking (T-parameterization)
  - optimize_mirror_descent      — Exponentiated gradient / mirror descent (delta-parameterization)
  - optimize_admm                — ADMM with diagonal Newton x-step + PAV z-step
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from utils.helpers import UsefulWorkHazardProblem, integrate_lambda, interval_work_integral
from problems import RobustStepHazardProblem
from projection import project_internal_knots_pav


# ── PGD ───────────────────────────────────────────────────────────────────────

def optimize_pgd_internal_knots(
    problem,
    max_iters: int = 500,
    step_size: float = 1e3,
    num_steps: int = 256,
    init_delta: np.ndarray | None = None,
) -> Dict[str, object]:
    """
    PGD in the T-parameterization with Armijo backtracking line search.

    The sufficient-decrease test uses the projected Armijo condition

        f(T + d) <= f(T) + sigma * grad^T d,

    where d = T_new - T_internal is the projected step.
    """
    K = problem.num_intervals
    if init_delta is None:
        delta0 = np.full(K, problem.total_useful_work / K)
    else:
        delta0 = np.asarray(init_delta, dtype=float).copy()
    T_knots   = problem.delta_to_knots(delta0)
    T_internal = T_knots[1:-1].copy()

    history = []
    sigma   = 1e-4

    for it in range(max_iters):
        grad     = problem.gradient_internal_knots(T_internal, num_steps=num_steps)
        obj_curr = problem.objective_from_internal_knots(T_internal, num_steps=num_steps)

        alpha  = step_size
        T_new  = T_internal.copy()
        obj_new = obj_curr

        for _ in range(30):
            candidate = T_internal - alpha * grad
            T_try     = project_internal_knots_pav(
                candidate,
                total_useful_work=problem.total_useful_work,
                epsilon=problem.epsilon,
            )
            d       = T_try - T_internal
            obj_try = problem.objective_from_internal_knots(T_try, num_steps=num_steps)

            if obj_try <= obj_curr + sigma * float(np.dot(grad, d)):
                T_new   = T_try
                obj_new = obj_try
                break

            alpha *= 0.5

        update_norm = np.linalg.norm(T_new - T_internal)
        history.append({"iter": it, "objective": obj_new, "update_norm": update_norm})
        T_internal = T_new
        if update_norm < 1e-6:
            break

    T_full = problem.internal_knots_to_full(T_internal)
    delta  = problem.full_knots_to_delta(T_full)
    return {
        "T_internal": T_internal,
        "delta":      delta,
        "objective":  problem.objective_from_delta(delta, num_steps=num_steps),
        "history":    history,
    }


# ── Mirror descent (EG) ───────────────────────────────────────────────────────

def optimize_mirror_descent(
    problem,
    max_iters: int = 1000,
    step_size: float = 1e-2,
    num_steps: int = 256,
    init_delta: np.ndarray | None = None,
) -> Dict[str, object]:
    """
    Mirror descent with negative-entropy mirror map (exponentiated gradient)
    in the shifted simplex coordinates Delta_tilde_k = Delta_k - epsilon.

    Update rule:
        Delta_tilde^{t+1}_k  ∝  Delta_tilde^t_k * exp(-alpha_t * g_k)

    where g = grad_Delta f and the weights are renormalised to sum T_tilde.
    Step schedule: alpha_t = step_size / sqrt(t+1)  (standard subgradient schedule).
    Best-iterate tracking ensures the returned schedule is never worse than the
    starting point.
    """
    K       = problem.num_intervals
    T_tilde = problem.total_useful_work - K * problem.epsilon

    if init_delta is None:
        delta_tilde = np.full(K, T_tilde / K, dtype=float)
    else:
        d0          = np.asarray(init_delta, dtype=float)
        delta_tilde = np.maximum(d0 - problem.epsilon, 1e-10)
        delta_tilde *= T_tilde / delta_tilde.sum()

    delta_init_full  = delta_tilde + problem.epsilon
    best_obj         = problem.objective_from_delta(delta_init_full, num_steps=num_steps)
    best_delta_tilde = delta_tilde.copy()

    history = []
    for it in range(max_iters):
        delta = delta_tilde + problem.epsilon
        grad  = problem.gradient_delta(delta, num_steps=num_steps)

        alpha = step_size / np.sqrt(it + 1)

        log_weights  = np.log(delta_tilde) - alpha * grad
        log_weights -= log_weights.max()
        weights      = np.exp(log_weights)
        delta_tilde_new = T_tilde * weights / weights.sum()

        update_norm = np.linalg.norm(delta_tilde_new - delta_tilde)

        obj_new = problem.objective_from_delta(delta_tilde_new + problem.epsilon,
                                               num_steps=num_steps)
        if obj_new < best_obj:
            best_obj         = obj_new
            best_delta_tilde = delta_tilde_new.copy()

        history.append({"iter": it, "objective": best_obj, "update_norm": update_norm})
        delta_tilde = delta_tilde_new
        if update_norm < 1e-6:
            break

    delta = best_delta_tilde + problem.epsilon
    return {
        "delta":     delta,
        "objective": best_obj,
        "history":   history,
    }


# ── ADMM ─────────────────────────────────────────────────────────────────────

def _interval_h_grad_and_hess(
    problem, k: int, a: float, b: float, num_steps: int = 64
) -> tuple:
    """
    Returns (dh/da, dh/db, d²h/da², d²h/db²) for interval k with endpoints a, b.

    Exact analytic formulas:
        dh/da   = -exp(L) * (1 + q * lambda(a))
        dh/db   =  lambda(b) * (ut + q * exp(L)) + 1
        d²h/da² =  lambda(a) * exp(L) * (1 + q * lambda(a))
        d²h/db² =  lambda(b)^2 * (ut + q * exp(L)) + lambda(b)

    where  L = ∫_a^b lambda,  ut = exp(L) * ∫_a^b exp(-Λ(u)) du.
    """
    q_k = float(problem.q[k])

    if isinstance(problem, RobustStepHazardProblem):
        theta  = problem.theta_worst
        L      = float(sum(
            theta[j] * max(0.0, min(b, problem.tau[j + 1]) - max(a, problem.tau[j]))
            for j in range(len(theta))
        ))
        exp_L  = np.exp(L)
        ut     = problem.expected_useful_work_term(a, b, theta)
        lam_a  = problem.lambda_at(a, theta)
        lam_b  = problem.lambda_at(b, theta)
    elif hasattr(problem, "_worst_problem"):
        wp     = problem._worst_problem
        L      = integrate_lambda(wp.lambda_fn, a, b, num_steps=num_steps)
        exp_L  = np.exp(L)
        I      = interval_work_integral(wp.lambda_fn, a, b, num_steps=num_steps)
        ut     = exp_L * I
        lam_a  = wp.lambda_fn(a)
        lam_b  = wp.lambda_fn(b)
    else:
        raise NotImplementedError(
            f"_interval_h_grad_and_hess: unsupported type {type(problem)}"
        )

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
    ADMM for checkpoint scheduling.

    Variables:
        x[k] = (x_k^-, x_k^+) ∈ R²  — local interval endpoints  (shape K×2)
        z[m] ∈ R                      — global consensus checkpoints (shape K+1)
    Constraint:  x_k^- = z[k], x_k^+ = z[k+1]

    x-step: K independent 2-D diagonal-Newton subproblems.
    z-step: closed-form unconstrained minimiser → PAV projection.
    Dual:   y ← y + ρ(x − Mz).
    """
    K       = problem.num_intervals
    T       = problem.total_useful_work
    epsilon = problem.epsilon

    if init_delta is None:
        delta0 = np.full(K, T / K)
    else:
        delta0 = np.asarray(init_delta, dtype=float).copy()

    z = problem.delta_to_knots(delta0).copy()
    x = np.stack([z[:-1], z[1:]], axis=1).copy()
    y = np.zeros((K, 2), dtype=float)

    history: List[Dict] = []

    for it in range(max_iters):
        Mz = np.stack([z[:-1], z[1:]], axis=1)

        # x-step
        for k in range(K):
            v_a, v_b = Mz[k, 0], Mz[k, 1]
            y_a, y_b = y[k, 0], y[k, 1]
            a, b     = x[k, 0], x[k, 1]
            for _ in range(inner_iters):
                b = max(b, a + 1e-6)
                dh_da, dh_db, d2h_da2, d2h_db2 = _interval_h_grad_and_hess(
                    problem, k, a, b, num_steps
                )
                H_a = d2h_da2 + rho
                H_b = d2h_db2 + rho
                a  -= (dh_da + y_a + rho * (a - v_a)) / H_a
                b  -= (dh_db + y_b + rho * (b - v_b)) / H_b
            x[k, 0] = a
            x[k, 1] = max(b, a + 1e-6)

        # z-step
        v_int      = (x[:-1, 1] + x[1:, 0]) / 2.0 + (y[:-1, 1] + y[1:, 0]) / (2.0 * rho)
        z[1:-1]    = project_internal_knots_pav(v_int, T, epsilon)

        # dual update
        Mz_new   = np.stack([z[:-1], z[1:]], axis=1)
        residual = x - Mz_new
        y       += rho * residual

        primal_res = float(np.linalg.norm(residual))
        delta      = np.diff(z)
        obj        = problem.objective_from_delta(delta, num_steps=num_steps)
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
