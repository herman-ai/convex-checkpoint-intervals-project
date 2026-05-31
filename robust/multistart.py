"""
Multi-start wrappers and automatic step-size selection.

  - _diverse_inits    — equal / back-loaded / front-loaded starting schedules
  - _auto_pgd_step    — auto-scale PGD step size from initial gradient
  - _auto_md_step     — auto-scale MD step size (in nats) from initial gradient
  - _auto_admm_rho    — auto-scale ADMM penalty to match PGD step
  - _best_pgd / _best_md / _best_admm — run from all three inits, return best
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from optimizers import optimize_pgd_internal_knots, optimize_mirror_descent, optimize_admm


# ── diverse initializations ───────────────────────────────────────────────────

def _diverse_inits(K: int, T: float, epsilon: float) -> list:
    """
    Three deterministic starting schedules for multi-start optimization:
      - equal:        T/K per interval
      - back-loaded:  linearly increasing intervals (weights 1, 2, ..., K)
      - front-loaded: linearly decreasing intervals (weights K, K-1, ..., 1)
    """
    T_tilde = T - K * epsilon
    equal   = np.full(K, T / K)
    w_up    = np.arange(1, K + 1, dtype=float)
    back    = epsilon + T_tilde * w_up / w_up.sum()
    w_dn    = np.arange(K, 0, -1, dtype=float)
    front   = epsilon + T_tilde * w_dn / w_dn.sum()
    return [equal, back, front]


# ── automatic step-size selection ─────────────────────────────────────────────

def _auto_pgd_step(prob, target_frac: float = 0.05) -> float:
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


def _auto_md_step(prob, target_logit: float = 0.05, max_alpha: float = 5.0) -> float:
    """
    Set the initial MD step size (alpha_0) so the first EG update shifts
    log-weights by about `target_logit` nats for the highest-gradient component.

    The EG update is: log(w_k^new) = log(w_k^old) - alpha * g_k + const,
    so alpha * g_inf is the log-weight shift of the steepest component.
    target_logit=0.05 → ~5% fractional weight change per step, matching
    _auto_pgd_step's target_frac=0.05 in Euclidean space.
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
    Set ADMM penalty ρ = 1 / (2 * pgd_step) so ADMM z-updates are on the same
    scale as PGD gradient steps.
    """
    return 1.0 / (2.0 * _auto_pgd_step(prob))


# ── best-of-three-starts wrappers ─────────────────────────────────────────────

def _best_pgd(problem, **kwargs) -> Dict[str, object]:
    """PGD from three diverse initializations; returns the best result."""
    inits   = _diverse_inits(problem.num_intervals, problem.total_useful_work, problem.epsilon)
    results = [optimize_pgd_internal_knots(problem, init_delta=d, **kwargs) for d in inits]
    return min(results, key=lambda r: r["objective"])


def _best_md(problem, **kwargs) -> Dict[str, object]:
    """Mirror descent from three diverse initializations; returns the best result."""
    inits   = _diverse_inits(problem.num_intervals, problem.total_useful_work, problem.epsilon)
    results = [optimize_mirror_descent(problem, init_delta=d, **kwargs) for d in inits]
    return min(results, key=lambda r: r["objective"])


def _best_admm(problem, **kwargs) -> Dict[str, object]:
    """ADMM from three diverse initializations; returns the best result."""
    inits   = _diverse_inits(problem.num_intervals, problem.total_useful_work, problem.epsilon)
    results = [optimize_admm(problem, init_delta=d, **kwargs) for d in inits]
    return min(results, key=lambda r: r["objective"])
