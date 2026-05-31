"""
Projection utilities for the checkpoint-scheduling feasible set.

    0 = T_0 <= T_1 <= ... <= T_{K-1} <= T_K = T
    T_k - T_{k-1} >= epsilon  for all k

Two implementations are provided:
  - project_internal_knots_pav   — fast O(K) Pool-Adjacent Violators algorithm
  - project_internal_knots_cvxpy — exact via CVXPY/SCS (slow; kept for reference)
"""

from __future__ import annotations

import numpy as np


def project_internal_knots_pav(
    v: np.ndarray, total_useful_work: float, epsilon: float
) -> np.ndarray:
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

    S_hat = np.empty(K + 1, dtype=float)
    S_hat[0] = 0.0
    for k in range(1, K):
        S_hat[k] = v[k - 1] - k * epsilon
    S_hat[K] = T_tilde

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

            blocks[i - 1] = {
                "mean": new_mu, "weight": new_w,
                "start": blocks[i - 1]["start"], "end": blocks[i]["end"],
            }
            blocks.pop(i)
            i = max(1, i - 1)
        else:
            i += 1

    S_star = np.empty(K + 1, dtype=float)
    for block in blocks:
        S_star[block["start"] : block["end"] + 1] = block["mean"]

    T_star = S_star + epsilon * np.arange(K + 1)
    return T_star[1:-1]


def project_internal_knots_cvxpy(
    v: np.ndarray, total_useful_work: float, epsilon: float
) -> np.ndarray:
    """
    Project v onto the feasible set via CVXPY/SCS (reference implementation).

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
