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
from simulator_v2 import monte_carlo_schedule_useful_work_hazard


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
) -> Dict[str, object]:
    """
    PGD in the T-parameterization with a fixed step size.
    """
    K = problem.num_intervals
    delta0 = np.full(K, problem.total_useful_work / K)
    T_knots = problem.delta_to_knots(delta0)
    T_internal = T_knots[1:-1].copy()

    history = []
    for it in range(max_iters):
        grad = problem.gradient_internal_knots(T_internal, num_steps=num_steps)
        candidate = T_internal - step_size * grad
        T_internal_new = project_internal_knots_pav(
            candidate,
            total_useful_work=problem.total_useful_work,
            epsilon=problem.epsilon,
        )

        obj = problem.objective_from_internal_knots(T_internal_new, num_steps=num_steps)
        update_norm = np.linalg.norm(T_internal_new - T_internal)
        history.append({"iter": it, "objective": obj, "update_norm": update_norm})

        T_internal = T_internal_new
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
    sup-norm ~ O(K).  The EG theoretical optimum for K intervals and T
    iterations is  alpha* = sqrt(ln K) / (G * sqrt(T)) ~ 0.01.
    """
    K = problem.num_intervals
    T_tilde = problem.total_useful_work - K * problem.epsilon

    # Initialize: uniform shifted intervals
    delta_tilde = np.full(K, T_tilde / K, dtype=float)

    history = []
    for it in range(max_iters):
        delta = delta_tilde + problem.epsilon
        grad = problem.gradient_delta(delta, num_steps=num_steps)
        obj  = problem.objective_from_delta(delta, num_steps=num_steps)

        # Exponentiated gradient update (log-sum-exp for numerical stability)
        log_weights = np.log(delta_tilde) - step_size * grad
        log_weights -= log_weights.max()              # shift for stability
        weights = np.exp(log_weights)
        delta_tilde_new = T_tilde * weights / weights.sum()

        update_norm = np.linalg.norm(delta_tilde_new - delta_tilde)
        history.append({"iter": it, "objective": obj, "update_norm": update_norm})

        delta_tilde = delta_tilde_new
        if update_norm < 1e-6:
            break

    delta = delta_tilde + problem.epsilon
    return {
        "delta": delta,
        "objective": problem.objective_from_delta(delta, num_steps=num_steps),
        "history": history,
    }



K = 8
T = 48.0 * SECONDS_PER_HOUR  # Total useful work (e.g., 48 hours)
epsilon = 0.5 * SECONDS_PER_HOUR  # Minimum interval length (e.g., 30 minutes)

q = np.array([
    5 * 60, 5 * 60, 10 * 60, 10 * 60,
    15 * 60, 15 * 60, 20 * 60, 20 * 60
], dtype=float)

HAZARD_FUNCTIONS = {
    "Step":        lambda_fn_step,
    "Linear":      lambda_fn_linear,
    "Convex":      lambda_fn_convex,
    "Three-phase": lambda_fn_three_phase,
}

# ── Run optimization for each hazard function ──────────────────────────────
results = {}
for name, lambda_fn in HAZARD_FUNCTIONS.items():
    print(f"\n=== {name} ===")
    prob = UsefulWorkHazardProblem(
        total_useful_work=T,
        num_intervals=K,
        epsilon=epsilon,
        lambda_fn=lambda_fn,
        q=q,
    )

    pgd = optimize_pgd_internal_knots(prob, max_iters=500, step_size=1e3, num_steps=256)
    md  = optimize_mirror_descent(prob, max_iters=500, step_size=1e-2, num_steps=256)

    equal_delta = np.full(K, T / K)
    equal_obj   = prob.objective_from_delta(equal_delta, num_steps=256)
    pgd_obj     = pgd["objective"]
    md_obj      = md["objective"]
    pgd_improvement = 100.0 * (equal_obj - pgd_obj) / equal_obj
    md_improvement  = 100.0 * (equal_obj - md_obj)  / equal_obj

    print(f"  Equal obj:        {equal_obj:.2f}")
    print(f"  PGD obj:          {pgd_obj:.2f}  ({pgd_improvement:.2f}%)")
    print(f"  Mirror descent:   {md_obj:.2f}  ({md_improvement:.2f}%)")

    results[name] = {
        "lambda_fn":        lambda_fn,
        "equal_delta":      equal_delta,
        "opt_delta":        pgd["delta"],
        "md_delta":         md["delta"],
        "equal_obj":        equal_obj,
        "opt_obj":          pgd_obj,
        "md_obj":           md_obj,
        "improvement":      pgd_improvement,
        "md_improvement":   md_improvement,
        "history":          pgd["history"],
        "md_history":       md["history"],
    }

# ── Monte Carlo simulations ────────────────────────────────────────────────
print("\n=== Monte Carlo simulations ===")
for name, res in results.items():
    lambda_fn = res["lambda_fn"]

    sim_equal = monte_carlo_schedule_useful_work_hazard(
        delta=res["equal_delta"],
        lambda_fn=lambda_fn,
        q=q,
        num_trials=1000,
    )
    sim_pgd = monte_carlo_schedule_useful_work_hazard(
        delta=res["opt_delta"],
        lambda_fn=lambda_fn,
        q=q,
        num_trials=1000,
    )
    sim_md = monte_carlo_schedule_useful_work_hazard(
        delta=res["md_delta"],
        lambda_fn=lambda_fn,
        q=q,
        num_trials=1000,
    )
    results[name]["sim_equal_runtime"] = sim_equal["mean_runtime"]
    results[name]["sim_pgd_runtime"]   = sim_pgd["mean_runtime"]
    results[name]["sim_md_runtime"]    = sim_md["mean_runtime"]
    print(f"\n{name}:")
    print(f"  Equal schedule sim: {sim_equal}")
    print(f"  PGD schedule sim:   {sim_pgd}")
    print(f"  MD schedule sim:    {sim_md}")

# ── Plot 1: 2×2 interval bars ─────────────────────────────────────────────
idx = np.arange(K)
bar_w = 0.25
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
names = list(results.keys())

fig, axes = plt.subplots(2, 2, figsize=(12, 9))
fig.suptitle("Schedules by hazard function", fontsize=13)

for i, (name, res) in enumerate(results.items()):
    ax = axes[i // 2][i % 2]

    ax.bar(idx - bar_w, res["equal_delta"] / SECONDS_PER_HOUR,
           width=bar_w, label="Equal", color="steelblue", alpha=0.7)
    ax.bar(idx,         res["opt_delta"]   / SECONDS_PER_HOUR,
           width=bar_w, label=f"PGD ({res['improvement']:.1f}%)", color="tomato", alpha=0.9)
    ax.bar(idx + bar_w, res["md_delta"]    / SECONDS_PER_HOUR,
           width=bar_w, label=f"MD ({res['md_improvement']:.1f}%)", color="seagreen", alpha=0.9)

    ax.set_title(name, fontsize=10)
    ax.set_xlabel("Interval index")
    ax.set_ylabel("Useful work (h)")
    ax.set_xticks(idx)
    ax.set_xticklabels([f"$k={k}$" for k in idx])
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig("figures/schedules_by_hazard.png", dpi=150, bbox_inches="tight")
print("\nSaved figures/schedules_by_hazard.png")

# ── Plot 2: 1×2 summary — analytic objective + simulated mean runtime ──────
fig2, (ax_obj, ax_sim) = plt.subplots(1, 2, figsize=(13, 4))
names             = list(results.keys())
equal_objs        = [results[n]["equal_obj"]         for n in names]
opt_objs          = [results[n]["opt_obj"]           for n in names]
md_objs           = [results[n]["md_obj"]            for n in names]
sim_equal_runtime = [results[n]["sim_equal_runtime"] for n in names]
sim_pgd_runtime   = [results[n]["sim_pgd_runtime"]   for n in names]
sim_md_runtime    = [results[n]["sim_md_runtime"]    for n in names]
x = np.arange(len(names))
sw = 0.25  # summary bar width

# Left: analytic objective
ax_obj.bar(x - sw, equal_objs, width=sw, label="Equal",  color="steelblue", alpha=0.7)
ax_obj.bar(x,      opt_objs,   width=sw, label="PGD",    color="tomato",    alpha=0.9)
ax_obj.bar(x + sw, md_objs,    width=sw, label="MD",     color="seagreen",  alpha=0.9)
for i, n in enumerate(names):
    ax_obj.text(i,      opt_objs[i] * 1.01, f"{results[n]['improvement']:.1f}%",   ha="center", fontsize=8, color="tomato")
    ax_obj.text(i + sw, md_objs[i]  * 1.01, f"{results[n]['md_improvement']:.1f}%", ha="center", fontsize=8, color="seagreen")
ax_obj.set_xticks(x)
ax_obj.set_xticklabels(names)
ax_obj.set_ylabel("Analytic objective (s)")
ax_obj.set_title("Analytic objective")
ax_obj.legend()

# Right: simulated mean runtime
sim_pgd_impr = [100.0 * (e - p) / e for e, p in zip(sim_equal_runtime, sim_pgd_runtime)]
sim_md_impr  = [100.0 * (e - p) / e for e, p in zip(sim_equal_runtime, sim_md_runtime)]
ax_sim.bar(x - sw, sim_equal_runtime, width=sw, label="Equal", color="steelblue", alpha=0.7)
ax_sim.bar(x,      sim_pgd_runtime,   width=sw, label="PGD",   color="tomato",    alpha=0.9)
ax_sim.bar(x + sw, sim_md_runtime,    width=sw, label="MD",    color="seagreen",  alpha=0.9)
for i in range(len(names)):
    top = max(sim_equal_runtime[i], sim_pgd_runtime[i], sim_md_runtime[i]) * 1.01
    ax_sim.text(i,      top, f"{sim_pgd_impr[i]:.1f}%", ha="center", fontsize=8, color="tomato")
    ax_sim.text(i + sw, top, f"{sim_md_impr[i]:.1f}%",  ha="center", fontsize=8, color="seagreen")
ax_sim.set_xticks(x)
ax_sim.set_xticklabels(names)
ax_sim.set_ylabel("Mean runtime (s)")
ax_sim.set_title("Simulated mean runtime (1000 trials)")
ax_sim.legend()
plt.tight_layout()
plt.savefig("figures/objective_comparison.png", dpi=150, bbox_inches="tight")
print("Saved figures/objective_comparison.png")

# ── Plot 3: objective vs. iteration for PGD and MD ────────────────────────
fig3, axes3 = plt.subplots(2, 2, figsize=(12, 8), sharex=False)
fig3.suptitle("Objective value vs. iteration", fontsize=13)

for i, (name, res) in enumerate(results.items()):
    ax = axes3[i // 2][i % 2]

    pgd_iters = [h["iter"] for h in res["history"]]
    pgd_objs  = [h["objective"] for h in res["history"]]
    md_iters  = [h["iter"] for h in res["md_history"]]
    md_objs   = [h["objective"] for h in res["md_history"]]

    ax.plot(pgd_iters, pgd_objs,  color="tomato",   linewidth=1.5, label="PGD")
    ax.plot(md_iters,  md_objs,   color="seagreen",  linewidth=1.5, label="MD (EG)")
    ax.axhline(res["equal_obj"], color="steelblue", linewidth=1.0,
               linestyle="--", label="Equal schedule")

    ax.set_title(name, fontsize=10)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Objective (s)")
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig("figures/convergence.png", dpi=150, bbox_inches="tight")
print("Saved figures/convergence.png")

# ── Plot 4: hazard functions ───────────────────────────────────────────────
t_vals = np.linspace(0, T, 1000)
hz_colors = {"Step": "steelblue", "Linear": "tomato", "Convex": "seagreen", "Three-phase": "darkorchid"}

fig4, ax_hz = plt.subplots(figsize=(8, 4))
for name, lambda_fn in HAZARD_FUNCTIONS.items():
    lam_vals = np.array([lambda_fn(t) * SECONDS_PER_HOUR for t in t_vals])  # convert to 1/hour
    ax_hz.plot(t_vals / SECONDS_PER_HOUR, lam_vals, label=name, color=hz_colors[name], linewidth=2)

ax_hz.set_xlabel("Useful work completed (h)")
ax_hz.set_ylabel("Hazard rate $\\lambda(t)$ (h$^{-1}$)")
ax_hz.set_title("Hazard functions")
ax_hz.legend()
ax_hz.set_xlim(0, T / SECONDS_PER_HOUR)
ax_hz.set_ylim(bottom=0)
plt.tight_layout()
plt.savefig("figures/hazard_functions.png", dpi=150, bbox_inches="tight")
print("Saved figures/hazard_functions.png")

plt.show()

