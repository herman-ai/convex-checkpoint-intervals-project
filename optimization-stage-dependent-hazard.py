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


def optimize_pgd_internal_knots(
    problem: UsefulWorkHazardProblem,
    max_iters: int = 200,
    step_size: float = 1e-2,
    num_steps: int = 256,
) -> Dict[str, object]:
    """
    PGD in the T-parameterization.
    """
    K = problem.num_intervals
    # Initialize equal spacing
    delta0 = np.full(K, problem.total_useful_work / K)
    T_knots = problem.delta_to_knots(delta0)
    T_internal = T_knots[1:-1].copy()

    history = []
    for it in range(max_iters):
        grad = problem.gradient_internal_knots(T_internal, num_steps=num_steps)
        candidate = T_internal - step_size * grad
        T_internal_new = project_internal_knots_cvxpy(
            candidate,
            total_useful_work=problem.total_useful_work,
            epsilon=problem.epsilon,
        )

        obj = problem.objective_from_internal_knots(T_internal_new, num_steps=num_steps)
        update_norm = np.linalg.norm(T_internal_new - T_internal)

        history.append(
            {
                "iter": it,
                "objective": obj,
                "update_norm": update_norm,
            }
        )

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

    pgd = optimize_pgd_internal_knots(prob, max_iters=100, step_size=1e3, num_steps=256)

    equal_delta = np.full(K, T / K)
    equal_obj   = prob.objective_from_delta(equal_delta, num_steps=256)
    opt_obj     = pgd["objective"]
    improvement = 100.0 * (equal_obj - opt_obj) / equal_obj

    print(f"  Equal obj:   {equal_obj:.2f}")
    print(f"  Opt obj:     {opt_obj:.2f}")
    print(f"  Improvement: {improvement:.2f}%")

    results[name] = {
        "lambda_fn":    lambda_fn,
        "equal_delta":  equal_delta,
        "opt_delta":    pgd["delta"],
        "equal_obj":    equal_obj,
        "opt_obj":      opt_obj,
        "improvement":  improvement,
        "history":      pgd["history"],
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
    results[name]["sim_equal_runtime"] = sim_equal["mean_runtime"]
    results[name]["sim_pgd_runtime"]   = sim_pgd["mean_runtime"]
    print(f"\n{name}:")
    print(f"  Equal schedule sim: {sim_equal}")
    print(f"  PGD schedule sim:   {sim_pgd}")

# ── Plot 1: 2×2 interval bars ─────────────────────────────────────────────
idx = np.arange(K)
bar_w = 0.35
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
names = list(results.keys())

fig, axes = plt.subplots(2, 2, figsize=(12, 9))
fig.suptitle("Optimized vs. equal schedules by hazard function", fontsize=13)

for i, (name, res) in enumerate(results.items()):
    ax = axes[i // 2][i % 2]

    ax.bar(idx - bar_w / 2, res["equal_delta"] / SECONDS_PER_HOUR,
           width=bar_w, label="Equal", color="steelblue", alpha=0.7)
    ax.bar(idx + bar_w / 2, res["opt_delta"] / SECONDS_PER_HOUR,
           width=bar_w, label="Optimized", color=colors[i], alpha=0.9)

    ax.set_title(f"{name}  ({res['improvement']:.1f}% improvement)", fontsize=10)
    ax.set_xlabel("Interval index")
    ax.set_ylabel("Useful work (h)")
    ax.set_xticks(idx)
    ax.set_xticklabels([f"$k={k}$" for k in idx])
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig("schedules_by_hazard.png", dpi=150, bbox_inches="tight")
print("\nSaved schedules_by_hazard.png")

# ── Plot 2: 1×2 summary — analytic objective + simulated mean runtime ──────
fig2, (ax_obj, ax_sim) = plt.subplots(1, 2, figsize=(12, 4))
names             = list(results.keys())
equal_objs        = [results[n]["equal_obj"]         for n in names]
opt_objs          = [results[n]["opt_obj"]           for n in names]
sim_equal_runtime = [results[n]["sim_equal_runtime"] for n in names]
sim_pgd_runtime   = [results[n]["sim_pgd_runtime"]   for n in names]
x = np.arange(len(names))

# Left: analytic objective
ax_obj.bar(x - bar_w / 2, equal_objs, width=bar_w, label="Equal",     color="steelblue", alpha=0.7)
ax_obj.bar(x + bar_w / 2, opt_objs,   width=bar_w, label="Optimized", color="tomato",    alpha=0.9)
for i, n in enumerate(names):
    ax_obj.text(i, max(equal_objs[i], opt_objs[i]) * 1.01,
                f"{results[n]['improvement']:.1f}%", ha="center", fontsize=9)
ax_obj.set_xticks(x)
ax_obj.set_xticklabels(names)
ax_obj.set_ylabel("Analytic objective (s)")
ax_obj.set_title("Analytic objective")
ax_obj.legend()

# Right: simulated mean runtime
sim_improvement = [100.0 * (e - p) / e for e, p in zip(sim_equal_runtime, sim_pgd_runtime)]
ax_sim.bar(x - bar_w / 2, sim_equal_runtime, width=bar_w, label="Equal",     color="steelblue", alpha=0.7)
ax_sim.bar(x + bar_w / 2, sim_pgd_runtime,   width=bar_w, label="Optimized", color="tomato",    alpha=0.9)
for i in range(len(names)):
    ax_sim.text(i, max(sim_equal_runtime[i], sim_pgd_runtime[i]) * 1.01,
                f"{sim_improvement[i]:.1f}%", ha="center", fontsize=9)
ax_sim.set_xticks(x)
ax_sim.set_xticklabels(names)
ax_sim.set_ylabel("Mean runtime (s)")
ax_sim.set_title("Simulated mean runtime (1000 trials)")
ax_sim.legend()
plt.tight_layout()
plt.savefig("objective_comparison.png", dpi=150, bbox_inches="tight")
print("Saved objective_comparison.png")

plt.show()

