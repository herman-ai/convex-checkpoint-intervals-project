from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "robust"))
from utils.helpers import SECONDS_PER_HOUR, UsefulWorkHazardProblem  # noqa: E402
from optimizers import optimize_pgd_internal_knots, optimize_mirror_descent, optimize_admm  # noqa: E402
from problems import RobustStepHazardProblem, RobustPolynomialHazardProblem, RobustPowerLawHazardProblem, make_step_lambda_fn  # noqa: E402


T = 48.0 * SECONDS_PER_HOUR


def lambda_fn_step(t: float) -> float:
    if t < 16 * SECONDS_PER_HOUR:
        return 1.0 / (48 * SECONDS_PER_HOUR)
    if t < 32 * SECONDS_PER_HOUR:
        return 1.0 / (18 * SECONDS_PER_HOUR)
    return 1.0 / (6 * SECONDS_PER_HOUR)


def lambda_fn_polynomial(t: float) -> float:
    alpha = min(max(t / T, 0.0), 1.0)
    base = 1.0 / (38 * SECONDS_PER_HOUR)
    curvature = 1.0 / (2 * SECONDS_PER_HOUR)
    return base + curvature * (alpha - 0.5) ** 2


def lambda_fn_power_law(t: float) -> float:
    theta0 = 1.0 / (8 * SECONDS_PER_HOUR)
    theta1 = 4.0
    return theta0 * (max(t, 0.0) / T) ** theta1


def _get_lambda_fns(prob: Any):
    """
    Return (nom_fn, worst_fn, best_fn) for visualization.

    worst = theta_hat + rho (upper corner of uncertainty set, clipped so lambda >= 0)
    best  = theta_hat - rho (lower corner of uncertainty set, lambda clipped to 0)
    """
    if isinstance(prob, RobustStepHazardProblem):
        theta_best = np.maximum(prob.theta_hat - prob.rho, 0.0)
        nom_fn   = lambda t: prob.lambda_at(t, prob.theta_hat)
        worst_fn = lambda t: prob.lambda_at(t, prob.theta_worst)
        best_fn  = lambda t: prob.lambda_at(t, theta_best)

    elif isinstance(prob, RobustPolynomialHazardProblem):
        theta_hat  = prob.theta_hat
        theta_best = theta_hat - prob.rho          # no clip on coefficients
        T_         = prob.T
        js         = np.arange(len(theta_hat), dtype=float)
        nom_fn   = lambda t: float(np.dot((float(t) / T_) ** js, theta_hat))
        worst_fn = prob._worst_problem.lambda_fn
        best_fn  = lambda t: max(float(np.dot((float(t) / T_) ** js, theta_best)), 0.0)

    elif isinstance(prob, RobustPowerLawHazardProblem):
        th0, th1 = float(prob.theta_hat[0]), float(prob.theta_hat[1])
        r0,  r1  = float(prob.rho[0]),       float(prob.rho[1])
        T_       = prob.T
        th0_best = max(th0 - r0, 0.0)
        th1_best = th1 + r1                        # higher exponent → steeper, lower early
        nom_fn   = lambda t: th0  * (max(float(t), 1e-12) / T_) ** th1
        worst_fn = prob._worst_problem.lambda_fn
        best_fn  = lambda t: th0_best * (max(float(t), 1e-12) / T_) ** th1_best

    else:
        raise NotImplementedError(f"_get_lambda_fns: unsupported type {type(prob)}")

    return nom_fn, worst_fn, best_fn


def plot_robust_hazards(output_path: str = "figures/robust/robust_hazards.png") -> None:
    os.makedirs("figures/robust", exist_ok=True)

    configs = [
        ("Step",       "steelblue"),
        ("Polynomial", "tomato"),
        ("Power-law",  "seagreen"),
    ]

    fig, axes = plt.subplots(1, len(configs), figsize=(15, 4.5), sharey=True)
    for ax, (label, color) in zip(axes, configs):
        prob = _make_robust_problem(label, nominal=False)
        nom_fn, worst_fn, best_fn = _get_lambda_fns(prob)

        t_values     = np.linspace(0.0, T, 1000)
        nom_values   = np.array([nom_fn(t)   * SECONDS_PER_HOUR for t in t_values], dtype=float)
        worst_values = np.array([worst_fn(t) * SECONDS_PER_HOUR for t in t_values], dtype=float)
        best_values  = np.array([best_fn(t)  * SECONDS_PER_HOUR for t in t_values], dtype=float)

        ax.fill_between(t_values / SECONDS_PER_HOUR, best_values, worst_values,
                        color=color, alpha=0.25, label="Uncertainty region")
        ax.plot(t_values / SECONDS_PER_HOUR, worst_values, color=color, linewidth=1.0,
                linestyle="--", alpha=0.7, label="Worst-case")
        ax.plot(t_values / SECONDS_PER_HOUR, best_values, color=color, linewidth=1.0,
                linestyle=":", alpha=0.7, label="Best-case")
        ax.plot(t_values / SECONDS_PER_HOUR, nom_values, color=color, linewidth=2, label="Nominal")
        ax.set_title(label)
        ax.set_xlabel("Useful work completed (h)")
        ax.set_xlim(0.0, T / SECONDS_PER_HOUR)
        ax.set_ylim(bottom=0.0)
        ax.legend(loc="best", framealpha=0.95)

    axes[0].set_ylabel("Hazard rate $\\lambda(t)$")
    fig.suptitle("Robust Hazards: Nominal and Worst-Case", fontsize=13)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")


def _make_robust_problem(label: str, nominal: bool) -> Any:
    if label == "Step":
        total_time = 48.0 * SECONDS_PER_HOUR
        K = 10
        epsilon = 0.5 * SECONDS_PER_HOUR
        tau = np.array([0.0, 16 * SECONDS_PER_HOUR, 32 * SECONDS_PER_HOUR, 48 * SECONDS_PER_HOUR], dtype=float)
        theta_hat = np.array([
            1.0 / (48 * SECONDS_PER_HOUR),
            1.0 / (18 * SECONDS_PER_HOUR),
            1.0 / (6  * SECONDS_PER_HOUR),
        ], dtype=float)
        rho = np.zeros_like(theta_hat) if nominal else np.array([0.1, 0.5, 0.1]) * theta_hat
        q = np.full(K, 10 * 60, dtype=float)
        return RobustStepHazardProblem(total_time, K, epsilon, tau, theta_hat, rho, q)

    if label == "Polynomial":
        total_time = 48.0 * SECONDS_PER_HOUR
        K = 10
        epsilon = 0.5 * SECONDS_PER_HOUR
        theta_hat = np.array([
            1.0 / (38 * SECONDS_PER_HOUR) + 0.25 / (2 * SECONDS_PER_HOUR),  # constant term
            -1.0 / (2 * SECONDS_PER_HOUR),                                    # linear term
            1.0 / (2 * SECONDS_PER_HOUR),                                     # quadratic term
        ], dtype=float)
        rho = np.zeros_like(theta_hat) if nominal else np.array([
            0.05 * theta_hat[0],
            0.1  / (2 * SECONDS_PER_HOUR),   # < |theta_1|=1/(2h), keeps linear term negative
            0.05 * theta_hat[2],
        ])
        q = np.full(K, 10 * 60, dtype=float)
        return RobustPolynomialHazardProblem(total_time, K, epsilon, theta_hat, rho, q)

    if label == "Power-law":
        total_time = 48.0 * SECONDS_PER_HOUR
        K = 10
        epsilon = 0.5 * SECONDS_PER_HOUR
        theta_hat = np.array([1.0 / (8 * SECONDS_PER_HOUR), 4.0], dtype=float)
        # worst-case: scale up (theta0 + rho0), exponent down (theta1 - rho1 = 2.5)
        rho = np.zeros_like(theta_hat) if nominal else np.array([
            0.2 * theta_hat[0],  # 20% scale uncertainty
            1.5,                  # exponent uncertainty: worst-case exponent = 4.0 - 1.5 = 2.5
        ])
        q = np.full(K, 10 * 60, dtype=float)
        return RobustPowerLawHazardProblem(total_time, K, epsilon, theta_hat, rho, q)

    raise ValueError(f"Unknown hazard label: {label}")


def run_multistart(
    problem: Any,
    n_starts: int = 20,
    seed: int = 42,
    pgd_kwargs: dict | None = None,
    md_kwargs: dict | None = None,
    admm_kwargs: dict | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    pgd_kwargs = pgd_kwargs or {}
    md_kwargs = md_kwargs or {}
    admm_kwargs = admm_kwargs or {}

    pgd_runs: list[dict] = []
    md_runs: list[dict] = []
    admm_runs: list[dict] = []

    K = problem.num_intervals
    T_tilde = problem.total_useful_work - K * problem.epsilon

    inits: list[np.ndarray] = []
    for _ in range(n_starts):
        proportions = rng.dirichlet(np.ones(K))
        inits.append(proportions * T_tilde + problem.epsilon)

    for i, init_delta in enumerate(inits):
        pgd_result = optimize_pgd_internal_knots(problem, init_delta=init_delta, **pgd_kwargs)
        pgd_runs.append(pgd_result)
        print(f"  PGD  start {i + 1}/{n_starts}: obj={pgd_result['objective']:.4f} ({len(pgd_result['history'])} iters)", flush=True)

        md_result = optimize_mirror_descent(problem, init_delta=init_delta, **md_kwargs)
        md_runs.append(md_result)
        print(f"  MD   start {i + 1}/{n_starts}: obj={md_result['objective']:.4f} ({len(md_result['history'])} iters)", flush=True)

        admm_result = optimize_admm(problem, init_delta=init_delta, **admm_kwargs)
        admm_runs.append(admm_result)
        print(f"  ADMM start {i + 1}/{n_starts}: obj={admm_result['objective']:.4f} ({len(admm_result['history'])} iters)", flush=True)

    all_runs = pgd_runs + md_runs + admm_runs
    best_run = min(all_runs, key=lambda r: r["objective"])

    return {
        "pgd_runs": pgd_runs,
        "md_runs": md_runs,
        "admm_runs": admm_runs,
        "best_delta": best_run["delta"],
        "best_obj": float(best_run["objective"]),
    }


def run_robust_comparison(
    label: str,
    n_starts: int = 10,
    pgd_kwargs: dict | None = None,
    md_kwargs: dict | None = None,
    admm_kwargs: dict | None = None,
) -> dict:
    pgd_kwargs = pgd_kwargs or {"max_iters": 2000, "step_size": 1e3, "num_steps": 256}
    md_kwargs = md_kwargs or {"max_iters": 2000, "step_size": 1e-2, "num_steps": 256}
    admm_kwargs = admm_kwargs or {"max_iters": 500, "rho": 5e-4, "num_steps": 256}

    robust_prob = _make_robust_problem(label, nominal=False)
    robust_ms = run_multistart(
        robust_prob, n_starts=n_starts,
        pgd_kwargs=pgd_kwargs, md_kwargs=md_kwargs, admm_kwargs=admm_kwargs,
    )
    equal_delta = np.full(robust_prob.num_intervals, robust_prob.total_useful_work / robust_prob.num_intervals)
    equal_obj = robust_prob.objective_from_delta(equal_delta)

    return {
        "label": label,
        "robust_ms": robust_ms,
        "robust_prob": robust_prob,
        "equal_obj": equal_obj,
        "best_obj": robust_ms["best_obj"],
    }


def plot_convergence_robust(
    label: str,
    pgd_runs: list[dict],
    md_runs: list[dict],
    admm_runs: list[dict],
    equal_obj: float,
    best_obj: float,
    output_path: str,
) -> None:
    pgd_best_idx  = min(range(len(pgd_runs)),  key=lambda i: pgd_runs[i]["objective"])
    md_best_idx   = min(range(len(md_runs)),   key=lambda i: md_runs[i]["objective"])
    admm_best_idx = min(range(len(admm_runs)), key=lambda i: admm_runs[i]["objective"])

    fig, ax = plt.subplots(figsize=(7, 4))

    for i, run in enumerate(pgd_runs):
        iters = [h["iter"] for h in run["history"]]
        objs  = [h["objective"] for h in run["history"]]
        if i == pgd_best_idx:
            ax.plot(iters, objs, color="tomato", linewidth=2.0, alpha=1.0, label="PGD (best)", zorder=3)
        else:
            ax.plot(iters, objs, color="tomato", linewidth=0.8, alpha=0.25, zorder=2)

    for i, run in enumerate(md_runs):
        iters = [h["iter"] for h in run["history"]]
        objs  = [h["objective"] for h in run["history"]]
        if i == md_best_idx:
            ax.plot(iters, objs, color="seagreen", linewidth=2.0, alpha=1.0, label="MD (best)", zorder=3)
        else:
            ax.plot(iters, objs, color="seagreen", linewidth=0.8, alpha=0.25, zorder=2)

    for i, run in enumerate(admm_runs):
        iters = [h["iter"] for h in run["history"]]
        objs  = [h["objective"] for h in run["history"]]
        if i == admm_best_idx:
            ax.plot(iters, objs, color="purple", linewidth=2.0, alpha=1.0, label="ADMM (best)", zorder=3)
        else:
            ax.plot(iters, objs, color="purple", linewidth=0.8, alpha=0.25, zorder=2)

    ax.axhline(equal_obj, color="steelblue", linestyle="--", linewidth=1.2,
               label=f"Equal Intervals ({equal_obj:.2f})")
    ax.axhline(best_obj, color="darkgreen", linestyle=":", linewidth=1.2,
               label=f"Best Optimized ({best_obj:.2f})")

    all_runs = pgd_runs + md_runs + admm_runs
    max_plotted = max(h["objective"] for run in all_runs for h in run["history"])
    #ax.set_ylim(best_obj * 0.95, min(equal_obj * 1.5, max_plotted * 1.05))
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Objective Value (s)")
    ax.set_title(f"{label} — Convergence ({len(pgd_runs)} starts)")
    ax.legend(framealpha=0.95)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")


def plot_convergence_combined_robust(entries: list[dict], output_path: str) -> None:
    n = len(entries)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), sharey=False)
    if n == 1:
        axes = [axes]

    for ax, entry in zip(axes, entries):
        label     = entry["label"]
        robust_ms = entry["robust_ms"]
        robust_prob = entry["robust_prob"]

        pgd_runs  = robust_ms["pgd_runs"]
        md_runs   = robust_ms["md_runs"]
        admm_runs = robust_ms["admm_runs"]

        pgd_best_idx  = min(range(len(pgd_runs)),  key=lambda i: pgd_runs[i]["objective"])
        md_best_idx   = min(range(len(md_runs)),   key=lambda i: md_runs[i]["objective"])
        admm_best_idx = min(range(len(admm_runs)), key=lambda i: admm_runs[i]["objective"])

        for i, run in enumerate(pgd_runs):
            iters = [h["iter"] for h in run["history"]]
            objs  = [h["objective"] for h in run["history"]]
            kw = {"color": "tomato", "linewidth": 2.0, "alpha": 1.0, "zorder": 3, "label": "PGD (best)"} \
                if i == pgd_best_idx else \
                {"color": "tomato", "linewidth": 0.8, "alpha": 0.25, "zorder": 2}
            ax.plot(iters, objs, **kw)

        for i, run in enumerate(md_runs):
            iters = [h["iter"] for h in run["history"]]
            objs  = [h["objective"] for h in run["history"]]
            kw = {"color": "seagreen", "linewidth": 2.0, "alpha": 1.0, "zorder": 3, "label": "MD (best)"} \
                if i == md_best_idx else \
                {"color": "seagreen", "linewidth": 0.8, "alpha": 0.25, "zorder": 2}
            ax.plot(iters, objs, **kw)

        for i, run in enumerate(admm_runs):
            iters = [h["iter"] for h in run["history"]]
            objs  = [h["objective"] for h in run["history"]]
            kw = {"color": "purple", "linewidth": 2.0, "alpha": 1.0, "zorder": 3, "label": "ADMM (best)"} \
                if i == admm_best_idx else \
                {"color": "purple", "linewidth": 0.8, "alpha": 0.25, "zorder": 2}
            ax.plot(iters, objs, **kw)

        equal_delta = np.full(robust_prob.num_intervals, robust_prob.total_useful_work / robust_prob.num_intervals)
        equal_robust = robust_prob.objective_from_delta(equal_delta)
        ax.axhline(equal_robust, color="steelblue", linestyle="--", linewidth=1.0,
                   label=f"Equal Intervals ({equal_robust:.2f})")

        all_runs = pgd_runs + md_runs + admm_runs
        max_plotted = max(h["objective"] for run in all_runs for h in run["history"])
        best_obj    = entry["best_obj"]
        #ax.set_ylim(best_obj * 0.95, min(equal_robust * 1.5, max_plotted * 1.05))
        ax.set_title(label, fontsize=10)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Objective Value (s)")
        ax.legend(fontsize=7, framealpha=0.95)

    fig.suptitle(f"PGD, MD, and ADMM Convergence by Hazard Type ({len(entries[0]['robust_ms']['pgd_runs'])} starts each)", fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")


def plot_hazard_with_bands_and_checkpoints(
    label: str,
    robust_ms: dict,
    robust_prob: Any,
    equal_obj: float,
    best_obj: float,
    output_path: str,
    color: str = "steelblue",
) -> None:
    os.makedirs("figures/robust", exist_ok=True)

    nom_fn, worst_fn, best_fn = _get_lambda_fns(robust_prob)
    total_time   = robust_prob.total_useful_work
    t_values     = np.linspace(0.0, total_time, 1000)
    nom_values   = np.array([nom_fn(t)   * SECONDS_PER_HOUR for t in t_values], dtype=float)
    worst_values = np.array([worst_fn(t) * SECONDS_PER_HOUR for t in t_values], dtype=float)
    best_values  = np.array([best_fn(t)  * SECONDS_PER_HOUR for t in t_values], dtype=float)

    optimal_checkpoints = robust_prob.delta_to_knots(robust_ms["best_delta"])[1:-1] / SECONDS_PER_HOUR
    band_edges = np.concatenate(([0.0], optimal_checkpoints, [total_time / SECONDS_PER_HOUR]))

    fig, ax = plt.subplots(figsize=(9, 4.5))

    for idx in range(len(band_edges) - 1):
        ax.axvspan(
            band_edges[idx], band_edges[idx + 1],
            facecolor="lightgray" if idx % 2 == 1 else "white",
            alpha=0.18 if idx % 2 == 1 else 0.0,
            zorder=0,
        )

    ax.fill_between(t_values / SECONDS_PER_HOUR, best_values, worst_values,
                    color=color, alpha=0.18, label="Uncertainty region")
    ax.plot(t_values / SECONDS_PER_HOUR, worst_values, color=color, linewidth=1.0,
            linestyle="--", alpha=0.7, label="Worst-case")
    ax.plot(t_values / SECONDS_PER_HOUR, best_values, color=color, linewidth=1.0,
            linestyle=":", alpha=0.7, label="Best-case")
    ax.plot(t_values / SECONDS_PER_HOUR, nom_values, color=color, linewidth=2, label="Nominal hazard")

    for idx, checkpoint in enumerate(optimal_checkpoints):
        ax.axvline(
            checkpoint, color="darkgreen", linestyle="--", linewidth=1.2, alpha=0.9,
            label="Optimal checkpoints" if idx == 0 else None,
        )

    ax.set_xlabel("Useful work completed (h)")
    ax.set_ylabel("Hazard rate $\\lambda(t)$")
    ax.set_title(f"{label} Hazard (Robust) with Optimal Checkpoints and Uncertainty Band")
    ax.set_xlim(0.0, total_time / SECONDS_PER_HOUR)
    ax.set_ylim(bottom=0.0)

    improvement_pct = 100.0 * (equal_obj - best_obj) / equal_obj
    legend_handles = [
        Line2D([0], [0], color=color, linewidth=2, label=label),
        Line2D([0], [0], color="darkgreen", linestyle="--", linewidth=1.2, label="Optimal checkpoints"),
        Line2D([0], [0], color="none", label=f"Best objective: {best_obj:.2f} s"),
        Line2D([0], [0], color="none", label=f"Equal intervals objective: {equal_obj:.2f} s"),
        Line2D([0], [0], color="none", label=f"Improvement: {improvement_pct:.2f}%"),
    ]
    ax.legend(handles=legend_handles, loc="best", framealpha=0.95)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")


def _make_true_cost_fn(prob: Any):
    """Return cost_fn(delta, theta) -> float: true cost of delta under a specific theta."""
    if isinstance(prob, RobustStepHazardProblem):
        def cost_fn(delta: np.ndarray, theta: np.ndarray) -> float:
            lam = make_step_lambda_fn(prob.tau, np.maximum(theta, 0.0))
            p = UsefulWorkHazardProblem(
                prob.total_useful_work, prob.num_intervals, prob.epsilon, lam, prob.q
            )
            return p.objective_from_delta(delta)
        return cost_fn

    if isinstance(prob, RobustPolynomialHazardProblem):
        T_ = prob.T
        js = np.arange(len(prob.theta_hat), dtype=float)

        def cost_fn(delta: np.ndarray, theta: np.ndarray) -> float:
            th = np.asarray(theta, dtype=float)

            def lam(t):
                t = np.asarray(t, dtype=float)
                scalar = t.ndim == 0
                t = np.atleast_1d(t)
                result = np.maximum((t[:, np.newaxis] / T_) ** js @ th, 0.0)
                return float(result[0]) if scalar else result

            p = UsefulWorkHazardProblem(
                prob.total_useful_work, prob.num_intervals, prob.epsilon, lam, prob.q
            )
            return p.objective_from_delta(delta)
        return cost_fn

    if isinstance(prob, RobustPowerLawHazardProblem):
        T_ = prob.T

        def cost_fn(delta: np.ndarray, theta: np.ndarray) -> float:
            th0 = max(float(theta[0]), 0.0)
            th1 = float(theta[1])

            def lam(t):
                t = np.asarray(t, dtype=float)
                scalar = t.ndim == 0
                t = np.atleast_1d(np.maximum(t, 1e-12))
                result = th0 * (t / T_) ** th1
                return float(result[0]) if scalar else result

            p = UsefulWorkHazardProblem(
                prob.total_useful_work, prob.num_intervals, prob.epsilon, lam, prob.q
            )
            return p.objective_from_delta(delta)
        return cost_fn

    raise NotImplementedError(f"_make_true_cost_fn: unsupported type {type(prob)}")


def _sample_theta(prob: Any, n_mc: int, rng: np.random.Generator) -> np.ndarray:
    """Sample theta uniformly from the uncertainty box [theta_hat - rho, theta_hat + rho]."""
    n = len(prob.theta_hat)
    u = rng.uniform(-1.0, 1.0, (n_mc, n))
    samples = prob.theta_hat + u * prob.rho
    if isinstance(prob, RobustStepHazardProblem):
        samples = np.maximum(samples, 0.0)
    elif isinstance(prob, RobustPowerLawHazardProblem):
        samples[:, 0] = np.maximum(samples[:, 0], 0.0)
        samples[:, 1] = np.maximum(samples[:, 1], -1.0 + 1e-6)
    return samples


def run_mc_nominal_robust_comparison(
    label: str,
    robust_delta: np.ndarray,
    n_mc: int = 500,
    n_starts: int = 10,
    seed: int = 42,
    pgd_kwargs: dict | None = None,
    md_kwargs: dict | None = None,
    admm_kwargs: dict | None = None,
) -> dict:
    """Optimize the nominal schedule, then evaluate all three schedules over n_mc sampled thetas."""
    pgd_kwargs  = pgd_kwargs  or {"max_iters": 2000, "step_size": 1e3,  "num_steps": 256}
    md_kwargs   = md_kwargs   or {"max_iters": 2000, "step_size": 1e-2, "num_steps": 256}
    admm_kwargs = admm_kwargs or {"max_iters": 500,  "rho":  5e-4,      "num_steps": 256}

    nominal_prob = _make_robust_problem(label, nominal=True)
    robust_prob  = _make_robust_problem(label, nominal=False)

    print(f"\n=== {label}: Optimizing NOMINAL schedule ===", flush=True)
    nominal_ms    = run_multistart(
        nominal_prob, n_starts=n_starts, seed=seed,
        pgd_kwargs=pgd_kwargs, md_kwargs=md_kwargs, admm_kwargs=admm_kwargs,
    )
    nominal_delta = nominal_ms["best_delta"]
    equal_delta   = np.full(robust_prob.num_intervals,
                            robust_prob.total_useful_work / robust_prob.num_intervals)

    rng     = np.random.default_rng(seed + 100)
    thetas  = _sample_theta(robust_prob, n_mc, rng)
    cost_fn = _make_true_cost_fn(robust_prob)

    print(f"=== {label}: Evaluating {n_mc} MC samples ===", flush=True)
    nominal_costs = np.array([cost_fn(nominal_delta, th) for th in thetas])
    robust_costs  = np.array([cost_fn(robust_delta,  th) for th in thetas])
    equal_costs   = np.array([cost_fn(equal_delta,   th) for th in thetas])

    return {
        "label":         label,
        "nominal_costs": nominal_costs,
        "robust_costs":  robust_costs,
        "equal_costs":   equal_costs,
        "thetas":        thetas,
    }


def plot_mc_comparison(entries: list[dict], output_path: str) -> None:
    """
    Violin + box plots comparing nominal, robust, and equal schedules over sampled thetas.

    Each panel shows one hazard family. The three groups per panel are the three scheduling
    policies; the distribution is the true expected rework cost evaluated over n_mc hazard
    parameters θ drawn uniformly from the uncertainty set.
    """
    n      = len(entries)
    colors = {"Nominal": "steelblue", "Robust": "tomato", "Equal": "gray"}
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), sharey=False)
    if n == 1:
        axes = [axes]

    for i, (ax, entry) in enumerate(zip(axes, entries)):
        label    = entry["label"]
        names    = ["Nominal", "Robust", "Equal"]
        # convert seconds → hours for readability
        datasets = [
            entry["nominal_costs"] / SECONDS_PER_HOUR,
            entry["robust_costs"]  / SECONDS_PER_HOUR,
            entry["equal_costs"]   / SECONDS_PER_HOUR,
        ]
        pos = [1, 2, 3]

        vp = ax.violinplot(datasets, positions=pos, showmedians=False, showextrema=False, widths=0.65)
        for pc, name in zip(vp["bodies"], names):
            pc.set_facecolor(colors[name])
            pc.set_alpha(0.35)

        ax.boxplot(
            datasets,
            positions=pos,
            widths=0.28,
            patch_artist=True,
            medianprops={"color": "black", "linewidth": 2},
            boxprops={"facecolor": "none", "edgecolor": "black", "linewidth": 1.2},
            whiskerprops={"linewidth": 1.2},
            capprops={"linewidth": 1.2},
            flierprops={"marker": ".", "markersize": 3, "alpha": 0.4, "color": "dimgray"},
            manage_ticks=False,
        )

        ax.set_xticks(pos)
        ax.set_xticklabels(names, fontsize=10)
        ax.set_title(label, fontsize=11)

        # only label y-axis on the leftmost subplot
        if i == 0:
            ax.set_ylabel("Expected rework cost (h)", fontsize=10)

        # tight y-limits: zoom in to where the data actually is
        all_vals = np.concatenate(datasets)
        ylo = all_vals.min()
        yhi = all_vals.max()
        margin = 0.15 * (yhi - ylo) if yhi > ylo else 0.1 * yhi
        ax.set_ylim(ylo - margin, yhi + margin)

        ax.tick_params(axis="y", labelsize=9)

    n_mc = len(entries[0]["nominal_costs"])
    fig.suptitle("Out-of-Sample Performance", fontsize=13, y=1.01)
    fig.text(
        0.5, 0.97,
        f"Distribution of expected rework cost over {n_mc} hazard parameters θ "
        f"drawn uniformly from the uncertainty set.\n"
        f"Nominal: schedule optimized at θ̂.  "
        f"Robust: schedule optimized for worst-case θ.  "
        f"Equal: uniform spacing.",
        ha="center", va="top", fontsize=9, color="dimgray",
        transform=fig.transFigure,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")


def main() -> None:
    os.makedirs("figures/robust", exist_ok=True)
    plot_robust_hazards()

    pgd_kwargs  = {"max_iters": 2000, "step_size": 1e3,  "num_steps": 256}
    md_kwargs   = {"max_iters": 2000, "step_size": 1e-2, "num_steps": 256}
    admm_kwargs = {"max_iters": 1000,  "rho": 5e-4,       "num_steps": 256}

    starts_by_label = {"Step": 20, "Polynomial": 10, "Power-law": 10}
    combined_entries = []

    for label in ["Step", "Polynomial", "Power-law"]:
        entry = run_robust_comparison(
            label, n_starts=starts_by_label[label],
            pgd_kwargs=pgd_kwargs, md_kwargs=md_kwargs, admm_kwargs=admm_kwargs,
        )

        label_key = label.lower().replace("-", "").replace(" ", "_")
        color = {"Step": "steelblue", "Polynomial": "tomato", "Power-law": "seagreen"}[label]

        plot_hazard_with_bands_and_checkpoints(
            entry["label"],
            entry["robust_ms"], entry["robust_prob"],
            entry["equal_obj"], entry["best_obj"],
            f"figures/robust/{label_key}_hazard_robust.png",
            color=color,
        )
        plot_convergence_robust(
            entry["label"],
            entry["robust_ms"]["pgd_runs"],
            entry["robust_ms"]["md_runs"],
            entry["robust_ms"]["admm_runs"],
            entry["equal_obj"],
            entry["best_obj"],
            f"figures/robust/{label_key}_convergence_robust.png",
        )
        combined_entries.append(entry)

    plot_convergence_combined_robust(combined_entries, "figures/robust/convergence_combined_robust.png")

    mc_entries = []
    for entry in combined_entries:
        mc_entry = run_mc_nominal_robust_comparison(
            entry["label"],
            robust_delta=entry["robust_ms"]["best_delta"],
            n_mc=500, n_starts=10,
            pgd_kwargs=pgd_kwargs, md_kwargs=md_kwargs, admm_kwargs=admm_kwargs,
        )
        mc_entries.append(mc_entry)
    plot_mc_comparison(mc_entries, "figures/robust/mc_comparison.png")


if __name__ == "__main__":
    main()
