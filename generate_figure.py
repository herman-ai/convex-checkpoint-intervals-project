from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).parent / "robust"))
from utils.helpers import SECONDS_PER_HOUR, UsefulWorkHazardProblem  # noqa: E402
from optimizers import optimize_pgd_internal_knots, optimize_mirror_descent, optimize_admm  # noqa: E402


def lambda_fn_step(t: float) -> float:
    if t < 16 * SECONDS_PER_HOUR:
        return 1.0 / (48 * SECONDS_PER_HOUR)
    if t < 32 * SECONDS_PER_HOUR:
        return 1.0 / (18 * SECONDS_PER_HOUR)
    return 1.0 / (6 * SECONDS_PER_HOUR)


def lambda_fn_polynomial(t: float) -> float:
    total_time = 48.0 * SECONDS_PER_HOUR
    alpha = min(max(t / total_time, 0.0), 1.0)
    base = 1.0 / (38 * SECONDS_PER_HOUR)
    curvature = 1.0 / (2 * SECONDS_PER_HOUR)
    return base + curvature * (alpha - 0.5) ** 2


def lambda_fn_power_law(t: float) -> float:
    total_time = 48.0 * SECONDS_PER_HOUR
    theta0 = 1.0 / (8 * SECONDS_PER_HOUR)
    theta1 = 4.0
    return theta0 * (max(t, 0.0) / total_time) ** theta1


def _make_problem(lambda_fn: Callable[[float], float]) -> UsefulWorkHazardProblem:
    total_time = 48.0 * SECONDS_PER_HOUR
    return UsefulWorkHazardProblem(
        total_useful_work=total_time,
        num_intervals=10,
        epsilon=0.5 * SECONDS_PER_HOUR,
        lambda_fn=lambda_fn,
        q=np.full(10, 10 * 60, dtype=float),
    )


def run_multistart(
    problem: UsefulWorkHazardProblem,
    n_starts: int = 20,
    seed: int = 42,
    pgd_kwargs: dict | None = None,
    md_kwargs: dict | None = None,
    admm_kwargs: dict | None = None,
) -> dict:
    """Run n_starts random initializations of PGD, MD, and ADMM; return all runs and global best."""
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


def plot_hazard_with_checkpoints(
    label: str,
    lambda_fn: Callable[[float], float],
    output_path: str,
    n_starts: int = 20,
    pgd_kwargs: dict | None = None,
    md_kwargs: dict | None = None,
    admm_kwargs: dict | None = None,
) -> dict:
    """Plot hazard rate with optimal checkpoint locations from the multistart best solution."""
    os.makedirs("figures", exist_ok=True)

    total_time = 48.0 * SECONDS_PER_HOUR
    t_values = np.linspace(0.0, total_time, 1000)
    lambda_values = np.array([lambda_fn(t) * SECONDS_PER_HOUR for t in t_values], dtype=float)

    problem = _make_problem(lambda_fn)

    print(f"Running {n_starts} random starts for '{label}'...")
    ms = run_multistart(
        problem, n_starts=n_starts,
        pgd_kwargs=pgd_kwargs, md_kwargs=md_kwargs, admm_kwargs=admm_kwargs,
    )

    equal_delta = np.full(10, total_time / 10)
    equal_obj = problem.objective_from_delta(equal_delta, num_steps=256)

    optimal_checkpoints = problem.delta_to_knots(ms["best_delta"])[1:-1] / SECONDS_PER_HOUR
    improvement_pct = 100.0 * (equal_obj - ms["best_obj"]) / equal_obj

    band_edges = np.concatenate(([0.0], optimal_checkpoints, [total_time / SECONDS_PER_HOUR]))

    fig, ax = plt.subplots(figsize=(9, 4.5))

    for idx in range(len(band_edges) - 1):
        ax.axvspan(
            band_edges[idx],
            band_edges[idx + 1],
            facecolor="lightgray" if idx % 2 == 1 else "white",
            alpha=0.18 if idx % 2 == 1 else 0.0,
            zorder=0,
        )

    ax.plot(t_values / SECONDS_PER_HOUR, lambda_values, color="steelblue", linewidth=2, label=label)

    for idx, checkpoint in enumerate(optimal_checkpoints):
        ax.axvline(
            checkpoint,
            color="darkgreen",
            linestyle="--",
            linewidth=1.2,
            alpha=0.9,
            label="Optimal checkpoints" if idx == 0 else None,
        )

    ax.set_xlabel("Useful work completed (h)")
    ax.set_ylabel("Hazard rate $\\lambda(t)$")
    ax.set_title(f"{label} Hazard with Optimal Checkpoints")
    ax.set_xlim(0.0, total_time / SECONDS_PER_HOUR)
    ax.set_ylim(bottom=0.0)

    legend_handles = [
        Line2D([0], [0], color="steelblue", linewidth=2, label=label),
        Line2D([0], [0], color="darkgreen", linestyle="--", linewidth=1.2, label="Optimal checkpoints"),
        Line2D([0], [0], color="none", label=f"Optimized objective: {ms['best_obj']:.2f}"),
        Line2D([0], [0], color="none", label=f"Equal intervals objective: {equal_obj:.2f}"),
        Line2D([0], [0], color="none", label=f"Improvement: {improvement_pct:.2f}%"),
    ]
    ax.legend(handles=legend_handles, loc="best", framealpha=0.95)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")

    ms["equal_obj"] = equal_obj
    return ms


def plot_convergence(
    label: str,
    pgd_runs: list[dict],
    md_runs: list[dict],
    admm_runs: list[dict],
    equal_obj: float,
    best_obj: float,
    output_path: str,
) -> None:
    """Plot convergence for all multistart runs, bolding the best run per algorithm."""
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


def plot_convergence_combined(
    entries: list[tuple[str, list, list, list, float, float]],
    output_path: str,
) -> None:
    """entries: list of (label, pgd_runs, md_runs, admm_runs, equal_obj, best_obj)."""
    n = len(entries)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), sharey=False)
    if n == 1:
        axes = [axes]

    for ax, (label, pgd_runs, md_runs, admm_runs, equal_obj, best_obj) in zip(axes, entries):
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

        ax.axhline(equal_obj, color="steelblue", linestyle="--", linewidth=1.0, label="Equal Intervals")
        ax.axhline(best_obj, color="darkgreen", linestyle=":", linewidth=1.0, label="Best Optimized")
        all_runs = pgd_runs + md_runs + admm_runs
        max_plotted = max(h["objective"] for run in all_runs for h in run["history"])
        #ax.set_ylim(best_obj * 0.98, min(equal_obj * 1.5, max_plotted * 1.05))
        ax.set_title(label, fontsize=10)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Objective Value (s)")
        ax.legend(fontsize=8)

    fig.suptitle(f"PGD, MD, and ADMM Convergence by Hazard Type ({len(entries[0][1])} starts each)", fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")


def plot_K_sweep(
    c: float,
    pgd_kwargs: dict | None = None,
    md_kwargs: dict | None = None,
    admm_kwargs: dict | None = None,
    output_path: str = "figures/K_sweep.png",
) -> dict[str, int]:
    """Plot optimal objective value vs. number of checkpoints K. Returns optimal K per hazard."""
    os.makedirs("figures", exist_ok=True)
    pgd_kwargs = pgd_kwargs or {"max_iters": 2000, "step_size": 1e3, "num_steps": 256}
    md_kwargs = md_kwargs or {"max_iters": 2000, "step_size": 1e-2, "num_steps": 256}
    admm_kwargs = admm_kwargs or {"max_iters": 500, "rho": 5e-4, "num_steps": 256}

    K_values = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
    total_time = 48.0 * SECONDS_PER_HOUR

    hazard_configs = [
        ("Step",       lambda_fn_step,       "steelblue"),
        ("Polynomial", lambda_fn_polynomial, "tomato"),
        ("Power-law",  lambda_fn_power_law,  "seagreen"),
    ]

    fig, ax = plt.subplots(figsize=(8, 5))
    optimal_K: dict[str, int] = {}

    for label, lambda_fn, color in hazard_configs:
        objectives = []
        print(f"\nK sweep — {label}:")
        for K in K_values:
            problem = UsefulWorkHazardProblem(
                total_useful_work=total_time,
                num_intervals=K,
                epsilon=0.5 * SECONDS_PER_HOUR,
                lambda_fn=lambda_fn,
                q=np.full(K, 10 * 60, dtype=float),
                checkpoint_costs=np.full(K, c, dtype=float),
            )
            pgd_result  = optimize_pgd_internal_knots(problem, **pgd_kwargs)
            md_result   = optimize_mirror_descent(problem, **md_kwargs)
            admm_result = optimize_admm(problem, **admm_kwargs)
            best_obj = min(pgd_result["objective"], md_result["objective"], admm_result["objective"])
            objectives.append(best_obj)
            print(f"  K={K:2d}: obj={best_obj:.4f}  (pgd={pgd_result['objective']:.4f}  md={md_result['objective']:.4f}  admm={admm_result['objective']:.4f})", flush=True)

        ax.plot(K_values, objectives, marker="o", label=label, color=color, linewidth=2)

        best_idx = int(np.argmin(objectives))
        K_star = K_values[best_idx]
        optimal_K[label] = K_star
        ax.axvline(K_star, color=color, linestyle="--", linewidth=1.5, alpha=0.75)
        print(f"  → K*={K_star} (obj={objectives[best_idx]:.4f})")

    ax.set_xlabel("Number of checkpoints $K$")
    ax.set_ylabel("Optimal objective value (s)")
    ax.set_title("Optimal Objective vs. Number of Checkpoints")
    ax.set_xticks(K_values)
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")
    return optimal_K


def plot_hazard_optimal_K(
    label: str,
    lambda_fn: Callable[[float], float],
    K_star: int,
    c: float,
    output_path: str,
    n_starts: int = 10,
    pgd_kwargs: dict | None = None,
    md_kwargs: dict | None = None,
    admm_kwargs: dict | None = None,
) -> None:
    """Plot hazard with optimal checkpoint locations at the K that minimises total cost."""
    os.makedirs("figures", exist_ok=True)
    pgd_kwargs = pgd_kwargs or {}
    md_kwargs = md_kwargs or {}
    admm_kwargs = admm_kwargs or {}

    total_time = 48.0 * SECONDS_PER_HOUR

    problem = UsefulWorkHazardProblem(
        total_useful_work=total_time,
        num_intervals=K_star,
        epsilon=0.5 * SECONDS_PER_HOUR,
        lambda_fn=lambda_fn,
        q=np.full(K_star, 10 * 60, dtype=float),
        checkpoint_costs=np.full(K_star, c, dtype=float),
    )

    print(f"Running {n_starts} random starts for '{label}' at K*={K_star}...")
    ms = run_multistart(
        problem, n_starts=n_starts,
        pgd_kwargs=pgd_kwargs, md_kwargs=md_kwargs, admm_kwargs=admm_kwargs,
    )

    t_values = np.linspace(0.0, total_time, 1000)
    lambda_values = np.array([lambda_fn(t) * SECONDS_PER_HOUR for t in t_values], dtype=float)

    equal_delta = np.full(K_star, total_time / K_star)
    equal_obj = problem.objective_from_delta(equal_delta, num_steps=256)

    optimal_checkpoints = problem.delta_to_knots(ms["best_delta"])[1:-1] / SECONDS_PER_HOUR
    band_edges = np.concatenate(([0.0], optimal_checkpoints, [total_time / SECONDS_PER_HOUR]))

    fig, ax = plt.subplots(figsize=(9, 4.5))

    for idx in range(len(band_edges) - 1):
        ax.axvspan(
            band_edges[idx], band_edges[idx + 1],
            facecolor="lightgray" if idx % 2 == 1 else "white",
            alpha=0.18 if idx % 2 == 1 else 0.0,
            zorder=0,
        )

    ax.plot(t_values / SECONDS_PER_HOUR, lambda_values, color="steelblue", linewidth=2)

    for idx, cp in enumerate(optimal_checkpoints):
        ax.axvline(
            cp, color="darkgreen", linestyle="--", linewidth=1.2, alpha=0.9,
            label="Optimal checkpoints" if idx == 0 else None,
        )

    ax.set_xlabel("Useful work completed (h)")
    ax.set_ylabel("Hazard rate $\\lambda(t)$")
    ax.set_title(f"{label} Hazard — Optimal Checkpoints at $K^*={K_star}$")
    ax.set_xlim(0.0, total_time / SECONDS_PER_HOUR)
    ax.set_ylim(bottom=0.0)

    improvement_pct = 100.0 * (equal_obj - ms["best_obj"]) / equal_obj
    legend_handles = [
        Line2D([0], [0], color="steelblue", linewidth=2, label=label),
        Line2D([0], [0], color="darkgreen", linestyle="--", linewidth=1.2,
               label=f"Optimal checkpoints ($K^*={K_star}$)"),
        Line2D([0], [0], color="none", label=f"Best objective: {ms['best_obj']:.2f} s"),
        Line2D([0], [0], color="none", label=f"Equal intervals objective: {equal_obj:.2f} s"),
        Line2D([0], [0], color="none", label=f"Improvement: {improvement_pct:.2f}%"),
    ]
    ax.legend(handles=legend_handles, loc="best", framealpha=0.95)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")


def main() -> None:
    n_starts = 10
    pgd_kwargs  = {"max_iters": 2000, "step_size": 1e3,  "num_steps": 256}
    md_kwargs   = {"max_iters": 2000, "step_size": 1e-2, "num_steps": 256}
    admm_kwargs = {"max_iters": 1000,  "rho": 5e-4,       "num_steps": 256}
    c = 60 * 20  # 20-minute checkpoint write cost (seconds)

    configs = [
        ("Step",       lambda_fn_step,       "figures/step_hazard.png",       "figures/step_convergence.png"),
        ("Polynomial", lambda_fn_polynomial, "figures/polynomial_hazard.png", "figures/polynomial_convergence.png"),
        ("Power-law",  lambda_fn_power_law,  "figures/powerlaw_hazard.png",   "figures/powerlaw_convergence.png"),
    ]

    combined_entries = []
    for label, lambda_fn, hazard_path, conv_path in configs:
        ms = plot_hazard_with_checkpoints(
            label, lambda_fn, hazard_path,
            n_starts=n_starts, pgd_kwargs=pgd_kwargs, md_kwargs=md_kwargs, admm_kwargs=admm_kwargs,
        )
        plot_convergence(
            label, ms["pgd_runs"], ms["md_runs"], ms["admm_runs"],
            ms["equal_obj"], ms["best_obj"], conv_path,
        )
        combined_entries.append((
            label, ms["pgd_runs"], ms["md_runs"], ms["admm_runs"], ms["equal_obj"], ms["best_obj"],
        ))

    plot_convergence_combined(combined_entries, "figures/final/convergence_combined.png")

    optimal_K = plot_K_sweep(
        c=c, pgd_kwargs=pgd_kwargs, md_kwargs=md_kwargs, admm_kwargs=admm_kwargs,
        output_path="figures/K_sweep.png",
    )

    for label, lambda_fn, path in [
        ("Step",       lambda_fn_step,       "figures/step_optimal_K.png"),
        ("Polynomial", lambda_fn_polynomial, "figures/polynomial_optimal_K.png"),
        ("Power-law",  lambda_fn_power_law,  "figures/powerlaw_optimal_K.png"),
    ]:
        plot_hazard_optimal_K(
            label, lambda_fn, optimal_K[label], c, path,
            n_starts=1, pgd_kwargs=pgd_kwargs, md_kwargs=md_kwargs, admm_kwargs=admm_kwargs,
        )


if __name__ == "__main__":
    main()
