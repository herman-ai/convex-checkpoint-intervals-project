"""
Experiment runner: constructs problems, runs PGD/MD/ADMM, prints tables, saves figures.

Run from the robust/ directory:  python run_experiments.py
"""

from __future__ import annotations

import os
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np

from simulator_v2 import monte_carlo_schedule_useful_work_hazard
from problems import (
    RobustStepHazardProblem,
    RobustPolynomialHazardProblem,
    RobustPowerLawHazardProblem,
    make_step_lambda_fn,
)
from multistart import (
    _best_pgd, _best_md, _best_admm,
    _auto_pgd_step, _auto_md_step, _auto_admm_rho,
)
from monte_carlo import monte_carlo_uncertain_theta


# ── experiment runner ─────────────────────────────────────────────────────────

def run_comparison(
    label: str,
    nominal_prob,
    robust_prob,
    lambda_nom,
    lambda_wc,
    K: int,
    q: np.ndarray,
    run_mc: bool = False,
    n_mc: int = 100,
) -> Dict:
    """Run PGD, MD, and ADMM on both problems; print table; return results dict."""
    sep = "─" * 65
    print(f"\n{sep}\n  {label}\n{sep}")

    equal_delta = np.full(K, nominal_prob.total_useful_work / K)

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
        print(
            f"\n  [{opt.upper():4s}]  Price of robustness: +{price:.1f}s (+{100*price/nom_nom:.2f}%)"
            f"   Worst-case savings: -{saving:.1f}s (-{100*saving/nom_wc:.2f}%)"
        )

    mc = {}
    if run_mc:
        print(f"\n  Monte Carlo ({n_mc} trials, nominal lambda / worst-case lambda):")
        for k in sched_keys:
            r_n = monte_carlo_schedule_useful_work_hazard(
                delta=deltas[k], lambda_fn=lambda_nom, q=q, num_trials=n_mc)
            r_w = monte_carlo_schedule_useful_work_hazard(
                delta=deltas[k], lambda_fn=lambda_wc,  q=q, num_trials=n_mc)
            mc[k] = (r_n, r_w)
            print(
                f"    {sched_lbl[k]}:  nom {r_n['mean_runtime']:.0f}s (\u00b1{r_n['stderr_runtime']:.0f})"
                f"   wc {r_w['mean_runtime']:.0f}s (\u00b1{r_w['stderr_runtime']:.0f})"
            )

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
        "equal_delta":  equal_delta,
    }


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    os.makedirs("figures", exist_ok=True)

    RUN_MC = False   # set True to run Monte Carlo (slow)
    K      = 8
    T      = 48 * 3600.0
    epsilon = 0.5 * 3600.0
    q = np.full(K, 10 * 60, dtype=float)
    N_MC = 100

    # ── 1. Step hazard (3-phase, ordering reversal) ───────────────────────────
    # At theta_hat phase ordering (safest→riskiest): 2 < 1 < 3
    #   theta_hat = [0.10, 0.05, 0.20] /h  → nominal concentrates work in phase 2 (16–32h)
    # At theta_worst phase ordering:          1 < 3 < 2  (complete reversal for phases 1&2)
    #   theta_worst ≈ [0.105, 0.30, 0.21] /h → robust concentrates work in phase 1 (0–16h)
    # Large uncertainty on phase 2 (rho = 500% of nominal) drives the reversal.
    tau      = np.array([0, 16*3600, 32*3600, 48*3600], dtype=float)
    th_step  = np.array([1/(10*3600), 1/(20*3600), 1/(5*3600)], dtype=float)
    rho_step = np.array([0.05, 5.0, 0.05]) * th_step
    step_res = run_comparison(
        "Step hazard  [3-phase ordering reversal; phase 2 safe nominally but risky at theta*]",
        RobustStepHazardProblem(T, K, epsilon, tau, th_step, np.zeros_like(th_step), q),
        RobustStepHazardProblem(T, K, epsilon, tau, th_step, rho_step, q),
        lambda_nom=make_step_lambda_fn(tau, th_step),
        lambda_wc =make_step_lambda_fn(tau, th_step + rho_step),
        K=K, q=q, run_mc=RUN_MC, n_mc=N_MC,
    )

    # ── 2. Polynomial hazard, degree 2 ───────────────────────────────────────
    # lambda(t) = a0 + a2*(t/T)^2; large uncertainty on a2 (late-job coefficient).
    # rho_a2=80% → worst-case a2 much steeper → robust front-loads more than nominal.
    th_poly  = np.array([1/(40*3600), 0.0, 3/(10*3600)], dtype=float)
    rho_poly = np.array([0.05, 0.0, 0.80]) * th_poly
    nom_poly = RobustPolynomialHazardProblem(T, K, epsilon, th_poly, np.zeros_like(th_poly), q)
    rob_poly = RobustPolynomialHazardProblem(T, K, epsilon, th_poly, rho_poly, q)
    poly_res = run_comparison(
        "Polynomial hazard, deg-2  [rho_a2 = 80%]",
        nom_poly, rob_poly,
        lambda_nom=nom_poly._worst_problem.lambda_fn,
        lambda_wc =rob_poly._worst_problem.lambda_fn,
        K=K, q=q, run_mc=RUN_MC, n_mc=N_MC,
    )

    # ── 3. Power-law hazard ───────────────────────────────────────────────────
    # lambda(t) = theta0*(t/T)^theta1; nominal strongly accelerating (theta1=2.5).
    # rho1=2.0 → worst-case theta1=0.5 (near-flat); rho0=30% scales up overall rate.
    th_pl  = np.array([1/(8*3600), 2.5], dtype=float)
    rho_pl = np.array([0.3 * th_pl[0], 2.0])
    nom_pl = RobustPowerLawHazardProblem(T, K, epsilon, th_pl, np.zeros_like(rho_pl), q)
    rob_pl = RobustPowerLawHazardProblem(T, K, epsilon, th_pl, rho_pl, q)
    pl_res = run_comparison(
        "Power-law hazard  [theta1=2.5, rho0=30%, rho1=2.0  ->  wc theta1=0.5]",
        nom_pl, rob_pl,
        lambda_nom=nom_pl._worst_problem.lambda_fn,
        lambda_wc =rob_pl._worst_problem.lambda_fn,
        K=K, q=q, run_mc=RUN_MC, n_mc=N_MC,
    )

    # ── plotting setup ────────────────────────────────────────────────────────
    all_results  = [step_res, poly_res, pl_res]
    short_labels = ["Step", "Polynomial", "Power-law"]
    sched_keys   = ["equal", "nom_pgd", "nom_md", "nom_admm", "rob_pgd", "rob_md", "rob_admm"]
    sched_disp   = ["equal", "nom\nPGD", "nom\nMD", "nom\nADMM", "rob\nPGD", "rob\nMD", "rob\nADMM"]
    bar_clrs7    = ["steelblue", "tomato", "#ff9999", "darkorange",
                    "seagreen",  "#90ee90", "#2e7d32"]
    k_idx = np.arange(K)
    bar_w = 0.18
    x7    = np.arange(7)

    # ── Plot 1: Schedules 3×2 ─────────────────────────────────────────────────
    fig1, axes1 = plt.subplots(3, 2, figsize=(14, 12))
    fig1.suptitle("Robust vs nominal optimal schedules (PGD, MD, ADMM)", fontsize=13)
    for row, (res, slabel) in enumerate(zip(all_results, short_labels)):
        equal_delta = res["equal_delta"]
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
            ax      = axes1[row, col]
            n_bars  = len(d_keys)
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

    # ── Plot 2: Convergence 3×2 ───────────────────────────────────────────────
    fig2, axes2 = plt.subplots(3, 2, figsize=(14, 12))
    fig2.suptitle("Convergence: objective vs iteration (PGD, MD, ADMM)", fontsize=13)
    for row, (res, slabel) in enumerate(zip(all_results, short_labels)):
        equal_delta = res["equal_delta"]
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

    # ── Plot 3: Analytic objective 3×2 ────────────────────────────────────────
    fig3, axes3 = plt.subplots(3, 2, figsize=(14, 12))
    fig3.suptitle("Analytic objective at theta_hat and theta*", fontsize=13)
    for row, (res, slabel) in enumerate(zip(all_results, short_labels)):
        for col, (theta_idx, theta_lbl) in enumerate([(0, "at theta_hat"), (1, "at theta*")]):
            ax   = axes3[row, col]
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

    # ── Plot 4: MC runtime 3×2 ────────────────────────────────────────────────
    if RUN_MC:
        fig4, axes4 = plt.subplots(3, 2, figsize=(14, 12))
        fig4.suptitle(f"Monte Carlo mean runtime ({N_MC} trials)", fontsize=13)
        for row, (res, slabel) in enumerate(zip(all_results, short_labels)):
            for col, (mc_idx, lam_lbl) in enumerate(
                [(0, "nominal lambda"), (1, "worst-case lambda")]
            ):
                ax   = axes4[row, col]
                vals = [res["mc"][k][mc_idx]["mean_runtime"]  for k in sched_keys]
                errs = [res["mc"][k][mc_idx]["stderr_runtime"] for k in sched_keys]
                ax.bar(x7, vals, yerr=errs, color=bar_clrs7, alpha=0.85, capsize=4)
                eq_val = vals[0]
                for xi in range(1, 7):
                    impr = 100 * (eq_val - vals[xi]) / eq_val
                    ax.text(xi, (vals[xi] + errs[xi]) * 1.02, f"{impr:+.1f}%",
                            ha="center", fontsize=7)
                ax.set_title(f"{slabel} — {lam_lbl}", fontsize=9)
                ax.set_xticks(x7); ax.set_xticklabels(sched_disp, fontsize=8)
                ax.set_ylabel("Mean runtime (s)")
        plt.tight_layout()
        plt.savefig("figures/mc_runtime_robust.png", dpi=150, bbox_inches="tight")
        print("Saved figures/mc_runtime_robust.png")


if __name__ == "__main__":
    main()
