"""
Robust checkpoint scheduling package.

Public API
----------
Problems:
    RobustStepHazardProblem
    RobustPolynomialHazardProblem
    RobustPowerLawHazardProblem
    make_step_lambda_fn

Projection:
    project_internal_knots_pav
    project_internal_knots_cvxpy

Optimizers:
    optimize_pgd_internal_knots
    optimize_mirror_descent
    optimize_admm

Multi-start:
    _diverse_inits
    _auto_pgd_step, _auto_md_step, _auto_admm_rho
    _best_pgd, _best_md, _best_admm

Monte Carlo:
    monte_carlo_uncertain_theta

Experiments:
    run_comparison
    main
"""

from .problems import (
    RobustStepHazardProblem,
    RobustPolynomialHazardProblem,
    RobustPowerLawHazardProblem,
    make_step_lambda_fn,
)
from .projection import project_internal_knots_pav, project_internal_knots_cvxpy
from .optimizers import optimize_pgd_internal_knots, optimize_mirror_descent, optimize_admm
from .multistart import (
    _diverse_inits,
    _auto_pgd_step, _auto_md_step, _auto_admm_rho,
    _best_pgd, _best_md, _best_admm,
)
from .monte_carlo import monte_carlo_uncertain_theta, monte_carlo_schedule_useful_work_hazard
from .run_experiments import run_comparison, main

__all__ = [
    "RobustStepHazardProblem",
    "RobustPolynomialHazardProblem",
    "RobustPowerLawHazardProblem",
    "make_step_lambda_fn",
    "project_internal_knots_pav",
    "project_internal_knots_cvxpy",
    "optimize_pgd_internal_knots",
    "optimize_mirror_descent",
    "optimize_admm",
    "_diverse_inits",
    "_auto_pgd_step",
    "_auto_md_step",
    "_auto_admm_rho",
    "_best_pgd",
    "_best_md",
    "_best_admm",
    "monte_carlo_uncertain_theta",
    "monte_carlo_schedule_useful_work_hazard",
    "run_comparison",
    "main",
]
