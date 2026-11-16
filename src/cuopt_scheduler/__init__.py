"""Nurse rostering demo: synthetic data, a PuLP MILP model, and CPU/GPU (cuOpt) solving."""

from cuopt_scheduler.data import Employee, Problem, generate_problem
from cuopt_scheduler.model import build_model
from cuopt_scheduler.solver import Assignment, Backend, SolveResult, cuopt_available, solve
from cuopt_scheduler.validate import violations

__all__ = [
    "Assignment",
    "Backend",
    "Employee",
    "Problem",
    "SolveResult",
    "build_model",
    "cuopt_available",
    "generate_problem",
    "solve",
    "violations",
]
