"""Nurse rostering demo: synthetic data, a PuLP MILP model, and CPU/GPU (cuOpt) solving."""

from cuopt_scheduler.data import Assignment, Employee, Problem, generate_problem
from cuopt_scheduler.model import build_model
from cuopt_scheduler.solver import Backend, Outcome, SolveResult, cuopt_available, solve
from cuopt_scheduler.validate import coverage_gaps, violations

__all__ = [
    "Assignment",
    "Backend",
    "Employee",
    "Outcome",
    "Problem",
    "SolveResult",
    "build_model",
    "coverage_gaps",
    "cuopt_available",
    "generate_problem",
    "solve",
    "violations",
]
