"""Solve a problem on CPU (CBC) or GPU (cuOpt), falling back to CPU when needed."""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum

import pulp

from cuopt_scheduler.data import Assignment, Problem
from cuopt_scheduler.model import build_model

__all__ = ["Assignment", "Backend", "Outcome", "SolveResult", "cuopt_available", "solve"]


class Backend(StrEnum):
    CPU = "cpu"
    GPU = "gpu"


class Outcome(StrEnum):
    OPTIMAL = "optimal"
    """Solver proved this is the best schedule (within the requested gap)."""
    FEASIBLE = "feasible"
    """A valid schedule, but the time limit hit before it was proven best."""
    NO_SOLUTION = "no_solution"


@dataclass(frozen=True)
class SolveResult:
    outcome: Outcome
    solver_status: str
    """Raw PuLP status string, for diagnostics."""
    backend: Backend
    """The backend that actually produced the result."""
    solve_seconds: float
    objective: float | None
    assignments: tuple[Assignment, ...]
    warnings: tuple[str, ...] = ()

    @property
    def feasible(self) -> bool:
        return self.outcome is not Outcome.NO_SOLUTION


def cuopt_available() -> bool:
    """True if PuLP can actually run cuOpt here. Importing pulp.CUOPT is not enough."""
    try:
        return bool(pulp.CUOPT(msg=False).available())
    except Exception:
        return False


class _CUOPT(pulp.CUOPT):
    """pulp.CUOPT that passes cuOpt parameters through.

    PuLP 3.3's adapter drops every extra parameter except `optimality_tolerance`, and maps
    `gapRel` to cuOpt's LP-only `relative_gap_tolerance`, so a MIP gap never reaches cuOpt.
    Only `callSolver` is replaced; revisit when moving to PuLP 4.
    """

    def __init__(
        self, cuopt_params: Mapping[str, object], *, msg: bool, timeLimit: float | None
    ) -> None:
        super().__init__(msg=msg, timeLimit=timeLimit)
        self.cuopt_params = dict(cuopt_params)

    def callSolver(self, lp: pulp.LpProblem, callback: object = None) -> object:
        from cuopt.linear_programming import solver, solver_settings  # pyright: ignore[reportMissingImports]

        settings = solver_settings.SolverSettings()
        settings.set_parameter("infeasibility_detection", True)
        settings.set_parameter("log_to_console", self.msg)
        if self.timeLimit:
            settings.set_parameter("time_limit", self.timeLimit)
        for key, value in self.cuopt_params.items():
            settings.set_parameter(key, value)
        start = time.perf_counter()
        solution = solver.Solve(lp.solverModel, settings)  # pyright: ignore[reportAttributeAccessIssue]
        self.solveTime = time.perf_counter() - start
        return solution


def _solver(backend: Backend, time_limit: float, gap_rel: float | None) -> pulp.LpSolver:
    if backend is Backend.GPU:
        params = {} if gap_rel is None else {"mip_relative_gap": gap_rel}
        return _CUOPT(params, msg=False, timeLimit=time_limit)
    # PuLP 3.x re-exports via `import *` and a Rust core, which pyright cannot follow.
    return pulp.PULP_CBC_CMD(msg=False, timeLimit=time_limit, gapRel=gap_rel)  # pyright: ignore[reportAttributeAccessIssue]


def classify(lp: pulp.LpProblem, tol: float = 1e-6) -> Outcome:
    """Classify a solved model without trusting solver-specific status codes.

    CBC reports a time-limited run with a solution as "Optimal" (only `sol_status` says it
    isn't proven), and PuLP's cuOpt adapter reports it as "Not Solved" while still loading
    the values. So: a result is usable if every variable has an integral value and every
    constraint holds; it is optimal only if the solver also proved optimality.
    """
    for v in lp.variables():
        value = v.varValue
        if value is None:
            return Outcome.NO_SOLUTION
        if v.cat == pulp.LpInteger and abs(value - round(value)) > tol:
            return Outcome.NO_SOLUTION
    # PuLP annotates eps as int, but it is compared against float slacks.
    if not lp.valid(tol):  # pyright: ignore[reportArgumentType]
        return Outcome.NO_SOLUTION
    proven = lp.status == pulp.LpStatusOptimal and lp.sol_status == pulp.LpSolutionOptimal  # pyright: ignore[reportAttributeAccessIssue]
    return Outcome.OPTIMAL if proven else Outcome.FEASIBLE


def solve(
    problem: Problem,
    backend: Backend = Backend.CPU,
    time_limit: float = 30,
    gap_rel: float | None = None,
) -> SolveResult:
    """Solve `problem`. `gap_rel` lets the solver stop once it is provably within that
    fraction of the best possible objective (None = solver default); the result then
    counts as optimal."""
    model = build_model(problem)
    warnings: list[str] = []

    if backend is Backend.GPU and not cuopt_available():
        warnings.append("cuOpt is not available; solved on CPU instead.")
        backend = Backend.CPU

    start = time.perf_counter()
    try:
        model.lp.solve(_solver(backend, time_limit, gap_rel))
    except Exception as exc:
        if backend is Backend.CPU:
            raise
        warnings.append(f"cuOpt failed ({exc}); solved on CPU instead.")
        backend = Backend.CPU
        start = time.perf_counter()
        model.lp.solve(_solver(backend, time_limit, gap_rel))
    elapsed = time.perf_counter() - start

    outcome = classify(model.lp)
    assignments: tuple[Assignment, ...] = ()
    objective = None
    if outcome is not Outcome.NO_SOLUTION:
        assignments = tuple(
            sorted(
                Assignment(day=d, shift=s, employee_id=emp)
                for (emp, s, d), var in model.x.items()
                if (var.varValue or 0) > 0.5
            )
        )
        objective = model.lp.objective.value() if model.lp.objective else None
    return SolveResult(
        outcome=outcome,
        solver_status=pulp.LpStatus[model.lp.status],  # pyright: ignore[reportAttributeAccessIssue]
        backend=backend,
        solve_seconds=elapsed,
        objective=objective,
        assignments=assignments,
        warnings=tuple(warnings),
    )
