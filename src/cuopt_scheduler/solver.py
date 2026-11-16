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
    """PuLP's stop reason (e.g. "Optimal", "TimeLimit"), for diagnostics."""
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

    PuLP's adapter (still as of 4.2) drops every extra parameter except
    `optimality_tolerance`, and maps `gapRel` to cuOpt's LP-only `relative_gap_tolerance`,
    so a MIP gap never reaches cuOpt. Only `callSolver` is replaced; check on PuLP upgrades
    whether this is still needed.
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
        solution = solver.Solve(lp.solverModel, settings)
        self.solveTime = time.perf_counter() - start
        return solution


def _solver(backend: Backend, time_limit: float, gap_rel: float | None) -> pulp.LpSolver:
    if backend is Backend.GPU:
        params = {} if gap_rel is None else {"mip_relative_gap": gap_rel}
        return _CUOPT(params, msg=False, timeLimit=time_limit)
    return pulp.COIN_CMD(msg=False, timeLimit=time_limit, gapRel=gap_rel)


# Stop reasons that mean "proven optimal", possibly within the requested gap.
_PROVEN = {pulp.LpSolveStatus.Optimal, pulp.LpSolveStatus.GapLimit}


def classify(lp: pulp.LpProblem, stats: pulp.LpSolveStats, tol: float = 1e-6) -> Outcome:
    """Turn PuLP's solve stats into an Outcome.

    A returned solution is also checked against the model (values present, integral, every
    constraint holds), so a solver-adapter bug can't pass off an invalid schedule.
    """
    if not stats.has_solution:
        return Outcome.NO_SOLUTION
    for v in lp.variables():
        value = v.varValue
        if value is None:
            return Outcome.NO_SOLUTION
        if v.cat == pulp.LpInteger and abs(value - round(value)) > tol:
            return Outcome.NO_SOLUTION
    if not lp.valid(tol):
        return Outcome.NO_SOLUTION
    return Outcome.OPTIMAL if stats.status in _PROVEN else Outcome.FEASIBLE


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
        stats = model.lp.solve(_solver(backend, time_limit, gap_rel))
    except Exception as exc:
        if backend is Backend.CPU:
            raise
        warnings.append(f"cuOpt failed ({exc}); solved on CPU instead.")
        backend = Backend.CPU
        start = time.perf_counter()
        stats = model.lp.solve(_solver(backend, time_limit, gap_rel))
    elapsed = time.perf_counter() - start

    outcome = classify(model.lp, stats)
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
        solver_status=stats.status.name,
        backend=backend,
        solve_seconds=elapsed,
        objective=objective,
        assignments=assignments,
        warnings=tuple(warnings),
    )
