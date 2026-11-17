"""Solve a problem on CPU (HiGHS) or GPU (cuOpt), falling back to CPU when needed.

Both solvers get the same arrays from `build_model`, through their own Python APIs.
"""

from __future__ import annotations

import contextlib
import functools
import os
import sys
import time
from collections.abc import Iterator
from dataclasses import dataclass
from enum import StrEnum

import highspy
import numpy as np

from cuopt_scheduler.data import Assignment, Problem
from cuopt_scheduler.model import FloatArray, ScheduleModel, build_model

__all__ = ["Assignment", "Backend", "Outcome", "SolveResult", "cuopt_available", "solve"]

# Every cost is an integer, so a bound within less than 1 of the best schedule proves it
# optimal. Both solvers get this, so neither keeps searching for a better score that can't exist.
_ABS_GAP = 0.999


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
    """The solver's own status, for diagnostics."""
    backend: Backend
    """The backend that actually produced the result."""
    solve_seconds: float
    """Time in the solver, including handing it the model."""
    objective: float | None
    assignments: tuple[Assignment, ...]
    warnings: tuple[str, ...] = ()
    build_seconds: float = 0.0
    """Time to build the model arrays (shared by both backends)."""

    @property
    def feasible(self) -> bool:
        return self.outcome is not Outcome.NO_SOLUTION


@dataclass(frozen=True)
class _Raw:
    values: FloatArray | None
    proven: bool
    """The solver proved this solution optimal (within the requested gap)."""
    status: str


@contextlib.contextmanager
def _quiet_stdout() -> Iterator[None]:
    """Silence C-level stdout: HiGHS prints a banner and cuOpt echoes every setting, even
    with logging off. Python's own prints in the block are lost too, so keep it narrow."""
    sys.stdout.flush()
    saved = os.dup(1)
    try:
        with open(os.devnull, "w") as devnull:
            os.dup2(devnull.fileno(), 1)
        yield
    finally:
        os.dup2(saved, 1)
        os.close(saved)


def _solve_highs(model: ScheduleModel, time_limit: float, gap_rel: float | None) -> _Raw:
    with _quiet_stdout():
        h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.setOptionValue("time_limit", float(time_limit))
    h.setOptionValue("mip_rel_gap", float(gap_rel or 0.0))
    h.setOptionValue("mip_abs_gap", _ABS_GAP)

    lp = highspy.HighsLp()
    lp.num_col_, lp.num_row_ = model.num_cols, model.num_rows
    lp.col_cost_ = model.cost
    lp.col_lower_, lp.col_upper_ = model.col_lower, model.col_upper
    lp.row_lower_, lp.row_upper_ = model.row_lower, model.row_upper
    lp.a_matrix_.format_ = highspy.MatrixFormat.kRowwise
    lp.a_matrix_.start_ = model.offsets
    lp.a_matrix_.index_ = model.indices
    lp.a_matrix_.value_ = model.values
    lp.integrality_ = [
        highspy.HighsVarType.kInteger if i else highspy.HighsVarType.kContinuous
        for i in model.integer
    ]
    h.passModel(lp)
    h.run()

    status = h.getModelStatus()
    has_solution = (
        h.getInfo().primal_solution_status == highspy.SolutionStatus.kSolutionStatusFeasible
    )
    return _Raw(
        values=np.asarray(h.getSolution().col_value) if has_solution else None,
        proven=status == highspy.HighsModelStatus.kOptimal,
        status=h.modelStatusToString(status),
    )


def _solve_cuopt(model: ScheduleModel, time_limit: float, gap_rel: float | None) -> _Raw:
    from cuopt.linear_programming import (  # pyright: ignore[reportMissingImports]
        data_model,
        solver,
        solver_settings,
    )
    from cuopt.linear_programming.solver.solver_wrapper import (  # pyright: ignore[reportMissingImports]
        ErrorStatus,
        MILPTerminationStatus,
    )

    dm = data_model.DataModel()
    dm.set_csr_constraint_matrix(model.values, model.indices, model.offsets)
    dm.set_constraint_lower_bounds(model.row_lower)
    dm.set_constraint_upper_bounds(model.row_upper)
    dm.set_objective_coefficients(model.cost)
    dm.set_variable_lower_bounds(model.col_lower)
    dm.set_variable_upper_bounds(model.col_upper)
    dm.set_variable_types(np.where(model.integer, "I", "C"))

    with _quiet_stdout():
        settings = solver_settings.SolverSettings()
        settings.set_parameter("log_to_console", False)
        settings.set_parameter("time_limit", float(time_limit))
        settings.set_parameter("mip_relative_gap", float(gap_rel or 0.0))
        settings.set_parameter("mip_absolute_gap", _ABS_GAP)
        solution = solver.Solve(dm, settings)
    if solution.get_error_status() != ErrorStatus.Success:
        raise RuntimeError(solution.get_error_message())
    status = solution.get_termination_status()
    values = np.asarray(solution.get_primal_solution(), dtype=np.float64)
    return _Raw(
        values=values if len(values) else None,
        proven=status == MILPTerminationStatus.Optimal,
        status=MILPTerminationStatus(status).name,
    )


@functools.cache
def cuopt_available() -> bool:
    """True if cuOpt can actually solve here: installed, with a usable GPU and driver.

    Solves a one-variable problem once, which also pays CUDA's start-up cost up front.
    """
    tiny = ScheduleModel(
        x_keys=(),
        gap_keys=(),
        cost=np.ones(1),
        col_lower=np.zeros(1),
        col_upper=np.ones(1),
        integer=np.ones(1, dtype=np.bool_),
        row_lower=np.ones(1),
        row_upper=np.full(1, np.inf),
        offsets=np.array([0, 1], np.int32),
        indices=np.zeros(1, np.int32),
        values=np.ones(1),
    )
    try:
        return _solve_cuopt(tiny, time_limit=10, gap_rel=None).proven
    except Exception:
        return False


def classify(model: ScheduleModel, values: FloatArray | None, proven: bool) -> Outcome:
    """Classify a solver's answer by checking it ourselves rather than trusting status codes.

    A result is usable only if every column has an integral, in-bounds value and every row
    holds; it is optimal only if the solver also proved it.
    """
    if values is None or not model.is_feasible(values):
        return Outcome.NO_SOLUTION
    return Outcome.OPTIMAL if proven else Outcome.FEASIBLE


_SOLVERS = {Backend.CPU: _solve_highs, Backend.GPU: _solve_cuopt}


def solve(
    problem: Problem,
    backend: Backend = Backend.CPU,
    time_limit: float = 30,
    gap_rel: float | None = None,
) -> SolveResult:
    """Solve `problem`. `gap_rel` lets the solver stop once it is provably within that
    fraction of the best possible objective (None = prove optimal); the result then
    counts as optimal."""
    start = time.perf_counter()
    model = build_model(problem)
    build_seconds = time.perf_counter() - start
    warnings: list[str] = []

    if backend is Backend.GPU and not cuopt_available():
        warnings.append("cuOpt is not available; solved on CPU instead.")
        backend = Backend.CPU

    start = time.perf_counter()
    try:
        raw = _SOLVERS[backend](model, time_limit, gap_rel)
    except Exception as exc:
        if backend is Backend.CPU:
            raise
        warnings.append(f"cuOpt failed ({exc}); solved on CPU instead.")
        backend = Backend.CPU
        start = time.perf_counter()
        raw = _SOLVERS[backend](model, time_limit, gap_rel)
    elapsed = time.perf_counter() - start

    outcome = classify(model, raw.values, raw.proven)
    assignments: tuple[Assignment, ...] = ()
    objective = None
    if outcome is not Outcome.NO_SOLUTION:
        assert raw.values is not None
        chosen = np.flatnonzero(raw.values[: len(model.x_keys)] > 0.5)
        assignments = tuple(
            sorted(
                Assignment(day=d, shift=s, employee_id=emp)
                for emp, s, d in (model.x_keys[i] for i in chosen)
            )
        )
        # Recomputed from the rounded solution, so both backends report it the same way.
        objective = float(model.cost @ np.round(raw.values, 6))
    return SolveResult(
        outcome=outcome,
        solver_status=raw.status,
        backend=backend,
        solve_seconds=elapsed,
        objective=objective,
        assignments=assignments,
        warnings=tuple(warnings),
        build_seconds=build_seconds,
    )
