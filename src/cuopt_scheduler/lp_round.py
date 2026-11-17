"""A heuristic that can run its heavy part on a GPU LP solver: solve the LP relaxation, then
round it into a valid schedule.

For this model the LP bound equals the integer optimum (see the README), so a good LP
solution is close to a good schedule. Rounding never breaks a rule because every rule is a
packing row (a sum of assignments <= a limit): assignments are only ever added while every
packing row they touch still has room. Demand rows have a slack column that absorbs any
shortfall.

LP engines: HiGHS on the CPU (dual simplex or PDLP) and cuOpt's PDLP on the GPU.

`solve_warm_started` goes one step further on the GPU: it hands the rounded schedule to
cuOpt's MILP solver as a starting point, which then only has to prove it (or improve it).
"""

from __future__ import annotations

import time
from enum import StrEnum

import numpy as np

from cuopt_scheduler.data import Problem
from cuopt_scheduler.model import FloatArray, ScheduleModel, build_model
from cuopt_scheduler.solver import (
    Backend,
    RawSolution,
    SolveResult,
    _quiet_stdout,
    _solve_cuopt,
    cuopt_data_model,
    highs_model,
    make_result,
)

CUOPT_PDLP_PARAMS: dict[str, int | float] = {
    "method": 1,  # PDLP
    # Stable2 was about twice as fast as 26.10's default (Stable3) on this model, with the
    # same rounded schedules (docs/benchmarks/README.md).
    "pdlp_solver_mode": 1,
}


class LPEngine(StrEnum):
    HIGHS = "highs"
    """CPU: HiGHS's default LP method (dual simplex)."""
    HIGHS_PDLP = "highs-pdlp"
    """CPU: HiGHS's first-order PDLP."""
    CUOPT_PDLP = "cuopt-pdlp"
    """GPU: cuOpt's first-order PDLP."""

    @property
    def backend(self) -> Backend:
        return Backend.GPU if self is LPEngine.CUOPT_PDLP else Backend.CPU


def solve_lp(model: ScheduleModel, engine: LPEngine, time_limit: float) -> FloatArray:
    """The LP relaxation's (approximate, for PDLP) optimal solution."""
    if engine.backend is Backend.CPU:
        h = highs_model(model, time_limit, relax=True)
        if engine is LPEngine.HIGHS_PDLP:
            h.setOptionValue("solver", "pdlp")
        h.run()
        return np.asarray(h.getSolution().col_value)

    from cuopt.linear_programming import solver, solver_settings  # pyright: ignore[reportMissingImports]

    dm = cuopt_data_model(model, relax=True)
    with _quiet_stdout():
        settings = solver_settings.SolverSettings()
        settings.set_parameter("log_to_console", False)
        settings.set_parameter("time_limit", float(time_limit))
        for name, value in CUOPT_PDLP_PARAMS.items():
            settings.set_parameter(name, value)
        solution = solver.Solve(dm, settings)
    return np.asarray(solution.get_primal_solution(), dtype=np.float64)


def round_lp(model: ScheduleModel, lp: FloatArray) -> FloatArray:
    """Turn an LP solution into a valid 0/1 schedule, preferring high LP values.

    Requires the model's structure: rows with a finite lower bound are demand rows, each with
    its own slack column; all other rows are packing rows with non-negative coefficients.
    """
    n_x = len(model.x_keys)
    rows_of = np.repeat(np.arange(model.num_rows), np.diff(model.offsets))
    on_x = model.indices < n_x
    cover = np.isfinite(model.row_lower)

    # Column-wise view of the x columns: the packing rows each one touches.
    pack = on_x & ~cover[rows_of]
    order = np.argsort(model.indices[pack], kind="stable")
    pack_rows = rows_of[pack][order].tolist()
    pack_vals = model.values[pack][order].tolist()
    pack_ptr = np.searchsorted(model.indices[pack][order], np.arange(n_x + 1)).tolist()

    # Row-wise view of demand rows: their x columns, best LP value first (then cheapest).
    room = model.row_upper.tolist()  # remaining capacity in packing rows
    x = np.zeros(model.num_cols)
    lp_x = np.clip(lp[:n_x], 0.0, 1.0)
    for r in np.flatnonzero(cover):
        start, stop = model.offsets[r], model.offsets[r + 1]
        cols = model.indices[start:stop]
        cols = cols[cols < n_x]
        cols = cols[model.col_upper[cols] > 0]
        cols = cols[np.lexsort((model.cost[cols], -lp_x[cols]))].tolist()
        need = model.row_lower[r]
        for j in cols:
            if need <= 0:
                break
            span = range(pack_ptr[j], pack_ptr[j + 1])
            if all(room[pack_rows[k]] >= pack_vals[k] for k in span):
                for k in span:
                    room[pack_rows[k]] -= pack_vals[k]
                x[j] = 1.0
                need -= 1
    # Slack: whatever demand is still unmet.
    activity = model.row_activity(x)
    is_slack = ~on_x
    slack_rows, slack_cols = rows_of[is_slack], model.indices[is_slack]
    x[slack_cols] = np.maximum(0.0, model.row_lower[slack_rows] - activity[slack_rows])
    return x


def solve_lp_round(
    problem: Problem, engine: LPEngine = LPEngine.HIGHS, time_limit: float = 30
) -> SolveResult:
    """Schedule `problem` by LP relaxation plus rounding. Never claims optimality."""
    start = time.perf_counter()
    model = build_model(problem)
    build_seconds = time.perf_counter() - start

    start = time.perf_counter()
    values = round_lp(model, solve_lp(model, engine, time_limit))
    elapsed = time.perf_counter() - start
    objective = float(model.cost @ values)
    raw = RawSolution(values, False, f"LP ({engine}) + rounding", ((elapsed, objective),))
    return _checked(make_result(model, raw, engine.backend, elapsed, build_seconds))


def solve_warm_started(
    problem: Problem, time_limit: float = 30, gap_rel: float | None = None
) -> SolveResult:
    """cuOpt PDLP + rounding, then cuOpt's MILP solver starting from that schedule.

    Has a good schedule as soon as the LP is solved, and can still prove it optimal. The
    time limit covers both steps.
    """
    start = time.perf_counter()
    model = build_model(problem)
    build_seconds = time.perf_counter() - start

    start = time.perf_counter()
    rounded = round_lp(model, solve_lp(model, LPEngine.CUOPT_PDLP, time_limit))
    t_lp = time.perf_counter() - start
    remaining = max(1.0, time_limit - t_lp)
    raw = _solve_cuopt(model, remaining, gap_rel, initial=rounded)
    elapsed = time.perf_counter() - start
    trace = ((t_lp, float(model.cost @ rounded)), *((t_lp + t, o) for t, o in raw.incumbents))
    if raw.values is None:  # cuOpt returned nothing better: keep the rounded schedule
        raw = RawSolution(rounded, False, raw.status, trace)
    else:
        raw = RawSolution(raw.values, raw.proven, raw.status, trace)
    return _checked(make_result(model, raw, Backend.GPU, elapsed, build_seconds))


def _checked(result: SolveResult) -> SolveResult:
    if not result.feasible:
        raise AssertionError("rounding produced an invalid schedule")  # a bug, not bad luck
    return result
